from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoLaneBoundaries:
    left_coefficients: np.ndarray
    right_coefficients: np.ndarray
    image_width: int
    image_height: int
    y_min: int
    y_max: int

    def _normalized_y(self, y: np.ndarray | float) -> np.ndarray:
        denominator = max(1, self.y_max - self.y_min)
        return (y.astype(np.float64) - self.y_min) / denominator

    def left_x(self, y: np.ndarray | float) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64)
        return np.polyval(self.left_coefficients, self._normalized_y(values))

    def right_x(self, y: np.ndarray | float) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64)
        return np.polyval(self.right_coefficients, self._normalized_y(values))

    def polylines(self, points: int = 80) -> tuple[np.ndarray, np.ndarray]:
        y = np.linspace(self.y_min, self.y_max, points)
        left = np.column_stack((self.left_x(y), y))
        right = np.column_stack((self.right_x(y), y))
        left[:, 0] = left[:, 0].clip(0, self.image_width - 1)
        right[:, 0] = right[:, 0].clip(0, self.image_width - 1)
        return np.rint(left).astype(np.int32), np.rint(right).astype(np.int32)


def _run_centers(indices: np.ndarray, maximum_gap: int = 5) -> np.ndarray:
    if len(indices) == 0:
        return np.empty(0, dtype=np.float64)
    split_points = np.flatnonzero(np.diff(indices) > maximum_gap) + 1
    runs = np.split(indices, split_points)
    return np.asarray([float(np.mean(run)) for run in runs if len(run)], dtype=np.float64)


def _robust_quadratic_fit(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    if len(y) < 18:
        return None

    keep = np.ones(len(y), dtype=bool)
    coefficients: np.ndarray | None = None
    for _ in range(4):
        if int(np.count_nonzero(keep)) < 18:
            return None
        coefficients = np.polyfit(y[keep], x[keep], deg=2)
        residuals = np.abs(x - np.polyval(coefficients, y))
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        threshold = max(18.0, median + 3.0 * max(mad, 1.0))
        keep = residuals <= threshold
    return coefficients


def estimate_ego_lane_boundaries(lane_mask: np.ndarray) -> EgoLaneBoundaries | None:
    image_height, image_width = lane_mask.shape
    center_x = image_width / 2.0
    y_min = int(round(image_height * 0.32))
    y_max = int(round(image_height * 0.96))
    denominator = max(1, y_max - y_min)

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    for y in range(y_min, y_max + 1, 3):
        centers = _run_centers(np.flatnonzero(lane_mask[y] > 0))
        if len(centers) == 0:
            continue
        minimum_center_offset = max(8.0, 0.012 * image_width)
        left = centers[centers < center_x - minimum_center_offset]
        right = centers[centers > center_x + minimum_center_offset]
        normalized_y = (y - y_min) / denominator
        if len(left):
            left_points.append((normalized_y, float(np.max(left))))
        if len(right):
            right_points.append((normalized_y, float(np.min(right))))

    if len(left_points) < 18 or len(right_points) < 18:
        return None

    left_array = np.asarray(left_points, dtype=np.float64)
    right_array = np.asarray(right_points, dtype=np.float64)
    left_coefficients = _robust_quadratic_fit(left_array[:, 0], left_array[:, 1])
    right_coefficients = _robust_quadratic_fit(right_array[:, 0], right_array[:, 1])
    if left_coefficients is None or right_coefficients is None:
        return None

    estimate = EgoLaneBoundaries(
        left_coefficients=left_coefficients,
        right_coefficients=right_coefficients,
        image_width=image_width,
        image_height=image_height,
        y_min=y_min,
        y_max=y_max,
    )
    validation_y = np.asarray(
        [image_height * 0.45, image_height * 0.65, image_height * 0.88],
        dtype=np.float64,
    )
    left_x = estimate.left_x(validation_y)
    right_x = estimate.right_x(validation_y)
    widths = right_x - left_x
    if np.any(left_x >= center_x) or np.any(right_x <= center_x):
        return None
    if np.any(widths < 0.08 * image_width) or np.any(widths > 0.95 * image_width):
        return None
    if widths[-1] < 0.75 * widths[0]:
        return None
    return estimate


class EgoLaneBoundaryTracker:
    def __init__(self, smoothing: float = 0.75, maximum_missing_frames: int = 10):
        self.smoothing = smoothing
        self.maximum_missing_frames = maximum_missing_frames
        self._last: EgoLaneBoundaries | None = None
        self._missing_frames = 0

    def update(self, lane_mask: np.ndarray) -> EgoLaneBoundaries | None:
        current = estimate_ego_lane_boundaries(lane_mask)
        if current is None:
            self._missing_frames += 1
            if self._missing_frames <= self.maximum_missing_frames:
                return self._last
            self._last = None
            return None

        self._missing_frames = 0
        if self._last is not None:
            current = EgoLaneBoundaries(
                left_coefficients=(
                    self.smoothing * self._last.left_coefficients
                    + (1.0 - self.smoothing) * current.left_coefficients
                ),
                right_coefficients=(
                    self.smoothing * self._last.right_coefficients
                    + (1.0 - self.smoothing) * current.right_coefficients
                ),
                image_width=current.image_width,
                image_height=current.image_height,
                y_min=current.y_min,
                y_max=current.y_max,
            )
        self._last = current
        return current


def footprint_lane_geometry(
    footprint: np.ndarray,
    boundaries: EgoLaneBoundaries,
) -> dict[str, float | str]:
    x = footprint[:, 0].astype(np.float64)
    y = footprint[:, 1].astype(np.float64)
    left_x = boundaries.left_x(y)
    right_x = boundaries.right_x(y)
    left_penetration = float(np.max(x - left_x))
    right_penetration = float(np.max(right_x - x))

    center = np.mean(footprint, axis=0)
    center_left = float(boundaries.left_x(center[1]))
    center_right = float(boundaries.right_x(center[1]))
    center_x = float(center[0])
    if center_x < center_left:
        lane_state = "LEFT"
        outside_distance = center_left - center_x
    elif center_x > center_right:
        lane_state = "RIGHT"
        outside_distance = center_x - center_right
    else:
        lane_state = "EGO"
        outside_distance = 0.0

    nearest_boundary_distance = min(
        abs(center_x - center_left), abs(center_right - center_x)
    )
    inside_vertices = np.logical_and(x >= left_x, x <= right_x)
    return {
        "lane_state": lane_state,
        "left_penetration_px": left_penetration,
        "right_penetration_px": right_penetration,
        "outside_distance_px": float(outside_distance),
        "nearest_boundary_distance_px": float(nearest_boundary_distance),
        "inside_vertex_fraction": float(np.mean(inside_vertices)),
    }
