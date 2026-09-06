from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_COLUMNS = [
    "ID",
    "collision_frame",
    "entry_frame",
    "evasion_space",
    "entry_side",
]


def frame_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else 0


def sorted_frame_paths(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (frame_number(path), path.name.lower()),
    )


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-6)


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    return np.convolve(values, np.array([0.2, 0.6, 0.2]), mode="same")


def _collision_score_gray(gray: list[np.ndarray]) -> np.ndarray:
    count = len(gray)
    if count <= 2:
        return np.arange(count, dtype=np.float32)
    residual = np.zeros(count, dtype=np.float32)
    flow_p90 = np.zeros(count, dtype=np.float32)
    flow_spread = np.zeros(count, dtype=np.float32)
    shift_jerk = np.zeros(count, dtype=np.float32)
    blur_drop = np.zeros(count, dtype=np.float32)
    laplacian = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in gray])
    shifts: list[np.ndarray] = [np.zeros(2, dtype=np.float32)]

    for index in range(1, count):
        previous, current = gray[index - 1], gray[index]
        shift, _ = cv2.phaseCorrelate(previous.astype(np.float32), current.astype(np.float32))
        dx, dy = float(shift[0]), float(shift[1])
        if abs(dx) > previous.shape[1] * 0.2 or abs(dy) > previous.shape[0] * 0.2:
            dx = dy = 0.0
        shifts.append(np.array([dx, dy], dtype=np.float32))
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            previous,
            transform,
            (previous.shape[1], previous.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )
        difference = cv2.absdiff(aligned, current)[current.shape[0] // 5 :]
        residual[index] = float(np.percentile(difference, 75))
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.linalg.norm(flow, axis=2)[flow.shape[0] // 4 :]
        flow_p90[index] = float(np.percentile(magnitude, 90))
        flow_spread[index] = float(np.percentile(magnitude, 90) - np.median(magnitude))
        blur_drop[index] = max(0.0, float(laplacian[index - 1] - laplacian[index]))

    shift_array = np.vstack(shifts)
    shift_jerk[2:] = np.linalg.norm(np.diff(shift_array, n=2, axis=0), axis=1)
    time_prior = np.linspace(-0.8, 0.6, count, dtype=np.float32)
    score = (
        0.28 * np.clip(_robust_z(_smooth(residual)), -2.0, 6.0)
        + 0.30 * np.clip(_robust_z(_smooth(flow_p90)), -2.0, 6.0)
        + 0.22 * np.clip(_robust_z(_smooth(flow_spread)), -2.0, 6.0)
        + 0.12 * np.clip(_robust_z(_smooth(shift_jerk)), -2.0, 6.0)
        + 0.08 * np.clip(_robust_z(_smooth(blur_drop)), -2.0, 6.0)
        + time_prior
    )
    score[: max(2, round(count * 0.30))] = -np.inf
    return score


def collision_score(frames_bgr: list[np.ndarray], working_width: int = 320) -> np.ndarray:
    """Camera-motion-compensated impact score for every frame transition."""
    if not frames_bgr:
        return np.empty(0, dtype=np.float32)
    working_height = max(
        1, round(frames_bgr[0].shape[0] * working_width / frames_bgr[0].shape[1])
    )
    gray = [
        cv2.cvtColor(
            cv2.resize(frame, (working_width, working_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in frames_bgr
    ]
    return _collision_score_gray(gray)


def collision_score_from_paths(
    paths: list[Path], working_width: int = 320
) -> tuple[list[Path], np.ndarray]:
    valid_paths: list[Path] = []
    gray: list[np.ndarray] = []
    working_height: int | None = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if working_height is None:
            working_height = max(1, round(image.shape[0] * working_width / image.shape[1]))
        gray.append(
            cv2.resize(
                image,
                (working_width, working_height),
                interpolation=cv2.INTER_AREA,
            )
        )
        valid_paths.append(path)
    return valid_paths, _collision_score_gray(gray)


def estimate_collision_index(frames_bgr: list[np.ndarray]) -> tuple[int, np.ndarray]:
    score = collision_score(frames_bgr)
    peak = int(np.argmax(score))
    # The strongest deformation/camera shock usually peaks 1--2 frames after
    # physical contact. This correction was selected on direct-ego public clips.
    index = max(0, peak - 2)
    return index, score


def _letterbox(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, (ratio, pad_x, pad_y, resized_width, resized_height)


def _preprocess(frame_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
    canvas, geometry = _letterbox(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), size)
    tensor = canvas.astype(np.float32) / 255.0
    tensor[..., 0] = (tensor[..., 0] - 0.485) / 0.229
    tensor[..., 1] = (tensor[..., 1] - 0.456) / 0.224
    tensor[..., 2] = (tensor[..., 2] - 0.406) / 0.225
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), geometry


def _restore_boxes(
    boxes: np.ndarray,
    geometry: tuple[float, int, int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    ratio, pad_x, pad_y, _, _ = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    return boxes


def _restore_mask(
    output: np.ndarray,
    geometry: tuple[float, int, int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    _, pad_x, pad_y, resized_width, resized_height = geometry
    cropped = output[:, :, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width]
    mask = np.argmax(cropped, axis=1)[0].astype(np.uint8)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-6)


@dataclass
class TrackRecord:
    frame_index: int
    box: np.ndarray
    confidence: float
    boundaries: Any = None


@dataclass
class Track:
    identifier: int
    records: list[TrackRecord] = field(default_factory=list)

    @property
    def last(self) -> TrackRecord:
        return self.records[-1]


class GreedyTracker:
    """Small dependency-free tracker tuned for short accident clips."""

    def __init__(self, maximum_gap: int = 5):
        self.maximum_gap = maximum_gap
        self.tracks: list[Track] = []
        self._next_identifier = 1

    def update(self, frame_index: int, boxes: np.ndarray, boundaries: Any) -> None:
        available = [
            track for track in self.tracks if frame_index - track.last.frame_index <= self.maximum_gap
        ]
        proposals: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(boxes):
            box = detection[:4]
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            detection_area = max(1.0, float(np.prod(box[2:] - box[:2])))
            for track_index, track in enumerate(available):
                last_box = track.last.box
                last_center = np.array(
                    [(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2]
                )
                predicted_center = last_center
                if len(track.records) >= 2:
                    previous_box = track.records[-2].box
                    previous_center = np.array(
                        [
                            (previous_box[0] + previous_box[2]) / 2,
                            (previous_box[1] + previous_box[3]) / 2,
                        ]
                    )
                    elapsed = max(1, track.last.frame_index - track.records[-2].frame_index)
                    gap = frame_index - track.last.frame_index
                    predicted_center = last_center + (last_center - previous_center) * gap / elapsed
                overlap = _iou(box, last_box)
                last_area = max(1.0, float(np.prod(last_box[2:] - last_box[:2])))
                scale_ratio = detection_area / last_area
                normalizer = max(
                    20.0,
                    0.5
                    * (
                        float(np.linalg.norm(box[2:] - box[:2]))
                        + float(np.linalg.norm(last_box[2:] - last_box[:2]))
                    ),
                )
                distance = float(np.linalg.norm(center - predicted_center)) / normalizer
                scale_consistent = 0.28 <= scale_ratio <= 3.6
                plausible = overlap >= 0.08 or (
                    distance <= 0.48 and scale_consistent
                )
                similarity = 2.5 * overlap - 0.55 * distance - 0.08 * abs(np.log(scale_ratio))
                if plausible and similarity >= -0.18:
                    proposals.append((similarity, detection_index, track_index))

        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for _, detection_index, track_index in sorted(proposals, reverse=True):
            if detection_index in used_detections or track_index in used_tracks:
                continue
            detection = boxes[detection_index]
            available[track_index].records.append(
                TrackRecord(frame_index, detection[:4].copy(), float(detection[4]), boundaries)
            )
            used_detections.add(detection_index)
            used_tracks.add(track_index)

        for detection_index, detection in enumerate(boxes):
            if detection_index in used_detections:
                continue
            track = Track(self._next_identifier)
            self._next_identifier += 1
            track.records.append(
                TrackRecord(frame_index, detection[:4].copy(), float(detection[4]), boundaries)
            )
            self.tracks.append(track)


class YoloPBackend:
    def __init__(self, model_dir: Path, size: int = 640):
        self.model_dir = Path(model_dir)
        self.size = size
        yolop_root = self.model_dir / "yolop"
        if not yolop_root.is_dir():
            # Development workspace layout.
            yolop_root = self.model_dir / "YOLOP-main" / "YOLOP-main"
        sys.path.insert(0, str(yolop_root))
        try:
            from lib.config import cfg
            from lib.core.general import non_max_suppression
            from lib.models import get_net
        finally:
            sys.path.remove(str(yolop_root))

        self.non_max_suppression = non_max_suppression
        # CUDA can report as available while all devices are masked (for
        # example in a CPU-only validation worker); verify an actual device.
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        )
        self.model = get_net(cfg)
        weight_path = self.model_dir / "End-to-end.pth"
        if not weight_path.is_file():
            weight_path = yolop_root / "weights" / "End-to-end.pth"
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device).eval()
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()
            torch.backends.cudnn.benchmark = True
        warmup = torch.zeros(
            (1, 3, size, size),
            device=self.device,
            dtype=torch.float16 if self.use_half else torch.float32,
        )
        with torch.inference_mode():
            self.model(warmup)

    def infer(
        self, frame: np.ndarray, confidence_threshold: float = 0.18
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        array, geometry = _preprocess(frame, self.size)
        tensor = torch.from_numpy(array).to(self.device)
        if self.use_half:
            tensor = tensor.half()
        with torch.inference_mode():
            detection_bundle, drivable_tensor, lane_tensor = self.model(tensor)
        detections = self.non_max_suppression(
            detection_bundle[0].float(),
            conf_thres=confidence_threshold,
            iou_thres=0.45,
        )[0]
        if detections is None or len(detections) == 0:
            boxes = np.empty((0, 6), dtype=np.float64)
        else:
            boxes = detections.cpu().numpy().astype(np.float64)
            boxes = _restore_boxes(boxes, geometry, width, height)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[(widths >= 6) & (heights >= 6)]
        drivable = _restore_mask(drivable_tensor.float().cpu().numpy(), geometry, width, height)
        lane = _restore_mask(lane_tensor.float().cpu().numpy(), geometry, width, height)
        return boxes, drivable, lane


def _fallback_boundaries(y: float, width: int, height: int) -> tuple[float, float]:
    vanish_y = 0.32 * height
    fraction = float(np.clip((y - vanish_y) / max(1.0, 0.66 * height), 0.0, 1.0))
    center = 0.50 * width
    half_width = width * (0.055 + 0.36 * fraction)
    return center - half_width, center + half_width


def _boundary_values(record: TrackRecord, width: int, height: int) -> tuple[float, float]:
    y = float(record.box[3] - 0.04 * (record.box[3] - record.box[1]))
    if record.boundaries is not None:
        left = float(record.boundaries.left_x(np.asarray(y)))
        right = float(record.boundaries.right_x(np.asarray(y)))
        if left < 0.5 * width < right and right - left >= 0.08 * width:
            return left, right
    return _fallback_boundaries(y, width, height)


def _track_risk(track: Track, collision_index: int, width: int, height: int) -> float:
    records = [record for record in track.records if record.frame_index <= collision_index + 3]
    if len(records) < 2:
        return -1.0
    first, last = records[0], records[-1]
    x1, y1, x2, y2 = last.box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    first_area = max(
        1.0,
        (first.box[2] - first.box[0]) * (first.box[3] - first.box[1]),
    )
    center_x = (x1 + x2) / 2
    first_center = (first.box[0] + first.box[2]) / 2
    scale = min(1.0, np.sqrt(area / (width * height)) / 0.30)
    center = float(np.exp(-abs(center_x - width / 2) / (0.34 * width)))
    bottom = float(np.clip((y2 / height - 0.25) / 0.65, 0.0, 1.0))
    recency = float(np.exp(-max(0, collision_index - last.frame_index) / 6.0))
    growth = float(np.clip(np.log(area / first_area + 1e-6) / 1.5, 0.0, 1.0))
    inward = float(
        np.clip((abs(first_center - width / 2) - abs(center_x - width / 2)) / (0.25 * width), 0, 1)
    )
    persistence = min(1.0, len(records) / 8.0)
    return float(
        0.36 * scale
        + 0.12 * center
        + 0.14 * bottom
        + 0.27 * recency
        + 0.06 * growth
        + 0.03 * inward
        + 0.02 * persistence
    )


def select_victim(
    tracks: list[Track], collision_index: int, width: int, height: int
) -> tuple[Track | None, list[tuple[int, float]]]:
    plausible = [
        track
        for track in tracks
        if len(track.records) >= 2 and track.last.frame_index >= collision_index - 6
    ]
    if not plausible:
        plausible = tracks
    ranking = sorted(
        (
            (track.identifier, _track_risk(track, collision_index, width, height))
            for track in plausible
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranking or ranking[0][1] < 0:
        return None, ranking
    identifier = ranking[0][0]
    return next(track for track in tracks if track.identifier == identifier), ranking


def infer_entry(
    victim: Track | None,
    collision_index: int,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[int, str, dict[str, Any]]:
    if victim is None or not victim.records:
        return max(0, collision_index - max(1, round(frame_count * 0.20))), "RIGHT", {
            "reason": "no_vehicle_detection"
        }
    records = [record for record in victim.records if record.frame_index <= collision_index]
    if not records:
        records = victim.records[:1]
    initial_centers = [float((r.box[0] + r.box[2]) / 2) for r in records[: min(3, len(records))]]
    initial_center = float(np.median(initial_centers))
    final_center = float((records[-1].box[0] + records[-1].box[2]) / 2)
    lateral_motion = final_center - initial_center
    # Motion direction is more reliable than the current screen position: a car
    # travelling rightward entered the ego path from the left, and vice versa.
    if abs(lateral_motion) >= 0.035 * width:
        side = "LEFT" if lateral_motion > 0 else "RIGHT"
    elif abs(initial_center - width / 2) >= 0.06 * width:
        side = "RIGHT" if initial_center > width / 2 else "LEFT"
    else:
        side = "RIGHT" if final_center >= width / 2 else "LEFT"

    signed: list[float] = []
    for record in records:
        x1, _, x2, y2 = record.box
        box_width = x2 - x1
        wheel_x = x1 + 0.15 * box_width if side == "RIGHT" else x2 - 0.15 * box_width
        left, right = _boundary_values(record, width, height)
        signed.append(float(right - wheel_x if side == "RIGHT" else wheel_x - left))

    # Require two consecutive contacts when possible; it suppresses one-frame lane-mask jumps.
    entry_index: int | None = None
    for position, value in enumerate(signed):
        next_inside = position + 1 >= len(signed) or signed[position + 1] >= -0.01 * width
        if value >= -0.01 * width and next_inside:
            entry_index = records[position].frame_index
            break

    reason = "observed_wheel_contact"
    if entry_index is None:
        indices = np.asarray([record.frame_index for record in records], dtype=np.float64)
        values = np.asarray(signed, dtype=np.float64)
        tail = min(6, len(records))
        if tail >= 2:
            slope, intercept = np.polyfit(indices[-tail:], values[-tail:], 1)
            crossing = -intercept / slope if slope > 0.25 else float(collision_index)
            entry_index = int(round(np.clip(crossing, records[0].frame_index, collision_index)))
            reason = "extrapolated_wheel_contact"
        else:
            entry_index = min(collision_index, records[-1].frame_index)
            reason = "single_detection_fallback"
    entry_index = int(np.clip(entry_index, 0, min(collision_index, frame_count - 1)))
    return entry_index, side, {
        "reason": reason,
        "track_id": victim.identifier,
        "signed_distances": signed,
        "record_indices": [record.frame_index for record in records],
    }


def _road_interval(mask: np.ndarray | None, y: int, width: int) -> tuple[float, float]:
    if mask is None:
        return 0.08 * width, 0.92 * width
    height = mask.shape[0]
    y1 = int(np.clip(y - 0.03 * height, 0, height - 1))
    y2 = int(np.clip(y + 0.03 * height, y1 + 1, height))
    occupancy = np.mean(mask[y1:y2] > 0, axis=0) >= 0.25
    indices = np.flatnonzero(occupancy)
    if len(indices) < 0.18 * width:
        return 0.08 * width, 0.92 * width
    gaps = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, gaps)
    center = width / 2
    containing = [run for run in runs if len(run) and run[0] <= center <= run[-1]]
    selected = max(containing or runs, key=len)
    if len(selected) < 0.18 * width:
        return 0.08 * width, 0.92 * width
    return float(selected[0]), float(selected[-1])


def infer_evasion(
    victim: Track | None,
    collision_index: int,
    drivable_mask: np.ndarray | None,
    width: int,
    height: int,
) -> tuple[int, dict[str, float]]:
    if victim is None or not victim.records:
        return 0, {"reason": 0.0}
    record = min(victim.records, key=lambda item: abs(item.frame_index - collision_index))
    x1, y1, x2, y2 = (float(value) for value in record.box)
    road_left, road_right = _road_interval(drivable_mask, int(y2), width)
    obstacle_width = max(1.0, x2 - x1)
    left_gap = max(0.0, x1 - road_left)
    right_gap = max(0.0, road_right - x2)
    required = max(0.16 * width, 0.72 * obstacle_width)
    available = int(max(left_gap, right_gap) >= required)
    return available, {
        "road_left": road_left,
        "road_right": road_right,
        "left_gap": left_gap,
        "right_gap": right_gap,
        "required_gap": required,
    }


class Stage2Predictor:
    def __init__(self, model_dir: Path, model_size: int = 640):
        self.model_dir = Path(model_dir)
        self.model_size = model_size
        self.backend = YoloPBackend(self.model_dir, model_size)

        try:
            from ego_lane_geometry import EgoLaneBoundaryTracker
        except ImportError:
            sys.path.insert(0, str(self.model_dir))
            try:
                from ego_lane_geometry import EgoLaneBoundaryTracker
            finally:
                sys.path.remove(str(self.model_dir))
        self.boundary_tracker_type = EgoLaneBoundaryTracker

    def predict_folder(self, folder: Path) -> tuple[dict[str, object], dict[str, Any]]:
        paths, impact_score = collision_score_from_paths(sorted_frame_paths(folder))
        if not paths:
            raise ValueError(f"No readable images in {folder}")
        impact_peak = int(np.argmax(impact_score))
        collision_index = max(0, impact_peak - 2)
        first_frame = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise ValueError(f"No readable images in {folder}")
        height, width = first_frame.shape[:2]

        tracker = GreedyTracker(maximum_gap=5)
        boundary_tracker = self.boundary_tracker_type(smoothing=0.70, maximum_missing_frames=4)
        collision_mask: np.ndarray | None = None
        last_mask: np.ndarray | None = None
        maximum_index = min(len(paths) - 1, collision_index + 3)
        for index, path in enumerate(paths[: maximum_index + 1]):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            boxes, drivable, lane = self.backend.infer(frame)
            boundaries = boundary_tracker.update(lane)
            tracker.update(index, boxes, boundaries)
            last_mask = drivable
            if index == collision_index:
                collision_mask = drivable
        if collision_mask is None:
            collision_mask = last_mask

        victim, ranking = select_victim(tracker.tracks, collision_index, width, height)
        entry_index, entry_side, entry_debug = infer_entry(
            victim, collision_index, len(paths), width, height
        )
        evasion_space, evasion_debug = infer_evasion(
            victim, collision_index, collision_mask, width, height
        )
        collision_index = int(np.clip(collision_index, 0, len(paths) - 1))
        entry_index = int(np.clip(entry_index, 0, collision_index))
        row = {
            "ID": folder.name,
            "collision_frame": int(frame_number(paths[collision_index])),
            "entry_frame": int(frame_number(paths[entry_index])),
            "evasion_space": int(evasion_space),
            "entry_side": entry_side,
        }
        debug = {
            "frame_count": len(paths),
            "collision_index": collision_index,
            "impact_peak_index": impact_peak,
            "victim_ranking": ranking[:10],
            "entry": entry_debug,
            "evasion": evasion_debug,
        }
        return row, debug

    def predict(self, data_dir: Path) -> pd.DataFrame:
        image_root = Path(data_dir) / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Stage 2 image directory not found: {image_root}")
        folders = sorted(path for path in image_root.iterdir() if path.is_dir())
        rows = [self.predict_folder(folder)[0] for folder in folders]
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


_PREDICTOR_CACHE: dict[str, Stage2Predictor] = {}


def predict_stage2(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    key = str(Path(model_dir).resolve())
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = Stage2Predictor(Path(model_dir))
        _PREDICTOR_CACHE[key] = predictor
    return predictor.predict(Path(data_dir))
