"""Content-suppressed forensic features for Stage 1 video recapture detection.

The extractor deliberately avoids file size, codec name, FOURCC, and path metadata.
Those values perfectly separate the ten public demonstration videos but are not
causal evidence of a physical screen recapture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


EPS = 1e-6


@dataclass(frozen=True)
class FeatureConfig:
    frames: int = 24
    patch_size: int = 192
    fft_size: int = 128
    temporal_width: int = 320
    use_temporal: bool = True


def _safe_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.10)),
        float(np.quantile(values, 0.90)),
    )


def decode_uniform(path: str | Path, count: int = 24) -> list[np.ndarray]:
    """Decode approximately uniform BGR frames without trusting video metadata."""

    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if 0 < total <= 300:
        wanted = np.linspace(0, max(0, total - 1), min(count, total)).round().astype(int)
        wanted_set = set(wanted.tolist())
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted_set:
                frames.append(frame)
            index += 1
    elif total > 300:
        wanted = np.linspace(0, total - 1, count).round().astype(int)
        for index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    else:
        # Unknown-length streams: bounded sequential decoding.
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    capture.release()

    if not frames:
        raise ValueError(f"cannot decode frames: {path}")
    if len(frames) < count:
        positions = np.linspace(0, len(frames) - 1, count).round().astype(int)
        frames = [frames[int(i)] for i in positions]
    return frames[:count]


def _patches(image: np.ndarray, patch_size: int) -> list[np.ndarray]:
    height, width = image.shape[:2]
    size = min(patch_size, height, width)
    ys = [0, max(0, (height - size) // 2), max(0, height - size)]
    xs = [0, max(0, (width - size) // 2), max(0, width - size)]
    positions = [
        (ys[1], xs[1]),
        (ys[0], xs[0]),
        (ys[0], xs[2]),
        (ys[2], xs[0]),
        (ys[2], xs[2]),
    ]
    return [image[y : y + size, x : x + size] for y, x in positions]


def _one_dimensional_peak(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 8:
        return 0.0
    profile = profile - cv2.GaussianBlur(profile[:, None], (1, 0), 2.0).ravel()
    spectrum = np.abs(np.fft.rfft(profile * np.hanning(len(profile))))
    spectrum = spectrum[2:]
    if spectrum.size == 0:
        return 0.0
    return float(np.quantile(spectrum, 0.98) / (np.mean(spectrum) + EPS))


def _spectral_metrics(channel: np.ndarray, fft_size: int) -> list[float]:
    channel = cv2.resize(channel, (fft_size, fft_size), interpolation=cv2.INTER_AREA)
    channel = channel.astype(np.float32)
    residual = channel - cv2.GaussianBlur(channel, (0, 0), 2.0)
    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual * window[:, None] * window[None, :])))
    spectrum = np.log1p(spectrum)

    fy, fx = np.mgrid[-0.5:0.5:complex(fft_size), -0.5:0.5:complex(fft_size)]
    radius = np.sqrt(fx * fx + fy * fy)
    mid_mask = (radius >= 0.04) & (radius < 0.20)
    high_mask = (radius >= 0.20) & (radius < 0.46)
    valid_mask = mid_mask | high_mask

    def band(mask: np.ndarray) -> tuple[float, float, float]:
        values = spectrum[mask]
        mean = float(np.mean(values))
        peak = float((np.quantile(values, 0.995) - mean) / (np.std(values) + EPS))
        probabilities = np.maximum(values, 0.0) + EPS
        probabilities /= np.sum(probabilities)
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities)))
        return mean, peak, entropy

    mid_mean, mid_peak, mid_entropy = band(mid_mask)
    high_mean, high_peak, high_entropy = band(high_mask)
    axis_mask = valid_mask & ((np.abs(fx) < 2.0 / fft_size) | (np.abs(fy) < 2.0 / fft_size))
    axis_ratio = float(np.mean(spectrum[axis_mask]) / (np.mean(spectrum[valid_mask]) + EPS))
    return [mid_mean, mid_peak, mid_entropy, high_mean, high_peak, high_entropy, axis_ratio]


def _single_patch_features(patch_bgr: np.ndarray, fft_size: int) -> np.ndarray:
    patch = patch_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(patch)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    rg = r - g
    bg = b - g

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    features: list[float] = [
        float(np.mean(y)),
        float(np.std(y)),
        float(np.mean(np.abs(rg))),
        float(np.mean(np.abs(bg))),
        float(np.mean(gradient)),
        float(np.std(gradient)),
        float(np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + EPS)),
    ]

    residuals = []
    for sigma in (0.7, 1.5, 3.0):
        residual = y - cv2.GaussianBlur(y, (0, 0), sigma)
        residuals.append(residual)
        absolute = np.abs(residual)
        features.extend(
            [float(np.mean(absolute)), float(np.std(residual)), float(np.quantile(absolute, 0.95))]
        )

    fine = residuals[0]
    features.extend(
        [
            _one_dimensional_peak(np.mean(fine, axis=1)),
            _one_dimensional_peak(np.mean(fine, axis=0)),
        ]
    )

    for opponent in (rg, bg):
        high = opponent - cv2.GaussianBlur(opponent, (0, 0), 1.2)
        features.extend([float(np.mean(np.abs(high))), float(np.std(high))])

    rb_high = (r - b) - cv2.GaussianBlur(r - b, (0, 0), 1.2)
    rg_high = rg - cv2.GaussianBlur(rg, (0, 0), 1.2)
    correlation = np.corrcoef(rb_high.ravel(), rg_high.ravel())[0, 1]
    features.append(float(correlation) if np.isfinite(correlation) else 0.0)
    features.extend(_spectral_metrics(y, fft_size))
    features.extend(_spectral_metrics(rg, fft_size)[1::3])
    features.extend(_spectral_metrics(bg, fft_size)[1::3])
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _frame_features(frame: np.ndarray, config: FeatureConfig) -> np.ndarray:
    patch_features = np.stack(
        [_single_patch_features(patch, config.fft_size) for patch in _patches(frame, config.patch_size)]
    )
    # Mean captures a global weak trace; maximum captures a localized screen trace.
    return np.concatenate([np.mean(patch_features, axis=0), np.max(patch_features, axis=0)]).astype(
        np.float32
    )


def _small_gray(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _temporal_features(frames: Sequence[np.ndarray], width: int) -> np.ndarray:
    gray = [_small_gray(frame, width) for frame in frames]
    luminance = np.asarray([float(np.mean(item)) for item in gray], dtype=np.float32)
    differences = np.asarray(
        [float(np.mean(np.abs(gray[i] - gray[i - 1]))) for i in range(1, len(gray))], dtype=np.float32
    )

    shifts = []
    responses = []
    for previous, current in zip(gray[:-1], gray[1:]):
        previous_blur = cv2.GaussianBlur(previous, (0, 0), 2.0)
        current_blur = cv2.GaussianBlur(current, (0, 0), 2.0)
        shift, response = cv2.phaseCorrelate(previous_blur, current_blur)
        shifts.append([shift[0], shift[1]])
        responses.append(response)
    shifts_array = np.asarray(shifts, dtype=np.float32) if shifts else np.zeros((1, 2), np.float32)
    shift_step = np.linalg.norm(shifts_array, axis=1)
    shift_jitter = (
        np.linalg.norm(np.diff(shifts_array, axis=0), axis=1)
        if len(shifts_array) > 1
        else np.zeros(1, np.float32)
    )

    detrended = luminance - cv2.GaussianBlur(luminance[:, None], (1, 0), 1.5).ravel()
    flicker = np.abs(np.fft.rfft(detrended))[1:]
    flicker_peak = float(np.max(flicker) / (np.mean(flicker) + EPS)) if flicker.size else 0.0

    values: list[float] = []
    for series in (luminance, differences, shift_step, shift_jitter, np.asarray(responses, np.float32)):
        values.extend(_safe_stats(series))
    values.extend(
        [
            flicker_peak,
            float(np.mean(differences < 0.002)) if differences.size else 0.0,
            float(np.mean(np.abs(np.diff(differences)))) if differences.size > 1 else 0.0,
        ]
    )
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def aggregate_sequence(sequence: np.ndarray, temporal: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    aggregate = np.concatenate(
        [
            np.mean(sequence, axis=0),
            np.std(sequence, axis=0),
            np.quantile(sequence, 0.10, axis=0),
            np.quantile(sequence, 0.90, axis=0),
            temporal,
        ]
    )
    return np.nan_to_num(aggregate.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_from_frames(
    frames: Sequence[np.ndarray], config: FeatureConfig = FeatureConfig()
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("at least one frame is required")
    positions = np.linspace(0, len(frames) - 1, config.frames).round().astype(int)
    sampled = [frames[int(index)] for index in positions]
    sequence = np.stack([_frame_features(frame, config) for frame in sampled])
    temporal = (
        _temporal_features(sampled, config.temporal_width)
        if config.use_temporal
        else np.zeros(23, dtype=np.float32)
    )
    return sequence, aggregate_sequence(sequence, temporal)


def extract_video(
    path: str | Path, config: FeatureConfig = FeatureConfig()
) -> tuple[np.ndarray, np.ndarray]:
    return extract_from_frames(decode_uniform(path, config.frames), config)


def iter_videos(root: str | Path) -> Iterable[Path]:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
