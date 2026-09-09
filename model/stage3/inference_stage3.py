"""Submission-compatible Stage 3 inference from 10 Hz forward video only."""

from __future__ import annotations

from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn


ACCEL_LABELS = ("ACCELERATING", "CONSTANT", "DECELERATING", "STOPPED")
STEER_LABELS = ("LEFT", "STRAIGHT", "RIGHT")
MOTION_FEATURES = (
    "speed_proxy",
    "accel_proxy",
    "divergence_smooth",
    "flow_mag_p90_smooth",
    "turn_proxy",
    "turn_rotation_proxy",
    "side_asymmetry",
    "affine_tx_smooth",
)


class Stage3MotionGRU(nn.Module):
    def __init__(self, motion_features: int = 8) -> None:
        super().__init__()
        self.motion_gru = nn.GRU(
            motion_features, 64, num_layers=2, batch_first=True, dropout=0.10
        )
        self.fusion = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.SiLU())
        self.accel_head = nn.Linear(128, len(ACCEL_LABELS))
        self.steer_head = nn.Linear(128, len(STEER_LABELS))
        self.aux_head = nn.Linear(128, 2)

    def forward(self, motion: torch.Tensor):
        sequence, _ = self.motion_gru(motion)
        hidden = self.fusion(sequence[:, -1])
        return self.accel_head(hidden), self.steer_head(hidden)


def _robust_affine_features(flow: np.ndarray, step: int = 4) -> dict[str, float]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[0:height:step, 0:width:step]
    sampled = flow[0:height:step, 0:width:step]
    x = (xx.ravel() - (width - 1) / 2) / max(width, 1)
    y = (yy.ravel() - (height - 1) / 2) / max(height, 1)
    u = sampled[..., 0].ravel()
    v = sampled[..., 1].ravel()
    magnitude = np.hypot(u, v)
    finite = np.isfinite(magnitude)
    if finite.sum() < 20:
        return {key: 0.0 for key in ("affine_tx", "affine_ty", "divergence", "rotation")}
    cap = np.quantile(magnitude[finite], 0.90)
    keep = finite & (magnitude <= cap)
    design = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
    coef_u, *_ = np.linalg.lstsq(design, u[keep], rcond=None)
    coef_v, *_ = np.linalg.lstsq(design, v[keep], rcond=None)
    return {
        "affine_tx": float(coef_u[2]),
        "affine_ty": float(coef_v[2]),
        "divergence": float(coef_u[0] + coef_v[1]),
        "rotation": float(0.5 * (coef_v[0] - coef_u[1])),
    }


def _flow_features(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(
        previous, current, None, 0.5, 3, 21, 3, 7, 1.5, 0
    )
    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.hypot(u, v)
    midpoint = flow.shape[1] // 2
    bottom = magnitude[int(flow.shape[0] * 0.55) :, :]
    return {
        "flow_x_median": float(np.median(u)),
        "flow_y_median": float(np.median(v)),
        "flow_mag_median": float(np.median(magnitude)),
        "flow_mag_p75": float(np.quantile(magnitude, 0.75)),
        "flow_mag_p90": float(np.quantile(magnitude, 0.90)),
        "bottom_mag_p75": float(np.quantile(bottom, 0.75)),
        "left_mag_median": float(np.median(magnitude[:, :midpoint])),
        "right_mag_median": float(np.median(magnitude[:, midpoint:])),
        **_robust_affine_features(flow),
    }


def _extract_motion(video_path: Path, width: int = 320, workers: int | None = None) -> np.ndarray:
    # OpenCV's image-operation thread limit does not limit FFmpeg decoders.
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open Stage 3 video: {video_path.name}")
    records = []
    previous = None
    zero_names = (
        "flow_x_median", "flow_y_median", "flow_mag_median", "flow_mag_p75",
        "flow_mag_p90", "bottom_mag_p75", "left_mag_median", "right_mag_median",
        "affine_tx", "affine_ty", "divergence", "rotation",
    )
    workers = min(4 if workers is None else max(1, workers), max(1, os.cpu_count() or 1))
    pending = deque()
    # Optical flow depends on a pair of images, not the preceding flow result.
    # Bound the queue so a long video retains at most a few small ROIs.
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height = round(frame.shape[0] * width / frame.shape[1])
                gray = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
                roi = gray[int(height * 0.36) : int(height * 0.90), int(width * 0.04) : int(width * 0.96)].copy()
                if previous is None:
                    records.append({name: 0.0 for name in zero_names})
                else:
                    pending.append(executor.submit(_flow_features, previous, roi))
                previous = roi
                if len(pending) >= workers * 2:
                    records.append(pending.popleft().result())
            records.extend(future.result() for future in pending)
    finally:
        capture.release()
    if not records:
        raise RuntimeError(f"No decodable frames in Stage 3 video: {video_path.name}")

    data = pd.DataFrame.from_records(records)
    for column in tuple(data.columns):
        data[f"{column}_smooth"] = data[column].rolling(9, center=True, min_periods=1).median()
    data["speed_proxy"] = data["bottom_mag_p75_smooth"]
    data["accel_proxy"] = data["speed_proxy"].rolling(11, center=True, min_periods=3).mean().diff(10)
    data["turn_proxy"] = data["flow_x_median_smooth"]
    data["turn_rotation_proxy"] = data["rotation_smooth"]
    data["side_asymmetry"] = data["right_mag_median_smooth"] - data["left_mag_median_smooth"]
    return data[list(MOTION_FEATURES)].fillna(0).to_numpy(dtype=np.float32)


def _smooth(probability: np.ndarray, window: int = 11) -> np.ndarray:
    return pd.DataFrame(probability).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _video_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 3 video directory not found: {root}")
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def predict_stage3(data_dir, model_dir):
    """Return [ID, sample_index, accel_label, steer_label] for every 10 Hz frame."""
    cv2.setNumThreads(1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    )
    candidates = ("best.pt", "best_submission.pt", "best_finetune.pt")
    model_path = next(
        (Path(model_dir) / name for name in candidates if (Path(model_dir) / name).is_file()),
        Path(model_dir) / candidates[0],
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["motion_names"]) != MOTION_FEATURES:
        raise ValueError("Stage 3 checkpoint feature order does not match inference code")
    model = Stage3MotionGRU(len(MOTION_FEATURES))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    mean = np.asarray(checkpoint["motion_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["motion_std"], dtype=np.float32)
    if mean.shape != (len(MOTION_FEATURES),) or std.shape != (len(MOTION_FEATURES),):
        raise ValueError("Stage 3 checkpoint normalization shape is invalid")
    std = np.maximum(np.abs(std), 1e-6)
    context = int(checkpoint["window_samples"])
    motion_width = int(checkpoint.get("motion_extraction_width", 320))
    if motion_width < 128:
        raise ValueError("Stage 3 motion extraction width is implausibly small")
    rows = []

    paths = _video_paths(Path(data_dir) / "videos")
    video_workers = min(2, max(1, os.cpu_count() or 1), len(paths) or 1)
    flow_workers = max(1, min(4, os.cpu_count() or 1) // video_workers)
    # At most two independent videos are decoded ahead. Their normalization,
    # rolling windows and GRU contexts remain strictly separate.
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=video_workers) as decoder:
        pending = deque(
            decoder.submit(_extract_motion, path, motion_width, flow_workers)
            for path in paths[:video_workers]
        )
        for video_index, video_path in enumerate(paths):
            started = time.perf_counter()
            raw_motion = pending.popleft().result()
            next_index = video_index + video_workers
            if next_index < len(paths):
                pending.append(decoder.submit(_extract_motion, paths[next_index], motion_width, flow_workers))
            motion = (raw_motion - mean) / std
            frame_count = len(motion)
            offsets = np.arange(context - 1, -1, -1)
            accel_parts, steer_parts = [], []
            for start in range(0, frame_count, 256):
                targets = np.arange(start, min(frame_count, start + 256))
                indices = np.clip(targets[:, None] - offsets[None, :], 0, frame_count - 1)
                batch = torch.from_numpy(motion[indices]).to(device, non_blocking=True)
                accel_logits, steer_logits = model(batch)
                accel_parts.append(torch.softmax(accel_logits, 1).cpu().numpy())
                steer_parts.append(torch.softmax(steer_logits, 1).cpu().numpy())
            print(f"[stage3] video={video_path.name} frames={frame_count} wait_and_infer_seconds={time.perf_counter()-started:.2f}", flush=True)
            accel_probability = _smooth(np.concatenate(accel_parts), 11)
            steer_probability = _smooth(np.concatenate(steer_parts), 11)
            accel_prediction = np.asarray(ACCEL_LABELS)[accel_probability.argmax(axis=1)]
            steer_prediction = np.asarray(STEER_LABELS)[steer_probability.argmax(axis=1)]
            rows.extend({
                "ID": video_path.stem,
                "sample_index": sample_index,
                "accel_label": str(accel),
                "steer_label": str(steer),
            } for sample_index, (accel, steer) in enumerate(zip(accel_prediction, steer_prediction)))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
