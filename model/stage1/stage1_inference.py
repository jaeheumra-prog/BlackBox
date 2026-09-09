"""Submission-compatible Stage 1 inference entry point."""

from __future__ import annotations

import json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stage1_features import (
    FeatureConfig,
    decode_stratified_views,
    extract_from_frames,
    extract_video,
    iter_videos,
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


class Stage1ForensicEnsemble:
    def __init__(self, checkpoint_path: str | Path):
        # best.npz에는 신경망 가중치가 아니라 정규화 통계와 여러 선형 분류기의 계수/threshold가 저장되어 있다.
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        self.means = checkpoint["means"].astype(np.float32)
        self.scales = checkpoint["scales"].astype(np.float32)
        self.coefficients = checkpoint["coefficients"].astype(np.float32)
        self.intercepts = checkpoint["intercepts"].astype(np.float32)
        self.member_weights = (
            checkpoint["member_weights"].astype(np.float32)
            if "member_weights" in checkpoint.files
            else np.full(len(self.intercepts), 1.0 / len(self.intercepts), dtype=np.float32)
        )
        self.member_weights /= np.sum(self.member_weights)
        self.threshold = float(checkpoint["threshold"][0])
        self.uncertainty_band = (
            checkpoint["uncertainty_band"].astype(np.float32)
            if "uncertainty_band" in checkpoint.files
            else None
        )
        config = json.loads(str(checkpoint["config"][0]))
        self.config = FeatureConfig(**config)

    def probability(self, aggregate: np.ndarray) -> tuple[float, float]:
        # 영상 특징을 학습 때의 분포로 정규화한 뒤 여러 선형 모델의 확률을
        # member_weights로 가중 평균한다.
        dimension = self.coefficients.shape[1]
        aggregate = np.asarray(aggregate, dtype=np.float32)[:dimension]
        normalized = np.clip((aggregate[None, :] - self.means) / self.scales, -8.0, 8.0)
        logits = np.sum(normalized * self.coefficients, axis=1) + self.intercepts
        probabilities = _sigmoid(logits)
        mean = float(np.sum(probabilities * self.member_weights))
        variance = float(np.sum(self.member_weights * np.square(probabilities - mean)))
        return mean, float(np.sqrt(max(variance, 0.0)))


def _logit_average(first: float, second: float) -> float:
    values = np.clip(np.asarray([first, second], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    return float(_sigmoid(np.asarray([np.mean(logits)]))[0])


def _fast_probability(path: Path, model: Stage1ForensicEnsemble):
    # Decode both interleaved views in one pass.  Feature extraction for the
    # second view is only paid when the first prediction is near the threshold.
    views = decode_stratified_views(path, model.config.frames, views=2)
    _, aggregate = extract_from_frames(views[0], model.config)
    first, first_uncertainty = model.probability(aggregate)
    if model.uncertainty_band is None:
        return first, first_uncertainty
    lower, upper = (float(value) for value in model.uncertainty_band)
    if not lower <= first <= upper:
        return first, first_uncertainty
    _, second_aggregate = extract_from_frames(views[1], model.config)
    second, second_uncertainty = model.probability(second_aggregate)
    return _logit_average(first, second), max(first_uncertainty, second_uncertainty, abs(first - second) / 2.0)


def _predict_one(path: Path, model: Stage1ForensicEnsemble, is_fast: bool):
    started = time.perf_counter()
    try:
        if is_fast:
            probability, uncertainty = _fast_probability(path, model)
        else:
            _, aggregate = extract_video(path, replace(model.config, frames=min(model.config.frames, 2), patches=1))
            probability, uncertainty = model.probability(aggregate)
        answer = "RERECORDED" if probability >= model.threshold else "ORIGINAL"
    except Exception as exc:
        print(f"[stage1] video={path.name} fallback=RERECORDED error={exc!r}", flush=True)
        # A corrupt/unsupported video cannot establish direct-capture
        # authenticity.  Keep the submission alive and choose the conservative
        # fallback used by the original sequential implementation.
        probability, uncertainty, answer = 1.0, 0.0, "RERECORDED"
    print(f"[stage1] video={path.name} seconds={time.perf_counter()-started:.2f}", flush=True)
    return {
        "ID": path.stem,
        "answer": answer,
        "_probability": probability,
        "_uncertainty": uncertainty,
    }


def predict_stage1(data_dir, model_dir):
    """Return columns [ID, answer] for every video under data_dir/videos."""

    root = Path(data_dir) / "videos"
    model_root = Path(model_dir)
    fast_checkpoint = model_root / "best_fast_8x3.npz"
    checkpoint = fast_checkpoint if fast_checkpoint.is_file() else model_root / "best.npz"
    model = Stage1ForensicEnsemble(checkpoint)
    is_fast = model.uncertainty_band is not None
    paths = list(iter_videos(root))
    # Stage 1 files are independent.  A small thread pool lets OpenCV/NumPy
    # overlap decode and feature extraction on the evaluator's 7 vCPUs while
    # keeping one model copy and deterministic input order.
    workers = min(4, max(1, os.cpu_count() or 1), len(paths) or 1)
    if workers > 1:
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda path: _predict_one(path, model, is_fast), paths))
    else:
        rows = [_predict_one(path, model, is_fast) for path in paths]
    frame = pd.DataFrame(rows, columns=["ID", "answer", "_probability", "_uncertainty"])
    return frame[["ID", "answer"]]


def predict_stage1_with_diagnostics(data_dir, model_dir):
    root = Path(data_dir) / "videos"
    model_root = Path(model_dir)
    checkpoint = (
        model_root / "best_fast_8x3.npz"
        if (model_root / "best_fast_8x3.npz").is_file()
        else model_root / "best.npz"
    )
    model = Stage1ForensicEnsemble(checkpoint)
    rows = []
    for path in iter_videos(root):
        if model.uncertainty_band is not None:
            probability, uncertainty = _fast_probability(path, model)
        else:
            _, aggregate = extract_video(path, model.config)
            probability, uncertainty = model.probability(aggregate)
        rows.append(
            {
                "ID": path.stem,
                "answer": "RERECORDED" if probability >= model.threshold else "ORIGINAL",
                "probability_rerecorded": probability,
                "ensemble_uncertainty": uncertainty,
            }
        )
    return pd.DataFrame(rows)
