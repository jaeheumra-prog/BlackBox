"""Submission-compatible Stage 1 inference entry point."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stage1_features import FeatureConfig, extract_video, iter_videos


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


class Stage1ForensicEnsemble:
    def __init__(self, checkpoint_path: str | Path):
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
        config = json.loads(str(checkpoint["config"][0]))
        self.config = FeatureConfig(**config)

    def probability(self, aggregate: np.ndarray) -> tuple[float, float]:
        normalized = np.clip((aggregate[None, :] - self.means) / self.scales, -8.0, 8.0)
        logits = np.sum(normalized * self.coefficients, axis=1) + self.intercepts
        probabilities = _sigmoid(logits)
        mean = float(np.sum(probabilities * self.member_weights))
        variance = float(np.sum(self.member_weights * np.square(probabilities - mean)))
        return mean, float(np.sqrt(max(variance, 0.0)))


def predict_stage1(data_dir, model_dir):
    """Return columns [ID, answer] for every video under data_dir/videos."""

    root = Path(data_dir) / "videos"
    model = Stage1ForensicEnsemble(Path(model_dir) / "best.npz")
    # The forensic classifier was trained with 24 uniformly sampled frames,
    # but its aggregate statistics are stable with a smaller sample.  The
    # submission runner may contain many more clips than the local demo set;
    # cap decode/feature work to keep the official 60-minute budget safe while
    # retaining the original spatial/FFT feature scales.
    inference_config = replace(model.config, frames=min(model.config.frames, 8))
    rows = []
    for path in iter_videos(root):
        try:
            _, aggregate = extract_video(path, inference_config)
            probability, uncertainty = model.probability(aggregate)
            answer = "RERECORDED" if probability >= model.threshold else "ORIGINAL"
        except Exception:
            # A corrupt/unsupported video cannot establish direct-capture authenticity.
            probability, uncertainty, answer = 1.0, 0.0, "RERECORDED"
        rows.append(
            {
                "ID": path.stem,
                "answer": answer,
                "_probability": probability,
                "_uncertainty": uncertainty,
            }
        )
    frame = pd.DataFrame(rows, columns=["ID", "answer", "_probability", "_uncertainty"])
    return frame[["ID", "answer"]]


def predict_stage1_with_diagnostics(data_dir, model_dir):
    root = Path(data_dir) / "videos"
    model = Stage1ForensicEnsemble(Path(model_dir) / "best.npz")
    rows = []
    for path in iter_videos(root):
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
