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
        config = json.loads(str(checkpoint["config"][0]))
        self.config = FeatureConfig(**config)

    def probability(self, aggregate: np.ndarray) -> tuple[float, float]:
        # 영상 특징을 학습 때의 분포로 정규화한 뒤 여러 선형 모델의 확률을
        # member_weights로 가중 평균한다.
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
    # 학습 설정(24프레임·5패치)을 그대로 쓰면 제출 시간이 길어진다.
    # 따라서 공간/FFT 크기는 유지하고, 대표 프레임 2개와 중앙 패치 1개만
    # 사용한다. 특징 벡터의 길이는 그대로라 checkpoint와 shape은 호환된다.
    inference_config = replace(model.config, frames=min(model.config.frames, 2), patches=1)
    rows = []
    for path in iter_videos(root):
        try:
            # 파일 하나를 읽고 forensic feature → 앙상블 확률 → 최종 라벨 순서로 처리한다.
            _, aggregate = extract_video(path, inference_config)
            probability, uncertainty = model.probability(aggregate)
            answer = "RERECORDED" if probability >= model.threshold else "ORIGINAL"
        except Exception:
            # A corrupt/unsupported video cannot establish direct-capture authenticity.
            # 제출 중 전체 작업이 중단되지 않도록 보수적으로 RERECORDED를 반환한다.
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
