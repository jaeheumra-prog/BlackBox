"""Unified DACON three-stage inference entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predict_stage1(data_dir, model_dir):
    impl_dir = Path(model_dir)
    if not (impl_dir / "stage1_inference.py").is_file():
        impl_dir = impl_dir / "stage1"
    import sys
    sys.path.insert(0, str(impl_dir))
    try:
        return _load("stage1_impl", impl_dir / "stage1_inference.py").predict_stage1(data_dir, impl_dir)
    finally:
        sys.path.remove(str(impl_dir))


def predict_stage2(data_dir, model_dir):
    impl_dir = Path(model_dir)
    if not (impl_dir / "stage2_predictor.py").is_file():
        impl_dir = impl_dir / "stage2"
    return _load("stage2_impl", impl_dir / "stage2_predictor.py").predict_stage2(data_dir, impl_dir)


def predict_stage3(data_dir, model_dir):
    impl_dir = Path(model_dir)
    if not (impl_dir / "inference_stage3.py").is_file():
        impl_dir = impl_dir / "stage3"
    import sys
    sys.path.insert(0, str(impl_dir))
    try:
        return _load("stage3_impl", impl_dir / "inference_stage3.py").predict_stage3(data_dir, impl_dir)
    finally:
        sys.path.remove(str(impl_dir))
