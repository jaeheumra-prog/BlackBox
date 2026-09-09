"""DACON three-stage entry points with bounded CPU use and visible failures."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import time

# Set before importing NumPy/PyTorch. Four independent CPU tasks fit 7 vCPUs.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

_MODULES = {}


def _run(stage, filename, data_dir, model_dir):
    import cv2
    import torch
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    supplied = Path(model_dir).resolve()
    candidates = (supplied, supplied / stage)
    impl_dir = next((p for p in candidates if (p / filename).is_file()), None)
    if impl_dir is None:
        raise FileNotFoundError(f'{stage}: {filename} not found in {candidates}')
    key = (stage, str(impl_dir))
    start = time.perf_counter()
    print(f'[{stage}] start model={impl_dir}', flush=True)
    # Limit BLAS even if the evaluator imported NumPy before inference.py.
    with threadpool_limits(limits=1):
        sys.path.insert(0, str(impl_dir))
        try:
            if key not in _MODULES:
                name = f'_submission_{stage}_{len(_MODULES)}'
                spec = importlib.util.spec_from_file_location(name, impl_dir / filename)
                if spec is None or spec.loader is None:
                    raise ImportError(str(impl_dir / filename))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(name, None)
                    raise
                _MODULES[key] = module
            result = getattr(_MODULES[key], f'predict_{stage}')(data_dir, impl_dir)
        except Exception as exc:
            print(f'[{stage}] failed after {time.perf_counter()-start:.1f}s: {exc!r}', flush=True)
            raise
        finally:
            sys.path.remove(str(impl_dir))
    print(f'[{stage}] done rows={len(result)} seconds={time.perf_counter()-start:.2f}', flush=True)
    return result


def predict_stage1(data_dir, model_dir):
    return _run('stage1', 'stage1_inference.py', data_dir, model_dir)


def predict_stage2(data_dir, model_dir):
    return _run('stage2', 'stage2_predictor.py', data_dir, model_dir)


def predict_stage3(data_dir, model_dir):
    return _run('stage3', 'inference_stage3.py', data_dir, model_dir)
