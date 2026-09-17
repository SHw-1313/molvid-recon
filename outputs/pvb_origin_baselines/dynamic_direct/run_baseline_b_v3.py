#!/usr/bin/env python3
"""Run the output-only Baseline B runner with the one-step shape fix."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


MODULE_NAME = "pvb_origin_baseline_runner"


def load_runner():
    path = Path(__file__).with_name("run_baseline_b.py")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def fixed_one_step_metrics(pred, target, record):
    runner = sys.modules[MODULE_NAME]
    # The original output-only runner passes prediction[1:] through an
    # additional unsqueeze.  Normalize [1,1,N,3] to [1,N,3] here.
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    if target.ndim == 4 and target.shape[0] == 1:
        target = target[0]
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"one-step tensors must be [T,N,3], got {pred.shape} and {target.shape}")
    mask = runner._valid_mask(record, pred.device)
    return {
        "rmsd": runner._frame_rmsd(pred, target, mask, [0]),
        "drmsd": runner._frame_drmsd(pred, target, mask, [0]),
        "bond_rmse": runner._bond_rmse(pred, target, record, mask, [0]),
        "contact_error": runner._contact_error(pred, target, record, mask, [0]),
        "clash_rate": runner._clash_rate(pred, record, mask, [0]),
    }


if __name__ == "__main__":
    origin_root = Path(os.environ.get("PVB_ORIGIN_ROOT", os.getcwd())).resolve()
    sys.path.insert(0, str(origin_root))
    runner = load_runner()
    runner.one_step_metrics = fixed_one_step_metrics
    raise SystemExit(runner.main())
