#!/usr/bin/env python3
"""Compatibility wrapper for the output-only Baseline B runner."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_runner():
    path = Path(__file__).with_name("run_baseline_b.py")
    spec = importlib.util.spec_from_file_location("pvb_origin_baseline_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_one_step_metrics(pred, target, record):
    runner = sys.modules["pvb_origin_baseline_runner"]
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
    runner = _load_runner()
    sys.modules["pvb_origin_baseline_runner"] = runner
    runner.one_step_metrics = _fixed_one_step_metrics
    raise SystemExit(runner.main())
