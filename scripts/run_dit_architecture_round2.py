#!/usr/bin/env python
"""Run only Round 2 of the R4 DiT architecture-sequential experiment.

Round 2 inherits the equal-exposure winner from Round 1 and compares ordinary
RF continuation with the same continuation plus a calibrated, differentiable
future bond-length auxiliary.  The frozen codec is never updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset
from evaluation.motion_metrics import (
    MOTION_RATIO_THRESHOLD,
    motion_arm_report,
    paired_motion_report,
)
from module.dit_geometry_supervision import FutureBondAuxiliary
from scripts.run_state_detail_codec_v2_t1 import TrajectoryCappedBatchSampler
from trainer.dit_trainer import DiTTrainer, module_state_hash

from scripts import run_dit_architecture_sequential_v1 as common


ROUND = "round2"
ARMS = ("control", "candidate")
FFN_NORM_SOURCE = "post_adaln"


def _settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = cfg.get("round2")
    if not isinstance(value, Mapping):
        raise ValueError("round2 configuration is missing")
    required = (
        "max_atom_frame_tokens",
        "tau_threshold",
        "target_auxiliary_gradient_ratio",
        "lambda_min",
        "lambda_max",
        "future_rounds_training_reserve_gpu_hours",
        "calibration",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"round2 configuration lacks {missing}")
    calibration = value["calibration"]
    if not isinstance(calibration, Mapping):
        raise ValueError("round2 calibration configuration must be a mapping")
    result = dict(value)
    result["calibration"] = dict(calibration)
    if int(result["max_atom_frame_tokens"]) < 56672:
        raise ValueError("round2 max_atom_frame_tokens cannot reject the largest frozen train clip")
    if not 0.0 <= float(result["tau_threshold"]) <= 1.0:
        raise ValueError("round2 tau_threshold must lie in [0,1]")
    if not 0.0 < float(result["target_auxiliary_gradient_ratio"]) < 1.0:
        raise ValueError("round2 target auxiliary gradient ratio must lie in (0,1)")
    if not 0.0 < float(result["lambda_min"]) <= float(result["lambda_max"]):
        raise ValueError("round2 lambda bounds are invalid")
    histories = tuple(int(item) for item in result["calibration"].get("histories", ()))
    if histories != (4, 8):
        raise ValueError("round2 calibration must use the fixed H4/H8 train clips")
    if int(result["calibration"].get("min_applicable_batches", 0)) < 1:
        raise ValueError("round2 calibration needs a positive minimum applicable-batch count")
    return result


def _code_provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    files = (
        "scripts/run_dit_architecture_round2.py",
        "scripts/run_dit_architecture_sequential_v1.py",
        "module/dit_geometry_supervision.py",
        "trainer/dit_trainer.py",
        "evaluation/dit_architecture_sequential_v1.py",
        "config/dit_architecture_sequential_v1.yaml",
        "config/dit_architecture_sequential_v1.neibu.yaml",
        "tests/test_dit_architecture_sequential_cuda.py",
    )
    hashes = {
        relative: common.sha256_file(PROJECT_ROOT / relative)
        if (PROJECT_ROOT / relative).is_file()
        else None
        for relative in files
    }
    return {
        "head": head,
        "files": hashes,
        "source_hash": common._canonical_hash(hashes),
    }


def _round2_seed(ctx: Any) -> int:
    return int(ctx.cfg["evaluation"]["training_seed_base"]) + 2


def _round2_sampler(ctx: Any, *, seed: int) -> TrajectoryCappedBatchSampler:
    settings = _settings(ctx.cfg)
    sampler = TrajectoryCappedBatchSampler(
        ctx.data.train,
        max_tokens=int(settings["max_atom_frame_tokens"]),
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
        seed=int(seed),
        shuffle=False,
    )
    sampler.set_epoch(0)
    return sampler


def _round2_tokens_for_updates(ctx: Any, updates: int, seed: int) -> int:
    sampler = _round2_sampler(ctx, seed=seed)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    total = 0
    for update in range(int(updates)):
        indices, _ = common._next_batch(sampler, cursor, plan_cache)
        total += common._batch_tokens(specs, indices, histories[update % len(histories)])
    return total


def _round1_parent(ctx: Any) -> Path:
    decision_path = ctx.output_dir / "round1" / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Round 2 requires the completed Round 1 decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") not in ("KEEP", "REJECT", "TRADEOFF", "INCONCLUSIVE"):
        raise RuntimeError("Round 1 must reach a final keep/reject decision before Round 2")
    arm = decision.get("selected_arm")
    if arm not in ARMS:
        raise RuntimeError("Round 1 final decision lacks a selected equal-exposure arm")
    checkpoints = common._round1_checkpoints(ctx)
    path = checkpoints[str(arm)]
    selected = decision.get("selected_checkpoint")
    if selected != str(path) or decision.get("selected_checkpoint_sha256") != common.sha256_file(path):
        raise RuntimeError("Round 1 decision/checkpoint provenance is stale")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_contract", {}).get("ffn_norm_source") != FFN_NORM_SOURCE:
        raise RuntimeError("Round 2 requires the post-AdaLN Round 1 control lineage")
    return path


def _round2_metadata(
    *, arm: str, lambda_bond: float, settings: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "enabled": arm == "candidate",
        "objective": "rf_plus_future_endpoint_bond_mse" if arm == "candidate" else "rf_only",
        "lambda_bond": float(lambda_bond) if arm == "candidate" else 0.0,
        "tau_threshold": float(settings["tau_threshold"]),
        "target_auxiliary_gradient_ratio": float(settings["target_auxiliary_gradient_ratio"]),
        "coordinate_units": "angstrom",
        "topology": "registered_static_covalent_bonds_only",
        "future_target_clean": True,
    }


def _new_trainer(
    ctx: Any,
    *,
    state: Mapping[str, torch.Tensor],
    state_hash: str,
    max_steps: int,
    arm: str,
    lambda_bond: float,
    round_name: str = ROUND,
) -> DiTTrainer:
    trainer = common._new_trainer(
        ctx,
        ffn_norm_source=FFN_NORM_SOURCE,
        max_steps=int(max_steps),
        initial_state=state,
        initial_hash=state_hash,
        arm=arm,
        round_name=round_name,
    )
    trainer.config.metadata["geometry_supervision"] = _round2_metadata(
        arm=arm, lambda_bond=lambda_bond, settings=_settings(ctx.cfg)
    )
    return trainer


def _load_parent(ctx: Any) -> tuple[Path, Mapping[str, Any], dict[str, torch.Tensor], str]:
    path = _round1_parent(ctx)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state, state_hash = common._state_from_payload(
        ctx.cfg, payload, ctx.device, ffn_norm_source=FFN_NORM_SOURCE
    )
    return path, payload, state, state_hash


def _auxiliary(
    ctx: Any,
    trainer: DiTTrainer,
    coordinate_batch: Any,
    history: int,
    lambda_bond: float,
) -> FutureBondAuxiliary:
    return FutureBondAuxiliary(
        adapter=trainer.adapter,
        statistics=ctx.statistics_device,
        codec=ctx.codec.model,
        coordinate_batch=coordinate_batch,
        history_frames=int(history),
        lambda_bond=float(lambda_bond),
        tau_threshold=float(_settings(ctx.cfg)["tau_threshold"]),
    )


def _calibration_seed(base: int, sample_id: str, history: int) -> int:
    digest = hashlib.sha256(
        f"dit-architecture-round2-calibration|{int(base)}|{sample_id}|{int(history)}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _gradient_l2(
    loss: Tensor,
    parameters: Sequence[Tensor],
    *,
    retain_graph: bool,
) -> Tensor:
    if loss.ndim != 0 or not loss.requires_grad:
        raise RuntimeError("calibration loss does not retain a scalar gradient graph")
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return loss.detach().new_zeros((), dtype=torch.float32)
    return torch.stack(squares).sum().sqrt()


def _calibration_path(ctx: Any) -> Path:
    return ctx.output_dir / ROUND / "calibration.json"


def _calibrate(ctx: Any) -> dict[str, Any]:
    settings = _settings(ctx.cfg)
    parent_path, parent, state, parent_hash = _load_parent(ctx)
    parent_sha = common.sha256_file(parent_path)
    target_path = _calibration_path(ctx)
    settings_hash = common._canonical_hash(settings)
    if target_path.is_file():
        existing = json.loads(target_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "PASS"
            and existing.get("parent_checkpoint_sha256") == parent_sha
            and existing.get("round2_settings_hash") == settings_hash
        ):
            return existing
        raise RuntimeError("round2 calibration already exists with incompatible provenance")

    trainer = _new_trainer(
        ctx,
        state=state,
        state_hash=parent_hash,
        max_steps=int(parent["step"]) + 1,
        arm="calibration",
        lambda_bond=1.0,
        round_name="round2_calibration",
    )
    common._restore_parent(
        trainer,
        parent,
        expected_codec_hash=ctx.codec.codec_state_hash,
        expected_statistics_hash=ctx.statistics.hash,
    )
    trainer.model.eval()
    frozen_before = trainer.frozen_state_hashes()
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    items = list(selection["quick_train"])
    if len(items) != int(ctx.cfg["evaluation"]["quick_train_systems"]):
        raise RuntimeError("round2 calibration selection is not the frozen eight-train-system set")
    parameters = tuple(parameter for parameter in trainer.model.parameters() if parameter.requires_grad)
    rows = []
    for history in tuple(int(value) for value in settings["calibration"]["histories"]):
        for item in items:
            latent, batch, observed = common._prepare_batch(
                ctx,
                ctx.data.train,
                (int(item["dataset_index"]),),
                trainer.adapter,
                history,
            )
            del latent
            center = common._source_center(ctx, batch, observed, trainer.adapter, history)
            generator = torch.Generator(device=ctx.device).manual_seed(
                _calibration_seed(
                    int(settings["calibration"]["seed"]), str(item["sample_id"]), history
                )
            )
            normalized = trainer._normalise_batch(observed)
            auxiliary = _auxiliary(ctx, trainer, batch, history, lambda_bond=1.0)
            with trainer.autocast_context():
                sample = trainer.flow.sample(
                    normalized,
                    generator=generator,
                    source_center=center,
                    source_mode="conditional",
                )
                prediction = trainer.model(normalized.with_fields(sample.interpolated), sample.tau)
                rf = trainer.flow.loss(prediction, sample.target, normalized)
                bond = auxiliary.raw_loss(prediction, sample, normalized)
            if bond.applicable_samples:
                rf_norm = _gradient_l2(rf.total, parameters, retain_graph=True)
                bond_norm = _gradient_l2(bond.loss, parameters, retain_graph=False)
                if not bool(torch.isfinite(rf_norm)) or not bool(torch.isfinite(bond_norm)):
                    raise FloatingPointError("round2 calibration produced a non-finite gradient norm")
                if float(rf_norm) <= 0.0 or float(bond_norm) <= 0.0:
                    raise RuntimeError("round2 calibration produced a zero RF or bond gradient norm")
                raw_ratio = float((bond_norm / rf_norm).detach().cpu())
                rows.append(
                    {
                        "sample_id": str(item["sample_id"]),
                        "history_frames": history,
                        "tau": [float(value) for value in sample.tau.detach().cpu()],
                        "rf_loss": float(rf.total.detach().cpu()),
                        "bond_loss": float(bond.loss.detach().cpu()),
                        "rf_gradient_l2": float(rf_norm.detach().cpu()),
                        "bond_gradient_l2": float(bond_norm.detach().cpu()),
                        "raw_bond_to_rf_gradient_ratio": raw_ratio,
                        **bond.diagnostics(),
                    }
                )
            del batch, observed, normalized, center, auxiliary
            torch.cuda.empty_cache()
    minimum = int(settings["calibration"]["min_applicable_batches"])
    if len(rows) < minimum:
        raise RuntimeError(
            f"round2 calibration has only {len(rows)} applicable batches; need {minimum}"
        )
    ratios = [float(row["raw_bond_to_rf_gradient_ratio"]) for row in rows]
    unclamped = float(settings["target_auxiliary_gradient_ratio"]) / float(np.median(ratios))
    lambda_bond = min(
        max(unclamped, float(settings["lambda_min"])), float(settings["lambda_max"])
    )
    if any(parameter.grad is not None for parameter in ctx.codec.model.parameters()):
        raise RuntimeError("frozen codec unexpectedly accumulated gradients during calibration")
    if trainer.frozen_state_hashes() != frozen_before:
        raise RuntimeError("frozen codec/frame encoder changed during calibration")
    result = {
        "schema": f"{common.SCHEMA}.round2.calibration.v1",
        "status": "PASS",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "parent_model_state_hash": parent_hash,
        "round2_settings_hash": settings_hash,
        "calibration_selection": "fixed_selection.quick_train × H4/H8",
        "calibration_rows": rows,
        "applicable_batch_count": len(rows),
        "raw_bond_to_rf_gradient_ratio": {
            "median": float(np.median(ratios)),
            "minimum": float(min(ratios)),
            "maximum": float(max(ratios)),
        },
        "target_auxiliary_gradient_ratio": float(settings["target_auxiliary_gradient_ratio"]),
        "lambda_unclamped": unclamped,
        "lambda_bond": lambda_bond,
        "lambda_bounds": [float(settings["lambda_min"]), float(settings["lambda_max"])],
        "achieved_median_auxiliary_to_rf_gradient_ratio": float(lambda_bond * np.median(ratios)),
        "frozen_hashes": frozen_before,
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    common._write_json(target_path, result)
    return result


def _load_calibration(ctx: Any) -> dict[str, Any]:
    path = _calibration_path(ctx)
    if not path.is_file():
        raise FileNotFoundError("Round 2 requires a frozen train-only gradient calibration")
    result = json.loads(path.read_text(encoding="utf-8"))
    parent = _round1_parent(ctx)
    if result.get("status") != "PASS":
        raise RuntimeError("round2 calibration did not pass")
    if result.get("parent_checkpoint_sha256") != common.sha256_file(parent):
        raise RuntimeError("round2 calibration parent differs from the selected Round 1 control")
    if result.get("round2_settings_hash") != common._canonical_hash(_settings(ctx.cfg)):
        raise RuntimeError("round2 calibration settings differ from the frozen settings")
    value = float(result.get("lambda_bond", float("nan")))
    if not math.isfinite(value):
        raise RuntimeError("round2 calibration has no finite lambda_bond")
    return result


def _profile_generation(ctx: Any, trainer: DiTTrainer, indices: Sequence[int], history: int) -> dict[str, Any]:
    selected = (int(indices[0]),)
    _latent, batch, observed = common._prepare_batch(
        ctx, ctx.data.train, selected, trainer.adapter, history
    )
    center = common._source_center(ctx, batch, observed, trainer.adapter, history)
    normalized = ctx.statistics_device.normalize(observed)
    seed = common.generation_seed(
        int(ctx.cfg["evaluation"]["generation_seed"]), str(batch.sample_id[0]), history, 0
    )
    noise, _meta = common.make_fixed_noise(normalized, seed=seed)
    common._sync(ctx.device)
    started = time.perf_counter()
    generated, generation = common.generate_fixed_noise_latent(
        trainer.model,
        trainer.adapter,
        observed,
        ctx.statistics_device,
        noise=noise,
        steps=int(ctx.cfg["evaluation"]["euler_steps"]),
        source_center=center,
        source_mode="conditional",
    )
    decoded = ctx.codec.model.decode(generated).x_hat.float()
    metrics = common.reassessment_trajectory_metrics(decoded, batch.x.float(), batch, history)
    common._sync(ctx.device)
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(decoded).all()) or not bool(generation["observed_clamp_exact"]):
        raise RuntimeError("round2 profile generation produced an invalid decode")
    return {
        "seconds": elapsed,
        "sample_id": str(batch.sample_id[0]),
        "history_frames": history,
        "bond_rmse": metrics["future"]["bond_rmse"],
        "contact_f1": metrics["future"]["contact_f1"],
        "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
    }


def _profile(ctx: Any) -> dict[str, Any]:
    budget_path = ctx.output_dir / ROUND / "budget.json"
    if budget_path.is_file():
        existing = json.loads(budget_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            return existing
        raise RuntimeError("a non-passing round2 budget already exists; refusing to overwrite it")
    calibration = _load_calibration(ctx)
    lambda_bond = float(calibration["lambda_bond"])
    parent_path, parent, state, parent_hash = _load_parent(ctx)
    warmup = int(ctx.cfg["budget"]["profile_warmup_updates"])
    measured = int(ctx.cfg["budget"]["profile_measured_updates"])
    total = warmup + measured
    seed = _round2_seed(ctx)
    sampler = _round2_sampler(ctx, seed=seed)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    batches = [common._next_batch(sampler, cursor, plan_cache)[0] for _ in range(total)]
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    rows: dict[str, Any] = {}
    generator_states = {}
    for arm in ARMS:
        trainer = _new_trainer(
            ctx,
            state=state,
            state_hash=parent_hash,
            max_steps=int(parent["step"]) + total,
            arm=arm,
            lambda_bond=lambda_bond,
            round_name="round2_profile",
        )
        common._restore_parent(
            trainer,
            parent,
            expected_codec_hash=ctx.codec.codec_state_hash,
            expected_statistics_hash=ctx.statistics.hash,
        )
        generator = torch.Generator(device=ctx.device).manual_seed(seed)
        timings = []
        last_geometry: dict[str, Any] = {}
        torch.cuda.reset_peak_memory_stats(ctx.device)
        for update, indices in enumerate(batches):
            history = histories[update % len(histories)]
            _latent, batch, observed = common._prepare_batch(
                ctx, ctx.data.train, indices, trainer.adapter, history
            )
            center = common._source_center(ctx, batch, observed, trainer.adapter, history)
            auxiliary = (
                _auxiliary(ctx, trainer, batch, history, lambda_bond)
                if arm == "candidate"
                else None
            )
            common._sync(ctx.device)
            started = time.perf_counter()
            log = trainer.train_step(
                observed,
                generator=generator,
                source_center=center,
                auxiliary_objective=auxiliary,
            )
            common._sync(ctx.device)
            timings.append(time.perf_counter() - started)
            last_geometry = {
                key: value for key, value in log.items() if str(key).startswith("geometry_")
            }
        values = sorted(timings[warmup:])
        generation = _profile_generation(
            ctx, trainer, batches[warmup][0:1], histories[warmup % len(histories)]
        )
        rows[arm] = {
            "warmup_updates": warmup,
            "measured_updates": measured,
            "seconds": timings[warmup:],
            "p50_seconds": values[len(values) // 2],
            "p90_seconds": values[max(0, math.ceil(0.9 * len(values)) - 1)],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
            "last_geometry": last_geometry,
            "generation_evaluation": generation,
            "parent_step": int(parent["step"]),
        }
        generator_states[arm] = generator.get_state().detach().cpu().clone()
        del trainer
        torch.cuda.empty_cache()
    if not torch.equal(generator_states["control"], generator_states["candidate"]):
        raise RuntimeError("round2 profile tau/epsilon generators diverged")
    frozen_budget = ctx.cfg["budget"]
    total_seconds = float(frozen_budget["gpu_hours_total"]) * 3600.0
    evaluation_reserve = float(frozen_budget["reserve_fraction"]) * total_seconds
    future_rounds_reserve = float(_settings(ctx.cfg)["future_rounds_training_reserve_gpu_hours"]) * 3600.0
    round1_summary = json.loads((ctx.output_dir / "round1" / "train_summary.json").read_text(encoding="utf-8"))
    prior_seconds = float(round1_summary["gpu_hours"]) * 3600.0
    available = total_seconds - evaluation_reserve - future_rounds_reserve - prior_seconds
    pair_seconds_per_update = float(rows["control"]["p90_seconds"]) + float(rows["candidate"]["p90_seconds"])
    choices = []
    selected_updates = 0
    selected_extension = 0
    for candidate in tuple(int(item) for item in frozen_budget["round_candidate_updates"]):
        extension = int(math.ceil(candidate * float(frozen_budget["inconclusive_extension_fraction"])))
        estimate = pair_seconds_per_update * (candidate + extension)
        feasible = estimate <= available
        choices.append(
            {
                "added_updates_per_arm": candidate,
                "extension_updates_per_arm": extension,
                "estimated_pair_gpu_seconds_including_extension": estimate,
                "feasible": feasible,
            }
        )
        if feasible and not selected_updates:
            selected_updates, selected_extension = candidate, extension
    if not selected_updates:
        raise RuntimeError("no round2 paired exposure fits the frozen total budget and future-round reserve")
    round_tokens = _round2_tokens_for_updates(ctx, selected_updates, seed)
    extended_tokens = _round2_tokens_for_updates(ctx, selected_updates + selected_extension, seed)
    result = {
        "schema": f"{common.SCHEMA}.round2.budget.v1",
        "status": "PASS",
        "frozen_before_training": True,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": common.sha256_file(parent_path),
        "calibration": str(_calibration_path(ctx)),
        "calibration_sha256": common.sha256_file(_calibration_path(ctx)),
        "lambda_bond": lambda_bond,
        "profile": rows,
        "pair_p90_seconds_per_update": pair_seconds_per_update,
        "gpu_seconds_total": total_seconds,
        "evaluation_recovery_reserve_seconds": evaluation_reserve,
        "future_rounds_training_reserve_seconds": future_rounds_reserve,
        "prior_round_training_seconds": prior_seconds,
        "available_round2_training_seconds": available,
        "selected_round2_added_updates": selected_updates,
        "selected_round2_future_atom_frame_tokens_per_arm": round_tokens,
        "selected_round2_extension_added_updates": selected_extension,
        "selected_round2_extension_future_atom_frame_tokens_per_arm": extended_tokens - round_tokens,
        "selected_round2_extended_total_updates": selected_updates + selected_extension,
        "selected_round2_extended_total_future_atom_frame_tokens_per_arm": extended_tokens,
        "choices": choices,
        "max_atom_frame_tokens": int(_settings(ctx.cfg)["max_atom_frame_tokens"]),
        "device": common._cuda_info(ctx.device),
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    common._write_json(budget_path, result)
    common._write_json(ctx.output_dir / ROUND / "profile.json", result)
    return result


def _load_budget(ctx: Any) -> dict[str, Any]:
    path = ctx.output_dir / ROUND / "budget.json"
    if not path.is_file():
        raise FileNotFoundError("Round 2 profile/budget is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("frozen_before_training") is not True:
        raise RuntimeError("Round 2 budget is not a frozen passing budget")
    if result.get("parent_checkpoint_sha256") != common.sha256_file(_round1_parent(ctx)):
        raise RuntimeError("Round 2 budget parent differs from selected Round 1 parent")
    if result.get("calibration_sha256") != common.sha256_file(_calibration_path(ctx)):
        raise RuntimeError("Round 2 budget differs from the frozen calibration")
    return result


def _exposure_plan(ctx: Any, budget: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    base = int(budget["selected_round2_added_updates"])
    extension = int(budget["selected_round2_extension_added_updates"])
    prior: dict[str, Any] = {}
    apply_extension = False
    summary_path = ctx.output_dir / ROUND / "train_summary.json"
    decision_path = ctx.output_dir / ROUND / "decision.json"
    if resume and summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {}
        if decision.get("status") != "NEEDS_COMMON_EXTENSION":
            raise RuntimeError("completed Round 2 can resume only after its one allowed common extension")
        if prior.get("status") != "PASS" or prior.get("common_extension_applied") is True:
            raise RuntimeError("Round 2 extension provenance is invalid")
        if int(prior.get("added_successful_updates_per_arm", -1)) != base:
            raise RuntimeError("Round 2 base exposure differs from its frozen budget")
        apply_extension = True
    return {
        "base_delta": base,
        "extension_delta": extension,
        "target_delta": base + (extension if apply_extension else 0),
        "common_extension": apply_extension,
        "prior_summary": prior,
    }


def _resume_paths(ctx: Any) -> dict[str, Path]:
    root = ctx.output_dir / ROUND
    latest = {arm: root / arm / "latest.pt" for arm in ARMS}
    if not all(path.is_file() for path in latest.values()):
        raise FileNotFoundError("Round 2 resume requires both arm checkpoints")

    def exposure(path: Path, arm: str) -> tuple[Any, ...]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sequential = payload.get("sequential", {})
        if sequential.get("round") != ROUND or sequential.get("arm") != arm:
            raise RuntimeError(f"Round 2 resume contract mismatch for {arm}")
        return (
            int(payload.get("successful_optimizer_updates", -1)),
            int(sequential.get("added_successful_updates", -1)),
            int(sequential.get("added_future_atom_frame_tokens", -1)),
            dict(sequential.get("cursor", {})),
        )

    current = {arm: exposure(path, arm) for arm, path in latest.items()}
    if current["control"] == current["candidate"]:
        return latest
    common_names: Optional[set[str]] = None
    for arm in ARMS:
        names = {path.name for path in (root / arm).glob("checkpoint_step*.pt") if path.is_file()}
        common_names = names if common_names is None else common_names & names
    for name in sorted(common_names or (), reverse=True):
        paths = {arm: root / arm / name for arm in ARMS}
        values = {arm: exposure(path, arm) for arm, path in paths.items()}
        if values["control"] == values["candidate"]:
            return paths
    raise RuntimeError("Round 2 has no common atomic paired checkpoint")


def _checkpoint_payload(
    trainer: DiTTrainer,
    *,
    arm: str,
    parent_path: Path,
    parent_sha256: str,
    parent_model_hash: str,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["sequential"] = {
        "schema": f"{common.SCHEMA}.checkpoint.v1",
        "round": ROUND,
        "arm": arm,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_model_state_hash": parent_model_hash,
        "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        "training_generator_state": generator.get_state(),
        "added_future_atom_frame_tokens": int(added_tokens),
        "added_successful_updates": int(added_updates),
        "schedule_hash": schedule_hash,
        "unique_variable": dict(unique_variable),
        "code": _code_provenance(),
        "test_payload_opened": False,
    }
    return payload


def _save_checkpoint(
    directory: Path,
    trainer: DiTTrainer,
    *,
    arm: str,
    parent_path: Path,
    parent_sha256: str,
    parent_model_hash: str,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
    final: bool,
) -> Path:
    payload = _checkpoint_payload(
        trainer,
        arm=arm,
        parent_path=parent_path,
        parent_sha256=parent_sha256,
        parent_model_hash=parent_model_hash,
        cursor=cursor,
        generator=generator,
        added_tokens=added_tokens,
        added_updates=added_updates,
        schedule_hash=schedule_hash,
        unique_variable=unique_variable,
    )
    name = "checkpoint_final.pt" if final else f"checkpoint_step{trainer.step:06d}.pt"
    path = directory / name
    common._atomic_torch_save(path, payload)
    common._atomic_torch_save(directory / "latest.pt", payload)
    return path


def _unique_variable(arm: str, calibration: Mapping[str, Any], settings: Mapping[str, Any], extension: bool) -> dict[str, Any]:
    return {
        "ffn_norm_source": FFN_NORM_SOURCE,
        "geometry_supervision": _round2_metadata(
            arm=arm, lambda_bond=float(calibration["lambda_bond"]), settings=settings
        ),
        "calibration_sha256": calibration.get("calibration_sha256", ""),
        "common_extension": bool(extension),
    }


def _train(ctx: Any, *, resume: bool) -> dict[str, Any]:
    smoke_path = ctx.output_dir / ROUND / "smoke.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Round 2 requires a passing real-clip CUDA smoke")
    calibration = _load_calibration(ctx)
    calibration = {**calibration, "calibration_sha256": common.sha256_file(_calibration_path(ctx))}
    settings = _settings(ctx.cfg)
    budget = _load_budget(ctx)
    exposure = _exposure_plan(ctx, budget, resume=resume)
    parent_path, parent, state, parent_hash = _load_parent(ctx)
    parent_sha = common.sha256_file(parent_path)
    parent_step = int(parent["step"])
    target_step = parent_step + int(exposure["target_delta"])
    lambda_bond = float(calibration["lambda_bond"])
    trainers = {
        arm: _new_trainer(
            ctx,
            state=state,
            state_hash=parent_hash,
            max_steps=target_step,
            arm=arm,
            lambda_bond=lambda_bond,
        )
        for arm in ARMS
    }
    for trainer in trainers.values():
        common._restore_parent(
            trainer,
            parent,
            expected_codec_hash=ctx.codec.codec_state_hash,
            expected_statistics_hash=ctx.statistics.hash,
        )
    common.assert_independent_trainers(trainers["control"], trainers["candidate"])
    frozen_before = trainers["control"].frozen_state_hashes()
    seed = _round2_seed(ctx)
    generators = {arm: torch.Generator(device=ctx.device).manual_seed(seed) for arm in ARMS}
    sampler = _round2_sampler(ctx, seed=seed)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    tokens = {arm: 0 for arm in ARMS}
    start_added = 0
    if resume:
        restored = {}
        paths = _resume_paths(ctx)
        for arm in ARMS:
            payload = trainers[arm].load_checkpoint(paths[arm], map_location=ctx.device)
            sequential = payload.get("sequential", {})
            generators[arm].set_state(sequential["training_generator_state"].detach().cpu())
            restored[arm] = sequential
            tokens[arm] = int(sequential["added_future_atom_frame_tokens"])
        first, second = restored["control"], restored["candidate"]
        if (
            first["cursor"] != second["cursor"]
            or tokens["control"] != tokens["candidate"]
            or first["added_successful_updates"] != second["added_successful_updates"]
        ):
            raise RuntimeError("Round 2 arms do not resume at equal exposure")
        cursor = {"epoch": int(first["cursor"]["epoch"]), "batch_index": int(first["cursor"]["batch_index"])}
        start_added = int(first["added_successful_updates"])
        if any(trainer.successful_updates != parent_step + start_added for trainer in trainers.values()):
            raise RuntimeError("Round 2 resume checkpoint step differs from its added exposure")
        for arm in ARMS:
            common._truncate_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", step_key="added_successful_updates", maximum_step=start_added)
            common._truncate_jsonl(ctx.output_dir / ROUND / arm / "validation_history.jsonl", step_key="added_successful_updates", maximum_step=start_added)
    elif any((ctx.output_dir / ROUND / arm / "latest.pt").exists() for arm in ARMS):
        raise RuntimeError("fresh Round 2 run refuses existing arm checkpoints")

    compute_seconds = {arm: 0.0 for arm in ARMS}
    torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    schedule_hash = ""
    while trainers["control"].successful_updates < target_step:
        added = trainers["control"].successful_updates - parent_step
        if trainers["candidate"].successful_updates - parent_step != added:
            raise RuntimeError("Round 2 arms have unequal successful updates")
        indices, schedule_hash = common._next_batch(sampler, cursor, plan_cache)
        history = histories[added % len(histories)]
        _latent, batch, observed = common._prepare_batch(
            ctx, ctx.data.train, indices, trainers["control"].adapter, history
        )
        center = common._source_center(ctx, batch, observed, trainers["control"].adapter, history)
        batch_tokens = common._batch_tokens(specs, indices, history)
        for arm in ARMS:
            auxiliary = (
                _auxiliary(ctx, trainers[arm], batch, history, lambda_bond)
                if arm == "candidate"
                else None
            )
            common._sync(ctx.device)
            arm_started = time.perf_counter()
            row = trainers[arm].train_step(
                observed,
                generator=generators[arm],
                source_center=center,
                auxiliary_objective=auxiliary,
            )
            common._sync(ctx.device)
            compute_seconds[arm] += time.perf_counter() - arm_started
            tokens[arm] += batch_tokens
            row.update(
                {
                    "round": 2,
                    "arm": arm,
                    "history_frames": history,
                    "batch_tokens": batch_tokens,
                    "added_future_atom_frame_tokens": tokens[arm],
                    "added_successful_updates": trainers[arm].successful_updates - parent_step,
                    "parent_step": parent_step,
                    "lambda_bond": lambda_bond if arm == "candidate" else 0.0,
                }
            )
            common._append_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", row)
        if not torch.equal(generators["control"].get_state(), generators["candidate"].get_state()):
            raise RuntimeError("Round 2 tau/epsilon generators diverged")
        current = trainers["control"].successful_updates - parent_step
        if current % int(ctx.cfg["schedule"]["validation_interval"]) == 0:
            for arm in ARMS:
                validation = common._validation_rf(ctx, trainers[arm], step=trainers[arm].successful_updates)
                validation.update({"arm": arm, "added_successful_updates": current})
                common._append_jsonl(ctx.output_dir / ROUND / arm / "validation_history.jsonl", validation)
        if current % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0:
            for arm in ARMS:
                _save_checkpoint(
                    ctx.output_dir / ROUND / arm,
                    trainers[arm],
                    arm=arm,
                    parent_path=parent_path,
                    parent_sha256=parent_sha,
                    parent_model_hash=parent_hash,
                    cursor=cursor,
                    generator=generators[arm],
                    added_tokens=tokens[arm],
                    added_updates=current,
                    schedule_hash=schedule_hash,
                    unique_variable=_unique_variable(arm, calibration, settings, bool(exposure["common_extension"])),
                    final=False,
                )
    elapsed = time.perf_counter() - started
    expected_tokens = int(
        budget[
            "selected_round2_extended_total_future_atom_frame_tokens_per_arm"
            if exposure["common_extension"]
            else "selected_round2_future_atom_frame_tokens_per_arm"
        ]
    )
    if tokens["control"] != tokens["candidate"] or tokens["control"] != expected_tokens:
        raise RuntimeError("Round 2 token exposure differs from its frozen paired budget")
    checkpoints = {}
    for arm in ARMS:
        checkpoints[arm] = _save_checkpoint(
            ctx.output_dir / ROUND / arm,
            trainers[arm],
            arm=arm,
            parent_path=parent_path,
            parent_sha256=parent_sha,
            parent_model_hash=parent_hash,
            cursor=cursor,
            generator=generators[arm],
            added_tokens=tokens[arm],
            added_updates=int(exposure["target_delta"]),
            schedule_hash=schedule_hash,
            unique_variable=_unique_variable(arm, calibration, settings, bool(exposure["common_extension"])),
            final=True,
        )
    if any(trainer.frozen_state_hashes() != frozen_before for trainer in trainers.values()):
        raise RuntimeError("frozen codec/frame encoder changed during Round 2 training")
    prior = dict(exposure["prior_summary"])
    result = {
        "schema": f"{common.SCHEMA}.round2.training.v1",
        "status": "PASS",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "parent_model_state_hash": parent_hash,
        "initialization_hash": parent_hash,
        "unique_variable": "RF only versus RF plus calibrated differentiable future bond loss",
        "calibration": calibration,
        "config_hash": common._canonical_hash(ctx.cfg),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "code": _code_provenance(),
        "parent_successful_updates": parent_step,
        "base_budget_successful_updates_per_arm": int(exposure["base_delta"]),
        "common_extension_applied": bool(exposure["common_extension"]),
        "extension_successful_updates_per_arm": int(exposure["extension_delta"] if exposure["common_extension"] else 0),
        "added_successful_updates_per_arm": int(exposure["target_delta"]),
        "successful_optimizer_updates": {arm: trainers[arm].successful_updates for arm in ARMS},
        "added_future_atom_frame_tokens_per_arm": expected_tokens,
        "updates_this_invocation_per_arm": int(exposure["target_delta"]) - start_added,
        "wall_seconds_this_invocation": elapsed,
        "wall_seconds": float(prior.get("wall_seconds", 0.0)) + elapsed,
        "gpu_hours_this_invocation": elapsed / 3600.0,
        "gpu_hours": float(prior.get("gpu_hours", 0.0)) + elapsed / 3600.0,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
        "arm_compute_seconds_this_invocation": compute_seconds,
        "arm_compute_seconds": {arm: float(prior.get("arm_compute_seconds", {}).get(arm, 0.0)) + compute_seconds[arm] for arm in ARMS},
        "checkpoints": {
            arm: {
                "path": str(path),
                "sha256": common.sha256_file(path),
                "model_state_hash": module_state_hash(trainers[arm].model),
                "adapter_state_hash": module_state_hash(trainers[arm].adapter),
                "optimizer_state_owned": trainers[arm].optimizer is not trainers[ARMS[1 - ARMS.index(arm)]].optimizer,
                "scaler_owned": trainers[arm].scaler is not trainers[ARMS[1 - ARMS.index(arm)]].scaler,
            }
            for arm, path in checkpoints.items()
        },
        "frozen_hashes_unchanged": True,
        "test_payload_opened": False,
    }
    common._write_json(ctx.output_dir / ROUND / "train_summary.json", result)
    return result


def _smoke(ctx: Any) -> dict[str, Any]:
    calibration = _load_calibration(ctx)
    lambda_bond = float(calibration["lambda_bond"])
    parent_path, parent, state, parent_hash = _load_parent(ctx)
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    smoke_items = list(selection["smoke"])
    trainers = {
        arm: _new_trainer(
            ctx,
            state=state,
            state_hash=parent_hash,
            max_steps=int(parent["step"]) + len(smoke_items),
            arm=arm,
            lambda_bond=lambda_bond,
            round_name="round2_smoke",
        )
        for arm in ARMS
    }
    for trainer in trainers.values():
        common._restore_parent(
            trainer,
            parent,
            expected_codec_hash=ctx.codec.codec_state_hash,
            expected_statistics_hash=ctx.statistics.hash,
        )
    common.assert_independent_trainers(trainers["control"], trainers["candidate"])
    frozen_before = trainers["control"].frozen_state_hashes()
    dataset = ClipMMapDataset(Path(selection["smoke_root"]) / "train")
    rows = []
    candidate_applicable_updates = 0
    try:
        generators = {arm: torch.Generator(device=ctx.device).manual_seed(2026091402) for arm in ARMS}
        histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
        for offset, item in enumerate(smoke_items):
            indices = (int(item["dataset_index"]),)
            train_history = histories[offset % len(histories)]
            _latent, batch, observed = common._prepare_batch(
                ctx, dataset, indices, trainers["control"].adapter, train_history
            )
            center = common._source_center(ctx, batch, observed, trainers["control"].adapter, train_history)
            train_rows = {}
            for arm in ARMS:
                auxiliary = _auxiliary(ctx, trainers[arm], batch, train_history, lambda_bond) if arm == "candidate" else None
                train_rows[arm] = trainers[arm].train_step(
                    observed,
                    generator=generators[arm],
                    source_center=center,
                    auxiliary_objective=auxiliary,
                )
            if not torch.equal(generators["control"].get_state(), generators["candidate"].get_state()):
                raise RuntimeError("Round 2 smoke tau/epsilon generators diverged")
            candidate_applicable_updates += int(train_rows["candidate"].get("geometry_applicable_samples", 0))
            for history in (4, 8):
                _latent, generation_batch, generation_observed = common._prepare_batch(
                    ctx, dataset, indices, trainers["control"].adapter, history
                )
                generation_center = common._source_center(ctx, generation_batch, generation_observed, trainers["control"].adapter, history)
                normalized = ctx.statistics_device.normalize(generation_observed)
                noise, noise_meta = common.make_fixed_noise(
                    normalized,
                    seed=common.generation_seed(20260914, "|".join(generation_batch.sample_id), history, 0),
                )
                for arm in ARMS:
                    generated, generation = common.generate_fixed_noise_latent(
                        trainers[arm].model,
                        trainers[arm].adapter,
                        generation_observed,
                        ctx.statistics_device,
                        noise=noise,
                        steps=8,
                        source_center=generation_center,
                        source_mode="conditional",
                    )
                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                    rows.append(
                        {
                            "sample_id": str(item["sample_id"]),
                            "system": str(item["system"]),
                            "replica": str(item["replica"]),
                            "history_frames": history,
                            "training_history_frames": train_history,
                            "arm": arm,
                            "loss": train_rows[arm]["loss"],
                            "rf_loss": train_rows[arm]["rf_loss"],
                            "finite": bool(torch.isfinite(decoded).all()),
                            "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
                            "noise_sha256": noise_meta["noise_sha256"],
                            "geometry": {key: value for key, value in train_rows[arm].items() if str(key).startswith("geometry_")},
                        }
                    )
        applicable = candidate_applicable_updates
        passed = bool(rows) and applicable > 0 and all(row["finite"] and row["observed_clamp_exact"] for row in rows)
        passed = passed and all(trainer.successful_updates == int(parent["step"]) + len(smoke_items) for trainer in trainers.values())
        passed = passed and all(trainer.frozen_state_hashes() == frozen_before for trainer in trainers.values())
        result = {
            "schema": f"{common.SCHEMA}.round2.smoke.v1",
            "status": "PASS" if passed else "FAIL",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": common.sha256_file(parent_path),
            "lambda_bond": lambda_bond,
            "tau_threshold": float(_settings(ctx.cfg)["tau_threshold"]),
            "systems": list(ctx.cfg["evaluation"]["smoke"]["systems"]),
            "trajectory_count": len(smoke_items),
            "optimizer_updates_per_arm": len(smoke_items),
            "batching": "serial_single_clip",
            "candidate_applicable_samples": applicable,
            "rows": rows,
            "ranking_use": False,
            "device": common._cuda_info(ctx.device),
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "smoke.json", result)
        return result
    finally:
        dataset.close()


def _checkpoints(ctx: Any) -> dict[str, Path]:
    path = ctx.output_dir / ROUND / "train_summary.json"
    if not path.is_file():
        raise FileNotFoundError("Round 2 training summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Round 2 training did not pass")
    result = {}
    for arm in ARMS:
        checkpoint = Path(summary["checkpoints"][arm]["path"])
        if common.sha256_file(checkpoint) != summary["checkpoints"][arm]["sha256"]:
            raise RuntimeError(f"Round 2 {arm} checkpoint changed")
        result[arm] = checkpoint
    return result


def _evaluate(ctx: Any, scope: str) -> dict[str, Any]:
    results = {}
    for arm, checkpoint in _checkpoints(ctx).items():
        results[arm] = common._evaluate_checkpoint(
            ctx,
            checkpoint=checkpoint,
            label=arm,
            round_name=ROUND,
            scope=scope,
        )
    return {
        "schema": f"{common.SCHEMA}.round2.evaluation_pair.v1",
        "scope": scope,
        "arms": results,
        "sequential_loading": True,
        "test_payload_opened": False,
    }


def _motion_ratio(summary: Mapping[str, Any], history: int) -> dict[str, float]:
    """Return finite prediction RMSF values, including a genuine value of zero.

    Decision code uses :func:`motion_arm_report` directly so missing values
    retain their reasons; this wrapper remains a small compatibility helper for
    callers that only need the valid value map.
    """

    report = motion_arm_report(summary, history, label="round2")
    return {
        system: float(item["prediction"])
        for system, item in report["values"].items()
        if item.get("prediction") is not None
    }


def _report(ctx: Any, decision: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ctx.output_dir / ROUND / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, history in zip(axes, (4, 8)):
        control = {row["system"]: row["future"].get("bond_rmse") for row in evidence["control"]["aggregates"]["valid"][f"H{history}"]["system_rows"]}
        candidate = {row["system"]: row["future"].get("bond_rmse") for row in evidence["candidate"]["aggregates"]["valid"][f"H{history}"]["system_rows"]}
        pairs = [(float(control[key]), float(candidate[key])) for key in sorted(set(control) & set(candidate)) if isinstance(control[key], (int, float)) and isinstance(candidate[key], (int, float))]
        if pairs:
            first, second = zip(*pairs)
            bound = max((*first, *second, 1.0e-6))
            axis.scatter(first, second)
            axis.plot((0.0, bound), (0.0, bound), linestyle="--", color="black")
        axis.set(title=f"H{history}", xlabel="RF bond RMSE (A)", ylabel="RF + bond RMSE (A)")
    figure.suptitle("Round 2 per-system free-generation bond error")
    figure.tight_layout()
    figure.savefig(plot_dir / "bond_pair.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, history in zip(axes, (4, 8)):
        for arm in ARMS:
            rows = [json.loads(line) for line in Path(evidence[arm]["rows"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            by_frame: dict[int, list[float]] = {}
            for row in rows:
                if row["split"] != "valid" or int(row["history_frames"]) != history:
                    continue
                for frame in row["metrics"]["frame_curve"]:
                    value = frame.get("bond_rmse")
                    if frame.get("future") and isinstance(value, (int, float)):
                        by_frame.setdefault(int(frame["frame"]), []).append(float(value))
            frames = sorted(by_frame)
            axis.plot(frames, [sum(by_frame[value]) / len(by_frame[value]) for value in frames], marker="o", label=arm)
        axis.set(title=f"H{history}", xlabel="frame", ylabel="bond RMSE (A)")
        axis.legend()
    figure.suptitle("Round 2 future bond curve")
    figure.tight_layout()
    figure.savefig(plot_dir / "future_bond_curve.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for arm in ARMS:
        history = [json.loads(line) for line in (ctx.output_dir / ROUND / arm / "train_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        axis.plot([row["added_successful_updates"] for row in history], [row["loss"] for row in history], label=arm)
    axis.set(xlabel="added successful update", ylabel="total loss", title="Round 2 matched continuation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "training_loss.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Round 2: differentiable future bond supervision",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        f"- Auxiliary lambda: `{decision['calibration']['lambda_bond']}`",
        f"- Added updates per arm: {decision['successful_updates_per_arm']}",
        f"- Future atom-frame tokens per arm: {decision['future_atom_frame_tokens_per_arm']}",
        "",
        "| H | bond relative improvement | systems improved | contact change | amplitude-error change | median motion ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for history in (4, 8):
        key = f"H{history}"
        guard = decision["guardrails"][key]
        lines.append(f"| {key} | {decision['relative_improvement'][key]:.6f} | {decision['systems_same_direction'][key]} | {guard['contact_f1_change']:.6f} | {guard['amplitude_relative_change']:.6f} | {guard['median_motion_ratio'] if guard['median_motion_ratio'] is not None else 'n/a'} |")
    lines.extend(("", decision["reason"], "", f"Remaining risk: {decision['remaining_risk']}", ""))
    (ctx.output_dir / ROUND / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _decide(ctx: Any) -> dict[str, Any]:
    checkpoints = _checkpoints(ctx)

    def evidence(scope: str) -> dict[str, dict[str, Any]]:
        result = {}
        for arm in ARMS:
            path = ctx.output_dir / ROUND / f"evaluation_{scope}" / arm / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Round 2 {scope} evaluation is missing for {arm}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") != "PASS" or summary.get("checkpoint_sha256") != common.sha256_file(checkpoints[arm]):
                raise RuntimeError(f"Round 2 {scope} evaluation is stale for {arm}")
            result[arm] = summary
        return result

    quick = evidence("quick")
    threshold = float(ctx.cfg["decision"]["primary_relative_improvement"])
    tolerance = float(ctx.cfg["decision"]["opposite_trend_tolerance"])
    minimum = int(ctx.cfg["decision"]["minimum_systems_same_direction"])

    def primary_and_direction(value: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, int]]:
        improvement, direction = {}, {}
        for history in (4, 8):
            control = common._metric(value["control"], history, "bond_rmse")
            candidate = common._metric(value["candidate"], history, "bond_rmse")
            improvement[f"H{history}"] = (control - candidate) / max(abs(control), 1.0e-12)
            direction[f"H{history}"] = common._direction_count(value["control"], value["candidate"], history, "bond_rmse")
        return improvement, direction

    quick_improvement, quick_direction = primary_and_direction(quick)
    quick_promising = (
        sum(quick_improvement.values()) / 2.0 >= threshold
        and min(quick_improvement.values()) > -tolerance
        and min(quick_direction.values()) >= minimum
    )
    final_root = ctx.output_dir / ROUND / "evaluation_final"
    if quick_promising and not all((final_root / arm / "summary.json").is_file() for arm in ARMS):
        result = {
            "schema": f"{common.SCHEMA}.decision.v1",
            "round": 2,
            "status": "NEEDS_FINAL_EVALUATION",
            "parent_checkpoint": str(_round1_parent(ctx)),
            "unique_variable": "RF only versus RF plus future bond loss",
            "quick_primary_improvement": quick_improvement,
            "quick_systems_same_direction": quick_direction,
            "next_command": "evaluate-round2 --scope final, then decide-round2",
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "decision.json", result)
        return result
    current = evidence("final") if quick_promising else quick
    scope = "final" if quick_promising else "quick"
    improvement, direction = primary_and_direction(current)
    guards = {}
    for history in (4, 8):
        control_contact = common._metric(current["control"], history, "contact_f1")
        candidate_contact = common._metric(current["candidate"], history, "contact_f1")
        control_amp = common._metric(current["control"], history, "amplitude_error_angstrom")
        candidate_amp = common._metric(current["candidate"], history, "amplitude_error_angstrom")
        control_motion = motion_arm_report(current["control"], history, label="control")
        candidate_motion = motion_arm_report(current["candidate"], history, label="candidate")
        motion = paired_motion_report(
            candidate_motion,
            control_motion,
            candidate_label="candidate",
            reference_label="control",
            expected_systems=sorted(
                set(control_motion["actual_systems"]) | set(candidate_motion["actual_systems"])
            ),
            threshold=MOTION_RATIO_THRESHOLD,
        )
        guards[f"H{history}"] = {
            "contact_f1_change": candidate_contact - control_contact,
            "amplitude_relative_change": (candidate_amp - control_amp) / max(abs(control_amp), 1.0e-12),
            "median_motion_ratio": motion["median_motion_ratio"],
            "motion_ratio_system_count": motion["motion_ratio_system_count"],
            "motion_status": motion["status"],
            "motion_status_reason": motion["status_reason"],
            "motion_expected_systems": motion["expected_systems"],
            "motion_candidate_actual_systems": motion["candidate_actual_systems"],
            "motion_reference_actual_systems": motion["reference_actual_systems"],
            "motion_missing_candidate_systems": motion["missing_candidate_systems"],
            "motion_missing_reference_systems": motion["missing_reference_systems"],
            "motion_noncomputable_systems": motion["noncomputable_systems"],
        }
    primary_pass = sum(improvement.values()) / 2.0 >= threshold and min(improvement.values()) > -tolerance and min(direction.values()) >= minimum
    guard_pass = all(
        values["contact_f1_change"] >= -float(ctx.cfg["decision"]["contact_f1_absolute_drop"])
        and values["amplitude_relative_change"] <= float(ctx.cfg["decision"]["amplitude_relative_worsening"])
        and values.get("motion_status") == "PASS"
        and values["median_motion_ratio"] is not None
        and values["median_motion_ratio"] >= MOTION_RATIO_THRESHOLD
        for values in guards.values()
    )
    train = json.loads((ctx.output_dir / ROUND / "train_summary.json").read_text(encoding="utf-8"))
    if primary_pass and guard_pass:
        status, selected_arm, reason = "KEEP", "candidate", "free-generation bond error passed the paired threshold without contact, amplitude, or motion-collapse guardrail failure"
    elif primary_pass:
        status, selected_arm, reason = "TRADEOFF", "control", "bond improved but a contact/amplitude/motion guardrail failed"
    elif any(value > 0.0 for value in improvement.values()) and not bool(train.get("common_extension_applied")):
        status, selected_arm, reason = "NEEDS_COMMON_EXTENSION", None, "heterogeneous/sub-threshold evidence receives the one frozen common extension"
    elif any(value > 0.0 for value in improvement.values()):
        status, selected_arm, reason = "INCONCLUSIVE", "control", "the matched extension remained heterogeneous; retain the simpler equal-exposure control"
    else:
        status, selected_arm, reason = "REJECT", "control", "endpoint bond supervision did not improve free-generation bond error"
    calibration = _load_calibration(ctx)
    result = {
        "schema": f"{common.SCHEMA}.decision.v1",
        "round": 2,
        "status": status,
        "parent_checkpoint": train["parent_checkpoint"],
        "parent_checkpoint_sha256": train["parent_checkpoint_sha256"],
        "unique_variable": "RF only versus RF plus calibrated differentiable future bond loss",
        "evaluation_scope": scope,
        "primary_metric": "system-equal future free-generation bond RMSE",
        "relative_improvement": improvement,
        "systems_same_direction": direction,
        "guardrails": guards,
        "selected_arm": selected_arm,
        "selected_checkpoint": None if selected_arm is None else str(checkpoints[selected_arm]),
        "selected_checkpoint_sha256": None if selected_arm is None else common.sha256_file(checkpoints[selected_arm]),
        "calibration": calibration,
        "initialization_hash": train["initialization_hash"],
        "config_hash": train["config_hash"],
        "data_hash": train["data_hash"],
        "codec_state_hash": train["codec_state_hash"],
        "statistics_hash": train["statistics_hash"],
        "code": train["code"],
        "checkpoints": train["checkpoints"],
        "successful_optimizer_updates": train["successful_optimizer_updates"],
        "successful_updates_per_arm": train["added_successful_updates_per_arm"],
        "future_atom_frame_tokens_per_arm": train["added_future_atom_frame_tokens_per_arm"],
        "gpu_hours": train["gpu_hours"],
        "reason": reason,
        "remaining_risk": "one training seed; endpoint-coupled geometry supervision is not a proof of long rollout stability",
        "next_command": "train-round2 --resume; then evaluate-round2 --scope quick and decide-round2" if status == "NEEDS_COMMON_EXTENSION" else None,
        "test_payload_opened": False,
    }
    common._write_json(ctx.output_dir / ROUND / "decision.json", result)
    if status in ("KEEP", "REJECT", "TRADEOFF", "INCONCLUSIVE"):
        _report(ctx, result, current)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_architecture_sequential_v1.yaml")
    parser.add_argument("--stage", required=True, choices=("calibrate-round2", "profile-round2", "smoke-round2", "train-round2", "evaluate-round2", "decide-round2"))
    parser.add_argument("--scope", choices=("quick", "final"), default="quick")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = common._load_config(args.config.resolve())
    if args.run_id:
        cfg["run_id"] = str(args.run_id)
    output_dir = common._resolve(cfg["output_root"]) / str(cfg["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "decide-round2":
        result = _decide(type("MetadataContext", (), {"cfg": cfg, "output_dir": output_dir})())
        print(json.dumps(common._safe(result), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    ctx = common._load_context(cfg, output_dir, device)
    try:
        if args.stage == "calibrate-round2":
            result = _calibrate(ctx)
        elif args.stage == "profile-round2":
            result = _profile(ctx)
        elif args.stage == "smoke-round2":
            result = _smoke(ctx)
        elif args.stage == "train-round2":
            result = _train(ctx, resume=bool(args.resume))
        elif args.stage == "evaluate-round2":
            result = _evaluate(ctx, args.scope)
        else:
            raise AssertionError(args.stage)
        print(json.dumps(common._safe(result), indent=2, sort_keys=True))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
