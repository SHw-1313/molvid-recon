#!/usr/bin/env python
"""Run only Round 3 of the R4 DiT architecture-sequential experiment.

Round 3 freezes the selected Round 2 DiT, adapter, codec, and statistics.  It
trains equal small coordinate-domain temporal refiners from identical
target-coupled endpoint pairs: ``local`` may use only within-four-frame edges;
``cross_block`` additionally uses the three adjacent-frame block boundaries.
The unrefined selected Round 2 parent is retained as the P baseline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset
from evaluation.dit_architecture_sequential_v1 import load_sequential_checkpoint
from evaluation.motion_metrics import (
    MOTION_RATIO_THRESHOLD,
    motion_arm_report,
    paired_motion_report,
)
from module.dit_geometry_supervision import endpoint_from_velocity, observed_frame_mask
from module.latent_rectified_flow import RectifiedFlowObjective, apply_observation_clamp
from module.trajectory_temporal_refiner import (
    TemporalRefinerOutput,
    TrajectoryTemporalRefiner,
    temporal_refiner_loss,
)
from scripts.run_state_detail_codec_v2_t1 import TrajectoryCappedBatchSampler
from trainer.dit_trainer import module_state_hash

from scripts import run_dit_architecture_sequential_v1 as common


ROUND = "round3"
TRAIN_ARMS = ("local", "cross_block")
EVALUATION_ARMS = ("parent", *TRAIN_ARMS)
REFINER_CHECKPOINT_SCHEMA = "pvb.dit.architecture_sequential.v1.round3.refiner_checkpoint.v1"


@dataclass(frozen=True)
class FrozenRefinerPair:
    """One detached, target-coupled decoder endpoint used by both refiner arms."""

    x_input: Tensor
    h: Tensor
    v: Tensor
    target: Tensor
    frame_mask: Tensor
    time_ps: Tensor
    atom_mask: Tensor
    observed_frames: Tensor
    bond_index: Tensor
    abid: Tensor
    sample_ids: tuple[str, ...]
    tau: Tensor


def _settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = cfg.get(ROUND)
    if not isinstance(value, Mapping):
        raise ValueError("round3 configuration is missing")
    required = (
        "max_atom_frame_tokens",
        "hidden",
        "block_frames",
        "tau_min",
        "tau_max",
        "coordinate_weight",
        "bond_weight",
        "motion_weight",
        "future_rounds_training_reserve_gpu_hours",
        "evaluation_recovery_reserve_gpu_hours",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"round3 configuration lacks {missing}")
    result = dict(value)
    if int(result["max_atom_frame_tokens"]) < 56672:
        raise ValueError("round3 token cap cannot reject the largest frozen train clip")
    if int(result["hidden"]) != 64 or int(result["block_frames"]) != 4:
        raise ValueError("round3 fixes the minimal hidden=64 and four-frame block contract")
    if not 0.75 <= float(result["tau_min"]) < float(result["tau_max"]) <= 0.95:
        raise ValueError("round3 endpoint tau range must lie within [.75,.95]")
    if float(result["coordinate_weight"]) != 1.0:
        raise ValueError("round3 coordinate loss weight is fixed at one")
    if float(result["bond_weight"]) != 0.1 or float(result["motion_weight"]) != 0.1:
        raise ValueError("round3 bond and motion loss weights are fixed at 0.1")
    if min(float(result["future_rounds_training_reserve_gpu_hours"]), float(result["evaluation_recovery_reserve_gpu_hours"])) < 0.0:
        raise ValueError("round3 reserves must be non-negative")
    return result


def _code_provenance() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    files = (
        "scripts/run_dit_architecture_round3.py",
        "scripts/run_dit_architecture_sequential_v1.py",
        "module/trajectory_temporal_refiner.py",
        "module/dit_geometry_supervision.py",
        "evaluation/dit_architecture_sequential_v1.py",
        "config/dit_architecture_sequential_v1.yaml",
        "config/dit_architecture_sequential_v1.neibu.yaml",
        "tests/test_dit_architecture_round3_cuda.py",
    )
    hashes = {
        path: common.sha256_file(PROJECT_ROOT / path) if (PROJECT_ROOT / path).is_file() else None
        for path in files
    }
    return {"head": head, "files": hashes, "source_hash": common._canonical_hash(hashes)}


def _round3_seed(ctx: Any) -> int:
    return int(ctx.cfg["evaluation"]["training_seed_base"]) + 3


def _sampler(ctx: Any, *, seed: int) -> TrajectoryCappedBatchSampler:
    sampler = TrajectoryCappedBatchSampler(
        ctx.data.train,
        max_tokens=int(_settings(ctx.cfg)["max_atom_frame_tokens"]),
        clips_per_trajectory=int(ctx.cfg["schedule"]["clips_per_trajectory"]),
        seed=int(seed),
        shuffle=False,
    )
    sampler.set_epoch(0)
    return sampler


def _tokens_for_updates(ctx: Any, updates: int, seed: int) -> int:
    sampler = _sampler(ctx, seed=seed)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    total = 0
    for update in range(int(updates)):
        indices, _ = common._next_batch(sampler, cursor, plan_cache)
        total += common._batch_tokens(specs, indices, histories[update % len(histories)])
    return total


def _round2_parent_path(ctx: Any) -> tuple[Path, Mapping[str, Any]]:
    decision_path = ctx.output_dir / "round2" / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Round 3 requires the completed Round 2 decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") not in ("KEEP", "REJECT", "TRADEOFF", "INCONCLUSIVE"):
        raise RuntimeError("Round 2 must have a final decision before Round 3")
    path = Path(str(decision.get("selected_checkpoint", "")))
    if not path.is_file():
        raise RuntimeError("Round 2 selected checkpoint is unavailable")
    if common.sha256_file(path) != decision.get("selected_checkpoint_sha256"):
        raise RuntimeError("Round 2 selected checkpoint changed after its decision")
    if int(decision.get("round", -1)) != 2:
        raise RuntimeError("Round 3 parent decision does not identify Round 2")
    return path, decision


def _load_parent(ctx: Any) -> tuple[Path, Mapping[str, Any], Any]:
    path, decision = _round2_parent_path(ctx)
    loaded = load_sequential_checkpoint(
        path,
        device=ctx.device,
        expected_data_hash=ctx.data_hash,
        expected_codec_hash=ctx.codec.codec_state_hash,
        expected_statistics_hash=ctx.statistics.hash,
    )
    if loaded.checkpoint_sha256 != decision.get("selected_checkpoint_sha256"):
        raise RuntimeError("fresh Round 3 parent load does not match Round 2 decision")
    if any(parameter.requires_grad for parameter in loaded.model.parameters()):
        raise RuntimeError("Round 3 parent DiT was not frozen")
    return path, decision, loaded


def _assert_independent_refiners(first: nn.Module, second: nn.Module) -> None:
    if first is second:
        raise RuntimeError("Round 3 arms must own distinct refiner objects")
    first_storage = {parameter.data_ptr() for parameter in first.parameters() if parameter.requires_grad}
    second_storage = {parameter.data_ptr() for parameter in second.parameters() if parameter.requires_grad}
    if first_storage & second_storage:
        raise RuntimeError("Round 3 arms share trainable refiner storage")


def _new_refiners(ctx: Any) -> tuple[dict[str, TrajectoryTemporalRefiner], str]:
    settings = _settings(ctx.cfg)
    torch.manual_seed(_round3_seed(ctx))
    prototype = TrajectoryTemporalRefiner(
        int(ctx.cfg["model"]["codec_width"]),
        hidden=int(settings["hidden"]),
        block_frames=int(settings["block_frames"]),
        allow_cross_block=False,
    ).to(ctx.device)
    state = {name: value.detach().cpu().clone() for name, value in prototype.state_dict().items()}
    initial_hash = module_state_hash(prototype)
    refiners: dict[str, TrajectoryTemporalRefiner] = {}
    for arm, cross in (("local", False), ("cross_block", True)):
        refiner = TrajectoryTemporalRefiner(
            int(ctx.cfg["model"]["codec_width"]),
            hidden=int(settings["hidden"]),
            block_frames=int(settings["block_frames"]),
            allow_cross_block=cross,
        ).to(ctx.device)
        refiner.load_state_dict(state, strict=True)
        if module_state_hash(refiner) != initial_hash:
            raise RuntimeError("Round 3 refiner initialization did not round-trip exactly")
        refiners[arm] = refiner
    _assert_independent_refiners(refiners["local"], refiners["cross_block"])
    return refiners, initial_hash


def _new_optimizer(ctx: Any, refiner: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in refiner.parameters() if parameter.requires_grad],
        lr=float(ctx.cfg["protocol"]["learning_rate"]),
        weight_decay=float(ctx.cfg["protocol"]["weight_decay"]),
    )


def _new_scaler() -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


def _optimizer_to_device(optimizer: torch.optim.Optimizer) -> None:
    parameter = optimizer.param_groups[0]["params"][0]
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device=parameter.device)



def _capture_rng_state(device: torch.device) -> dict[str, Tensor]:
    return {
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device=device).detach().cpu().clone(),
    }


def _rng_states_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(
        isinstance(first.get(key), Tensor)
        and isinstance(second.get(key), Tensor)
        and torch.equal(first[key], second[key])
        for key in ("torch_cpu", "torch_cuda")
    )


def _restore_rng_state(ctx: Any, state: Mapping[str, Any]) -> None:
    if not _rng_states_equal(state, state):
        raise RuntimeError("Round 3 checkpoint has an invalid RNG state")
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu", dtype=torch.uint8))
    torch.cuda.set_rng_state(
        torch.as_tensor(state["torch_cuda"], device="cpu", dtype=torch.uint8), device=ctx.device
    )
def _atom_observed_mask(observed_frames: Tensor, abid: Tensor) -> Tensor:
    return observed_frames.index_select(0, abid).transpose(0, 1)


def _clamp_observed_coordinates(x_hat: Tensor, target: Tensor, observed_frames: Tensor, abid: Tensor) -> Tensor:
    observed_atom = _atom_observed_mask(observed_frames, abid)
    return torch.where(observed_atom.unsqueeze(-1), target.to(dtype=x_hat.dtype), x_hat)


@torch.no_grad()
def _endpoint_pair(
    ctx: Any,
    parent: Any,
    flow: RectifiedFlowObjective,
    dataset: Any,
    indices: Sequence[int],
    history: int,
    generator: torch.Generator,
) -> FrozenRefinerPair:
    _latent, coordinate, observed = common._prepare_batch(
        ctx, dataset, indices, parent.adapter, int(history)
    )
    center = common._source_center(ctx, coordinate, observed, parent.adapter, int(history))
    normalized = ctx.statistics_device.normalize(observed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sample = flow.sample(
            normalized,
            generator=generator,
            source_center=center,
            source_mode="conditional",
        )
        velocity = parent.model(normalized.with_fields(sample.interpolated), sample.tau)
    endpoint = endpoint_from_velocity(
        sample.interpolated, velocity, sample.tau, sample_ids=normalized.abid
    )
    endpoint = apply_observation_clamp(endpoint, normalized.fields, normalized)
    physical = ctx.statistics_device.inverse_fields(endpoint)
    decoded_batch = normalized.with_fields(physical).zero_invalid()
    decode_context = torch.autocast(device_type="cuda", enabled=False)
    with decode_context:
        decoded_batch = decoded_batch.with_fields(decoded_batch.fields.to(dtype=torch.float32))
        latent = parent.adapter.make_generated_latent(decoded_batch, decoded_batch.fields)
        decoded = ctx.codec.model.decode(latent)
    observed_frames = observed_frame_mask(normalized)
    target = torch.as_tensor(coordinate.x, device=ctx.device, dtype=torch.float32)
    x_input = _clamp_observed_coordinates(
        torch.as_tensor(decoded.x_hat, device=ctx.device, dtype=torch.float32),
        target,
        observed_frames,
        coordinate.abid,
    )
    observed_atom = _atom_observed_mask(observed_frames, coordinate.abid)
    if not torch.equal(x_input[observed_atom], target[observed_atom]):
        raise RuntimeError("Round 3 endpoint pair did not preserve supplied observed coordinates")
    settings = _settings(ctx.cfg)
    tau = sample.tau.detach().float()
    if bool(torch.any(tau < float(settings["tau_min"]))) or bool(torch.any(tau > float(settings["tau_max"]))):
        raise RuntimeError("Round 3 endpoint pair tau escaped its frozen range")
    return FrozenRefinerPair(
        x_input=x_input.detach(),
        h=torch.as_tensor(decoded.h, device=ctx.device, dtype=torch.float32).detach(),
        v=torch.as_tensor(decoded.v, device=ctx.device, dtype=torch.float32).detach(),
        target=target.detach(),
        frame_mask=torch.as_tensor(decoded.frame_mask, device=ctx.device, dtype=torch.bool).detach(),
        time_ps=torch.as_tensor(decoded.time_ps, device=ctx.device, dtype=torch.float32).detach(),
        atom_mask=torch.as_tensor(coordinate.loss_mask, device=ctx.device, dtype=torch.bool).detach(),
        observed_frames=observed_frames.detach(),
        bond_index=torch.as_tensor(coordinate.bond_index, device=ctx.device, dtype=torch.long).detach(),
        abid=torch.as_tensor(coordinate.abid, device=ctx.device, dtype=torch.long).detach(),
        sample_ids=tuple(str(value) for value in coordinate.sample_id),
        tau=tau.detach(),
    )


def _refine(refiner: TrajectoryTemporalRefiner, pair: FrozenRefinerPair) -> TemporalRefinerOutput:
    return refiner(
        pair.x_input,
        pair.h,
        pair.v,
        frame_mask=pair.frame_mask,
        time_ps=pair.time_ps,
        atom_mask=pair.atom_mask,
        observed_frames=pair.observed_frames,
        abid=pair.abid,
    )


def _refiner_step(
    ctx: Any,
    refiner: TrajectoryTemporalRefiner,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    pair: FrozenRefinerPair,
) -> dict[str, Any]:
    settings = _settings(ctx.cfg)
    refiner.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _refine(refiner, pair)
        loss = temporal_refiner_loss(
            output.x_refined,
            pair.target,
            frame_mask=pair.frame_mask,
            observed_frames=pair.observed_frames,
            time_ps=pair.time_ps,
            atom_mask=pair.atom_mask,
            bond_index=pair.bond_index,
            abid=pair.abid,
            bond_weight=float(settings["bond_weight"]),
            motion_weight=float(settings["motion_weight"]),
        )
    if not torch.isfinite(loss.total):
        raise FloatingPointError("Round 3 refiner loss is non-finite")
    loss.total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in refiner.parameters() if parameter.requires_grad],
        float(ctx.cfg["protocol"]["grad_clip"]),
    )
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("Round 3 refiner gradient is non-finite")
    optimizer.step()
    if any(
        not torch.isfinite(parameter.detach()).all()
        for parameter in refiner.parameters()
        if parameter.requires_grad
    ):
        raise FloatingPointError("Round 3 optimizer produced non-finite refiner parameters")
    observed_atom = _atom_observed_mask(pair.observed_frames, pair.abid)
    if not torch.equal(output.x_refined[observed_atom], pair.target[observed_atom]):
        raise RuntimeError("Round 3 refiner altered an observed coordinate")
    return {
        **loss.diagnostics(),
        "loss": loss.total.detach(),
        "grad_norm": torch.as_tensor(grad_norm).detach(),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "edge_count": int(output.edge_mask.sum().item()),
        "future_atom_count": int(output.future_atom_mask.sum().item()),
        "tau_mean": float(pair.tau.mean().item()),
        "tau_min": float(pair.tau.min().item()),
        "tau_max": float(pair.tau.max().item()),
        "observed_clamp_exact": True,
    }


def _prior_training_hours(ctx: Any) -> float:
    total = 0.0
    for prior_round in ("round1", "round2"):
        path = ctx.output_dir / prior_round / "train_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Round 3 requires {prior_round} training evidence")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"Round 3 refuses non-PASS {prior_round} training evidence")
        total += float(summary.get("gpu_hours", 0.0))
    return total


def _profile_generation(ctx: Any, parent: Any, refiners: Mapping[str, TrajectoryTemporalRefiner]) -> dict[str, Any]:
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    item = selection["quick_valid"][0]
    _latent, coordinate, observed = common._prepare_batch(
        ctx, ctx.data.valid, (int(item["dataset_index"]),), parent.adapter, 8
    )
    center = common._source_center(ctx, coordinate, observed, parent.adapter, 8)
    normalized = ctx.statistics_device.normalize(observed)
    noise, _meta = common.make_fixed_noise(
        normalized,
        seed=common.generation_seed(int(ctx.cfg["evaluation"]["generation_seed"]), str(item["sample_id"]), 8, 0),
    )
    common._sync(ctx.device)
    started = time.perf_counter()
    generated, generation = common.generate_fixed_noise_latent(
        parent.model,
        parent.adapter,
        observed,
        ctx.statistics_device,
        noise=noise,
        steps=int(ctx.cfg["evaluation"]["euler_steps"]),
        source_center=center,
        source_mode="conditional",
    )
    with torch.autocast(device_type="cuda", enabled=False):
        decoded = ctx.codec.model.decode(generated)
    observed_frames = observed_frame_mask(observed)
    raw = _clamp_observed_coordinates(
        decoded.x_hat.float(), coordinate.x.float(), observed_frames, coordinate.abid
    )
    for refiner in refiners.values():
        output = refiner(
            raw,
            decoded.h.float(),
            decoded.v.float(),
            frame_mask=decoded.frame_mask,
            time_ps=decoded.time_ps,
            atom_mask=coordinate.loss_mask,
            observed_frames=observed_frames,
            abid=coordinate.abid,
        )
        if not torch.isfinite(output.x_refined).all():
            raise RuntimeError("Round 3 profile generation produced non-finite coordinates")
    common._sync(ctx.device)
    return {
        "seconds": time.perf_counter() - started,
        "sample_id": str(item["sample_id"]),
        "history_frames": 8,
        "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
    }


def _profile(ctx: Any) -> dict[str, Any]:
    budget_path = ctx.output_dir / ROUND / "budget.json"
    if budget_path.is_file():
        existing = json.loads(budget_path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise RuntimeError("a non-passing Round 3 budget already exists; refusing to overwrite it")
        if existing.get("round3_settings_hash") != common._canonical_hash(_settings(ctx.cfg)):
            raise RuntimeError("existing Round 3 budget uses different settings")
        parent_path, _ = _round2_parent_path(ctx)
        if existing.get("parent_checkpoint_sha256") != common.sha256_file(parent_path):
            raise RuntimeError("existing Round 3 budget uses a different selected Round 2 parent")
        return existing
    settings = _settings(ctx.cfg)
    parent_path, decision, parent = _load_parent(ctx)
    refiners, initialization_hash = _new_refiners(ctx)
    optimizers = {arm: _new_optimizer(ctx, refiners[arm]) for arm in TRAIN_ARMS}
    scalers = {arm: _new_scaler() for arm in TRAIN_ARMS}
    flow = RectifiedFlowObjective(tau_min=float(settings["tau_min"]), tau_max=float(settings["tau_max"]))
    sampler = _sampler(ctx, seed=_round3_seed(ctx))
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    generator = torch.Generator(device=ctx.device).manual_seed(_round3_seed(ctx))
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    warmup = int(ctx.cfg["budget"]["profile_warmup_updates"])
    measured = int(ctx.cfg["budget"]["profile_measured_updates"])
    pair_seconds: list[float] = []
    peak_before = torch.cuda.max_memory_allocated(ctx.device)
    parent_hash_before = module_state_hash(parent.model)
    adapter_hash_before = module_state_hash(parent.adapter)
    codec_hash_before = module_state_hash(ctx.codec.model)
    try:
        for update in range(warmup + measured):
            indices, _schedule = common._next_batch(sampler, cursor, plan_cache)
            history = histories[update % len(histories)]
            common._sync(ctx.device)
            started = time.perf_counter()
            pair = _endpoint_pair(ctx, parent, flow, ctx.data.train, indices, history, generator)
            for arm in TRAIN_ARMS:
                _refiner_step(ctx, refiners[arm], optimizers[arm], scalers[arm], pair)
            common._sync(ctx.device)
            if update >= warmup:
                pair_seconds.append(time.perf_counter() - started)
        generation = _profile_generation(ctx, parent, refiners)
        if module_state_hash(parent.model) != parent_hash_before or module_state_hash(parent.adapter) != adapter_hash_before:
            raise RuntimeError("Round 3 profile altered its frozen parent")
        if module_state_hash(ctx.codec.model) != codec_hash_before:
            raise RuntimeError("Round 3 profile altered the frozen codec")
        p90 = float(np.quantile(pair_seconds, 0.9))
        prior_hours = _prior_training_hours(ctx)
        total_seconds = float(ctx.cfg["budget"]["gpu_hours_total"]) * 3600.0
        recovery = float(settings["evaluation_recovery_reserve_gpu_hours"]) * 3600.0
        future = float(settings["future_rounds_training_reserve_gpu_hours"]) * 3600.0
        available = total_seconds - prior_hours * 3600.0 - recovery - future
        fraction = float(ctx.cfg["budget"]["inconclusive_extension_fraction"])
        choices = []
        selected = None
        for updates in (int(value) for value in ctx.cfg["budget"]["round_candidate_updates"]):
            extension = int(math.ceil(updates * fraction))
            estimate = p90 * (updates + extension)
            feasible = estimate <= available
            choices.append({
                "added_updates_per_arm": updates,
                "extension_updates_per_arm": extension,
                "estimated_pair_gpu_seconds_including_extension": estimate,
                "feasible": feasible,
            })
            if selected is None and feasible:
                selected = updates
        if selected is None:
            raise RuntimeError("Round 3 cannot fit even the smallest paired refiner budget")
        extension = int(math.ceil(selected * fraction))
        seed = _round3_seed(ctx)
        round_tokens = _tokens_for_updates(ctx, selected, seed)
        extended_tokens = _tokens_for_updates(ctx, selected + extension, seed)
        result = {
            "schema": f"{common.SCHEMA}.round3.budget.v1",
            "status": "PASS",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": common.sha256_file(parent_path),
            "parent_model_state_hash": parent.model_state_hash,
            "parent_adapter_state_hash": parent.adapter_state_hash,
            "round2_decision_status": decision["status"],
            "round3_settings": settings,
            "round3_settings_hash": common._canonical_hash(settings),
            "refiner_initialization_hash": initialization_hash,
            "pair_cache": "streaming_no_cache",
            "profile": {
                "warmup_updates": warmup,
                "measured_updates": measured,
                "pair_seconds": pair_seconds,
                "pair_p90_seconds_per_update": p90,
                "generation": generation,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
                "peak_memory_delta_bytes": int(torch.cuda.max_memory_allocated(ctx.device) - peak_before),
            },
            "prior_round_training_seconds": prior_hours * 3600.0,
            "gpu_seconds_total": total_seconds,
            "evaluation_recovery_reserve_seconds": recovery,
            "future_rounds_training_reserve_seconds": future,
            "available_round3_training_seconds": available,
            "choices": choices,
            "selected_round3_added_updates": selected,
            "selected_round3_extension_added_updates": extension,
            "selected_round3_future_atom_frame_tokens_per_arm": round_tokens,
            "selected_round3_extension_future_atom_frame_tokens_per_arm": extended_tokens - round_tokens,
            "selected_round3_extended_total_updates": selected + extension,
            "code": _code_provenance(),
            "frozen_before_training": True,
            "device": common._cuda_info(ctx.device),
            "test_payload_opened": False,
        }
        result["selected_round3_extended_total_future_atom_frame_tokens_per_arm"] = extended_tokens
        common._write_json(budget_path, result)
        common._write_json(ctx.output_dir / ROUND / "profile.json", result)
        return result
    finally:
        del parent, refiners, optimizers, scalers
        torch.cuda.empty_cache()


def _load_budget(ctx: Any) -> dict[str, Any]:
    path = ctx.output_dir / ROUND / "budget.json"
    if not path.is_file():
        raise FileNotFoundError("Round 3 training requires a frozen profile budget")
    budget = json.loads(path.read_text(encoding="utf-8"))
    if budget.get("status") != "PASS" or budget.get("frozen_before_training") is not True:
        raise RuntimeError("Round 3 budget did not pass")
    if budget.get("round3_settings_hash") != common._canonical_hash(_settings(ctx.cfg)):
        raise RuntimeError("Round 3 settings differ from the frozen budget")
    parent_path, _ = _round2_parent_path(ctx)
    if budget.get("parent_checkpoint_sha256") != common.sha256_file(parent_path):
        raise RuntimeError("Round 3 budget parent differs from the selected Round 2 parent")
    return budget


def _smoke(ctx: Any) -> dict[str, Any]:
    parent_path, _decision, parent = _load_parent(ctx)
    refiners, initial_hash = _new_refiners(ctx)
    optimizers = {arm: _new_optimizer(ctx, refiners[arm]) for arm in TRAIN_ARMS}
    scalers = {arm: _new_scaler() for arm in TRAIN_ARMS}
    settings = _settings(ctx.cfg)
    flow = RectifiedFlowObjective(tau_min=float(settings["tau_min"]), tau_max=float(settings["tau_max"]))
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    dataset = ClipMMapDataset(Path(selection["smoke_root"]) / "train")
    generator = torch.Generator(device=ctx.device).manual_seed(_round3_seed(ctx))
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    parent_hash = module_state_hash(parent.model)
    adapter_hash = module_state_hash(parent.adapter)
    codec_hash = module_state_hash(ctx.codec.model)
    rows = []
    try:
        for update, item in enumerate(selection["smoke"]):
            indices = (int(item["dataset_index"]),)
            train_history = histories[update % len(histories)]
            pair = _endpoint_pair(ctx, parent, flow, dataset, indices, train_history, generator)
            step_rows = {
                arm: _refiner_step(ctx, refiners[arm], optimizers[arm], scalers[arm], pair)
                for arm in TRAIN_ARMS
            }
            for history in (4, 8):
                raw, decoded, coordinate, observed, generation, noise_sha = _free_decode(
                    ctx, parent, dataset, indices, history, draw=0
                )
                observed_frames = observed_frame_mask(observed)
                variants: dict[str, Tensor] = {"parent": raw}
                for arm in TRAIN_ARMS:
                    output = refiners[arm](
                        raw,
                        decoded.h.float(),
                        decoded.v.float(),
                        frame_mask=decoded.frame_mask,
                        time_ps=decoded.time_ps,
                        atom_mask=coordinate.loss_mask,
                        observed_frames=observed_frames,
                        abid=coordinate.abid,
                    )
                    variants[arm] = output.x_refined
                observed_atom = _atom_observed_mask(observed_frames, coordinate.abid)
                for arm, coordinates in variants.items():
                    rows.append({
                        "sample_id": str(item["sample_id"]),
                        "system": str(item["system"]),
                        "replica": str(item["replica"]),
                        "history_frames": history,
                        "training_history_frames": train_history,
                        "arm": arm,
                        "finite": bool(torch.isfinite(coordinates).all()),
                        "observed_clamp_exact": bool(torch.equal(coordinates[observed_atom], coordinate.x.float()[observed_atom])),
                        "latent_observed_clamp_exact": bool(generation["observed_clamp_exact"]),
                        "noise_sha256": noise_sha,
                        "local_loss": step_rows["local"]["loss"],
                        "cross_block_loss": step_rows["cross_block"]["loss"],
                    })
        passed = bool(rows) and all(
            row["finite"] and row["observed_clamp_exact"] and row["latent_observed_clamp_exact"]
            for row in rows
        )
        passed = passed and module_state_hash(parent.model) == parent_hash
        passed = passed and module_state_hash(parent.adapter) == adapter_hash
        passed = passed and module_state_hash(ctx.codec.model) == codec_hash
        result = {
            "schema": f"{common.SCHEMA}.round3.smoke.v1",
            "status": "PASS" if passed else "FAIL",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": common.sha256_file(parent_path),
            "refiner_initialization_hash": initial_hash,
            "systems": list(ctx.cfg["evaluation"]["smoke"]["systems"]),
            "trajectory_count": len(selection["smoke"]),
            "optimizer_updates_per_refiner": len(selection["smoke"]),
            "batching": "serial_single_clip",
            "rows": rows,
            "ranking_use": False,
            "device": common._cuda_info(ctx.device),
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "smoke.json", result)
        return result
    finally:
        dataset.close()
        del parent, refiners, optimizers, scalers
        torch.cuda.empty_cache()


@torch.no_grad()
def _free_decode(
    ctx: Any,
    parent: Any,
    dataset: Any,
    indices: Sequence[int],
    history: int,
    *,
    draw: int,
) -> tuple[Tensor, Any, Any, Any, Mapping[str, Any], str]:
    _latent, coordinate, observed = common._prepare_batch(ctx, dataset, indices, parent.adapter, history)
    center = common._source_center(ctx, coordinate, observed, parent.adapter, history)
    normalized = ctx.statistics_device.normalize(observed)
    seed = common.generation_seed(
        int(ctx.cfg["evaluation"]["generation_seed"]),
        "|".join(str(value) for value in coordinate.sample_id),
        history,
        draw,
    )
    noise, noise_meta = common.make_fixed_noise(normalized, seed=seed)
    generated, generation = common.generate_fixed_noise_latent(
        parent.model,
        parent.adapter,
        observed,
        ctx.statistics_device,
        noise=noise,
        steps=int(ctx.cfg["evaluation"]["euler_steps"]),
        source_center=center,
        source_mode="conditional",
    )
    with torch.autocast(device_type="cuda", enabled=False):
        decoded = ctx.codec.model.decode(generated)
    observed_frames = observed_frame_mask(observed)
    raw = _clamp_observed_coordinates(decoded.x_hat.float(), coordinate.x.float(), observed_frames, coordinate.abid)
    return raw, decoded, coordinate, observed, generation, str(noise_meta["noise_sha256"])


def _checkpoint_payload(
    ctx: Any,
    refiner: TrajectoryTemporalRefiner,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    arm: str,
    parent_path: Path,
    parent: Any,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_updates: int,
    added_tokens: int,
    schedule_hash: str,
    initialization_hash: str,
    common_extension: bool,
    rng_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    return {
        "config": copy.deepcopy(dict(ctx.cfg)),
        "config_hash": common._canonical_hash(ctx.cfg),
        "schema": REFINER_CHECKPOINT_SCHEMA,
        "round": ROUND,
        "arm": arm,
        "refiner_contract": refiner.contract(),
        "refiner_state": refiner.state_dict(),
        "refiner_state_hash": module_state_hash(refiner),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_model_state_hash": parent.model_state_hash,
        "parent_adapter_state_hash": parent.adapter_state_hash,
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "initialization_hash": initialization_hash,
        "round3_settings_hash": common._canonical_hash(_settings(ctx.cfg)),
        "frozen_assets": {
            "parent_model": module_state_hash(parent.model),
            "parent_adapter": module_state_hash(parent.adapter),
            "codec": module_state_hash(ctx.codec.model),
            "statistics": ctx.statistics.hash,
        },
        "sequential": {
            "schema": f"{common.SCHEMA}.round3.checkpoint.v1",
            "rng_state": {
                key: torch.as_tensor(value, device="cpu", dtype=torch.uint8).detach().clone()
                for key, value in rng_state.items()
            },
            "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
            "pair_generator_state": generator.get_state(),
            "added_successful_updates": int(added_updates),
            "added_future_atom_frame_tokens": int(added_tokens),
            "schedule_hash": str(schedule_hash),
            "common_extension": bool(common_extension),
            "pair_cache": "streaming_no_cache",
            "unique_variable": {
                "refiner_supervision": "coordinate_plus_0.1_bond_plus_0.1_motion",
                "cross_block_edges": arm == "cross_block",
            },
            "code": _code_provenance(),
            "test_payload_opened": False,
        },
    }


def _save_checkpoint(
    ctx: Any,
    refiner: TrajectoryTemporalRefiner,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    arm: str,
    parent_path: Path,
    parent: Any,
    cursor: Mapping[str, int],
    generator: torch.Generator,
    added_updates: int,
    added_tokens: int,
    rng_state: Mapping[str, Tensor],
    schedule_hash: str,
    initialization_hash: str,
    common_extension: bool,
    final: bool,
) -> Path:
    payload = _checkpoint_payload(
        ctx,
        refiner,
        optimizer,
        scaler,
        arm=arm,
        parent_path=parent_path,
        parent=parent,
        cursor=cursor,
        generator=generator,
        added_updates=added_updates,
        added_tokens=added_tokens,
        schedule_hash=schedule_hash,
        rng_state=rng_state,
        initialization_hash=initialization_hash,
        common_extension=common_extension,
    )
    directory = ctx.output_dir / ROUND / arm
    name = "checkpoint_final.pt" if final else f"checkpoint_step{added_updates:06d}.pt"
    path = directory / name
    common._atomic_torch_save(path, payload)
    common._atomic_torch_save(directory / "latest.pt", payload)
    return path


def _checkpoint_exposure(path: Path, arm: str) -> tuple[int, int, dict[str, int], Tensor, bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != REFINER_CHECKPOINT_SCHEMA or payload.get("arm") != arm:
        raise RuntimeError(f"Round 3 checkpoint contract mismatch: {path}")
    sequential = payload.get("sequential")
    if not isinstance(sequential, Mapping) or sequential.get("schema") != f"{common.SCHEMA}.round3.checkpoint.v1":
        raise RuntimeError(f"Round 3 checkpoint lacks sequential state: {path}")
    generator = sequential.get("pair_generator_state")
    if not isinstance(generator, Tensor):
        raise RuntimeError(f"Round 3 checkpoint lacks pair generator state: {path}")
    return (
        int(sequential.get("added_successful_updates", -1)),
        int(sequential.get("added_future_atom_frame_tokens", -1)),
        {"epoch": int(sequential["cursor"]["epoch"]), "batch_index": int(sequential["cursor"]["batch_index"])},
        generator,
        bool(sequential.get("common_extension", False)),
    )


def _resume_paths(ctx: Any) -> dict[str, Path]:
    root = ctx.output_dir / ROUND
    latest = {arm: root / arm / "latest.pt" for arm in TRAIN_ARMS}
    if not all(path.is_file() for path in latest.values()):
        raise FileNotFoundError("Round 3 resume requires both arm checkpoints")
    exposures = {arm: _checkpoint_exposure(path, arm) for arm, path in latest.items()}
    first, second = exposures["local"], exposures["cross_block"]
    if first[:3] == second[:3] and torch.equal(first[3], second[3]) and first[4] == second[4]:
        return latest
    names: set[str] | None = None
    for arm in TRAIN_ARMS:
        current = {path.name for path in (root / arm).glob("checkpoint_step*.pt") if path.is_file()}
        names = current if names is None else names & current
    for name in sorted(names or (), reverse=True):
        candidate = {arm: root / arm / name for arm in TRAIN_ARMS}
        values = {arm: _checkpoint_exposure(path, arm) for arm, path in candidate.items()}
        first, second = values["local"], values["cross_block"]
        if first[:3] == second[:3] and torch.equal(first[3], second[3]) and first[4] == second[4]:
            return candidate
    raise RuntimeError("Round 3 has no common equal-exposure checkpoint for resume")


def _restore_checkpoint(
    ctx: Any,
    path: Path,
    *,
    arm: str,
    parent_path: Path,
    parent: Any,
    refiner: TrajectoryTemporalRefiner,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    initialization_hash: str,
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != REFINER_CHECKPOINT_SCHEMA or payload.get("arm") != arm:
        raise RuntimeError("Round 3 resume checkpoint arm/schema mismatch")
    expected = {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_model_state_hash": parent.model_state_hash,
        "parent_adapter_state_hash": parent.adapter_state_hash,
        "data_hash": ctx.data_hash,
        "config_hash": common._canonical_hash(ctx.cfg),
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "initialization_hash": initialization_hash,
        "round3_settings_hash": common._canonical_hash(_settings(ctx.cfg)),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Round 3 resume checkpoint {key} mismatch")
    if dict(payload.get("refiner_contract", {})) != refiner.contract():
        raise RuntimeError("Round 3 resume refiner contract mismatch")
    refiner.load_state_dict(payload["refiner_state"], strict=True)
    if module_state_hash(refiner) != payload.get("refiner_state_hash"):
        raise RuntimeError("Round 3 resume refiner hash mismatch")
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _optimizer_to_device(optimizer)
    scaler.load_state_dict(copy.deepcopy(payload.get("scaler_state", {})))
    sequential = payload.get("sequential")
    if not isinstance(sequential, Mapping) or not isinstance(sequential.get("rng_state"), Mapping):
        raise RuntimeError("Round 3 resume checkpoint lacks sequential payload")
    if not _rng_states_equal(sequential["rng_state"], sequential["rng_state"]):
        raise RuntimeError("Round 3 resume checkpoint has an invalid RNG state")
    if not isinstance(sequential.get("pair_generator_state"), Tensor):
        raise RuntimeError("Round 3 resume checkpoint lacks its endpoint-pair generator state")
    return sequential


def _train(ctx: Any, *, resume: bool) -> dict[str, Any]:
    smoke_path = ctx.output_dir / ROUND / "smoke.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Round 3 training requires a passing real-clip smoke")
    budget = _load_budget(ctx)
    previous_decision_path = ctx.output_dir / ROUND / "decision.json"
    previous = json.loads(previous_decision_path.read_text(encoding="utf-8")) if previous_decision_path.is_file() else {}
    common_extension = previous.get("status") == "NEEDS_COMMON_EXTENSION"
    if common_extension and not resume:
        raise RuntimeError("Round 3 common extension must resume the matched base exposure")
    summary_path = ctx.output_dir / ROUND / "train_summary.json"
    prior_summary: Mapping[str, Any] = {}
    if summary_path.is_file():
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if common_extension:
        if prior_summary.get("status") != "PASS":
            raise RuntimeError("Round 3 common extension requires the completed base training summary")
        if int(prior_summary.get("successful_updates_per_arm", -1)) != int(budget["selected_round3_added_updates"]):
            raise RuntimeError("Round 3 common extension must start from exactly the matched base exposure")
        if int(prior_summary.get("added_future_atom_frame_tokens_per_arm", -1)) != int(budget["selected_round3_future_atom_frame_tokens_per_arm"]):
            raise RuntimeError("Round 3 common extension base token exposure is inconsistent")
        if bool(prior_summary.get("common_extension_applied")):
            raise RuntimeError("Round 3 only permits one common extension")

    base_updates = int(budget["selected_round3_added_updates"])
    extension_updates = int(budget["selected_round3_extension_added_updates"])
    target_updates = base_updates + extension_updates if common_extension else base_updates
    expected_tokens = int(
        budget[
            "selected_round3_extended_total_future_atom_frame_tokens_per_arm"
            if common_extension
            else "selected_round3_future_atom_frame_tokens_per_arm"
        ]
    )
    parent_path, _decision, parent = _load_parent(ctx)
    refiners, initialization_hash = _new_refiners(ctx)
    optimizers = {arm: _new_optimizer(ctx, refiners[arm]) for arm in TRAIN_ARMS}
    scalers = {arm: _new_scaler() for arm in TRAIN_ARMS}
    flow = RectifiedFlowObjective(
        tau_min=float(_settings(ctx.cfg)["tau_min"]), tau_max=float(_settings(ctx.cfg)["tau_max"])
    )
    sampler = _sampler(ctx, seed=_round3_seed(ctx))
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    generator = torch.Generator(device=ctx.device).manual_seed(_round3_seed(ctx))
    start_updates = 0
    tokens = 0
    if resume:
        paths = _resume_paths(ctx)
        restored = {
            arm: _restore_checkpoint(
                ctx,
                paths[arm],
                arm=arm,
                parent_path=parent_path,
                parent=parent,
                refiner=refiners[arm],
                optimizer=optimizers[arm],
                scaler=scalers[arm],
                initialization_hash=initialization_hash,
            )
            for arm in TRAIN_ARMS
        }
        first, second = restored["local"], restored["cross_block"]
        if (dict(first["cursor"]) != dict(second["cursor"]) or int(first["added_successful_updates"]) != int(second["added_successful_updates"]) or int(first["added_future_atom_frame_tokens"]) != int(second["added_future_atom_frame_tokens"])):
            raise RuntimeError("Round 3 resume arms do not have equal exposure")
        if not torch.equal(first["pair_generator_state"], second["pair_generator_state"]):
            raise RuntimeError("Round 3 resume arms do not share the same pair generator state")
        cursor = {"epoch": int(first["cursor"]["epoch"]), "batch_index": int(first["cursor"]["batch_index"])}
        if not _rng_states_equal(first["rng_state"], second["rng_state"]):
            raise RuntimeError("Round 3 resume arms do not share the same RNG state")
        generator.set_state(first["pair_generator_state"].detach().cpu())
        _restore_rng_state(ctx, first["rng_state"])
        start_updates = int(first["added_successful_updates"])
        tokens = int(first["added_future_atom_frame_tokens"])
        for arm in TRAIN_ARMS:
            common._truncate_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", step_key="added_successful_updates", maximum_step=start_updates)
    elif any((ctx.output_dir / ROUND / arm / "latest.pt").is_file() for arm in TRAIN_ARMS):
        raise RuntimeError("fresh Round 3 run refuses existing checkpoints")
    invocation_start_updates = start_updates
    if start_updates >= target_updates:
        raise RuntimeError("Round 3 resume already meets or exceeds its target")
    parent_hash_before = module_state_hash(parent.model)
    adapter_hash_before = module_state_hash(parent.adapter)
    codec_hash_before = module_state_hash(ctx.codec.model)
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    specs = ctx.data.train.clip_spec_table()
    arm_seconds = {arm: 0.0 for arm in TRAIN_ARMS}
    started = time.perf_counter()
    schedule_hash = ""
    try:
        while start_updates < target_updates:
            indices, schedule_hash = common._next_batch(sampler, cursor, plan_cache)
            history = histories[start_updates % len(histories)]
            pair = _endpoint_pair(ctx, parent, flow, ctx.data.train, indices, history, generator)
            batch_tokens = common._batch_tokens(specs, indices, history)
            rows: dict[str, dict[str, Any]] = {}
            for arm in TRAIN_ARMS:
                common._sync(ctx.device)
                arm_started = time.perf_counter()
                rows[arm] = _refiner_step(ctx, refiners[arm], optimizers[arm], scalers[arm], pair)
                common._sync(ctx.device)
                arm_seconds[arm] += time.perf_counter() - arm_started
            start_updates += 1
            tokens += batch_tokens
            for arm in TRAIN_ARMS:
                row = dict(rows[arm])
                row.update({
                    "round": 3,
                    "arm": arm,
                    "history_frames": history,
                    "batch_tokens": batch_tokens,
                    "added_future_atom_frame_tokens": tokens,
                    "added_successful_updates": start_updates,
                    "parent_checkpoint_sha256": parent.checkpoint_sha256,
                })
                common._append_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", row)
            if start_updates % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0:
                checkpoint_rng_state = _capture_rng_state(ctx.device)
                for arm in TRAIN_ARMS:
                    _save_checkpoint(
                        ctx, refiners[arm], optimizers[arm], scalers[arm], arm=arm,
                        parent_path=parent_path, parent=parent, cursor=cursor, generator=generator,
                        added_updates=start_updates, added_tokens=tokens, rng_state=checkpoint_rng_state, schedule_hash=schedule_hash,
                        initialization_hash=initialization_hash, common_extension=common_extension, final=False,
                    )
        if tokens != expected_tokens:
            raise RuntimeError("Round 3 paired token exposure differs from its frozen budget")
        checkpoints = {}
        checkpoint_rng_state = _capture_rng_state(ctx.device)
        for arm in TRAIN_ARMS:
            path = _save_checkpoint(
                ctx, refiners[arm], optimizers[arm], scalers[arm], arm=arm,
                parent_path=parent_path, parent=parent, cursor=cursor, generator=generator,
                added_updates=start_updates, added_tokens=tokens, rng_state=checkpoint_rng_state, schedule_hash=schedule_hash,
                initialization_hash=initialization_hash, common_extension=common_extension, final=True,
            )
            checkpoints[arm] = {
                "path": str(path),
                "sha256": common.sha256_file(path),
                "refiner_state_hash": module_state_hash(refiners[arm]),
                "optimizer_state_owned": optimizers[arm] is not optimizers["cross_block" if arm == "local" else "local"],
                "scaler_owned": scalers[arm] is not scalers["cross_block" if arm == "local" else "local"],
            }
        if module_state_hash(parent.model) != parent_hash_before or module_state_hash(parent.adapter) != adapter_hash_before:
            raise RuntimeError("Round 3 training altered the frozen parent")
        if module_state_hash(ctx.codec.model) != codec_hash_before:
            raise RuntimeError("Round 3 training altered the frozen codec")
        elapsed = time.perf_counter() - started
        prior_wall_seconds = float(prior_summary.get("wall_seconds", 0.0)) if resume else 0.0
        prior_gpu_hours = float(prior_summary.get("gpu_hours", 0.0)) if resume else 0.0
        prior_arm_seconds = {
            arm: float(prior_summary.get("arm_compute_seconds", {}).get(arm, 0.0)) if resume else 0.0
            for arm in TRAIN_ARMS
        }
        result = {
            "schema": f"{common.SCHEMA}.round3.training.v1",
            "status": "PASS",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": parent.checkpoint_sha256,
            "parent_model_state_hash": parent.model_state_hash,
            "parent_adapter_state_hash": parent.adapter_state_hash,
            "initialization_hash": initialization_hash,
            "refiner_contracts": {arm: refiners[arm].contract() for arm in TRAIN_ARMS},
            "checkpoints": checkpoints,
            "successful_updates_per_arm": start_updates,
            "base_budget_successful_updates_per_arm": base_updates,
            "extension_successful_updates_per_arm": max(0, start_updates - base_updates),
            "common_extension_applied": common_extension,
            "added_future_atom_frame_tokens_per_arm": tokens,
            "updates_this_invocation_per_arm": start_updates - invocation_start_updates,
            "arm_compute_seconds_this_invocation": arm_seconds,
            "arm_compute_seconds": {arm: prior_arm_seconds[arm] + arm_seconds[arm] for arm in TRAIN_ARMS},
            "wall_seconds_this_invocation": elapsed,
            "wall_seconds": prior_wall_seconds + elapsed,
            "gpu_hours_this_invocation": elapsed / 3600.0,
            "gpu_hours": prior_gpu_hours + elapsed / 3600.0,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
            "frozen_hashes_unchanged": True,
            "config_hash": common._canonical_hash(ctx.cfg),
            "data_hash": ctx.data_hash,
            "codec_state_hash": ctx.codec.codec_state_hash,
            "statistics_hash": ctx.statistics.hash,
            "code": _code_provenance(),
            "pair_cache": "streaming_no_cache",
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "train_summary.json", result)
        return result
    finally:
        del parent, refiners, optimizers, scalers
        torch.cuda.empty_cache()


def _checkpoints(ctx: Any) -> dict[str, Path]:
    summary_path = ctx.output_dir / ROUND / "train_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Round 3 training summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Round 3 training did not pass")
    result = {}
    for arm in TRAIN_ARMS:
        path = Path(str(summary["checkpoints"][arm]["path"]))
        if not path.is_file() or common.sha256_file(path) != summary["checkpoints"][arm]["sha256"]:
            raise RuntimeError(f"Round 3 {arm} checkpoint changed")
        result[arm] = path
    return result


def _load_refiner_for_evaluation(ctx: Any, path: Path, *, arm: str, parent_path: Path, parent: Any) -> TrajectoryTemporalRefiner:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != REFINER_CHECKPOINT_SCHEMA or payload.get("arm") != arm:
        raise RuntimeError("Round 3 evaluation checkpoint arm/schema mismatch")
    settings = _settings(ctx.cfg)
    refiner = TrajectoryTemporalRefiner(
        int(ctx.cfg["model"]["codec_width"]),
        hidden=int(settings["hidden"]),
        block_frames=int(settings["block_frames"]),
        allow_cross_block=arm == "cross_block",
    ).to(ctx.device)
    if dict(payload.get("refiner_contract", {})) != refiner.contract():
        raise RuntimeError("Round 3 evaluation refiner contract mismatch")
    for key, expected in {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_model_state_hash": parent.model_state_hash,
        "parent_adapter_state_hash": parent.adapter_state_hash,
        "config_hash": common._canonical_hash(ctx.cfg),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "statistics_hash": ctx.statistics.hash,
        "round3_settings_hash": common._canonical_hash(settings),
    }.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Round 3 evaluation checkpoint {key} mismatch")
    refiner.load_state_dict(payload["refiner_state"], strict=True)
    if module_state_hash(refiner) != payload.get("refiner_state_hash"):
        raise RuntimeError("Round 3 evaluation refiner state hash mismatch")
    refiner.eval()
    for parameter in refiner.parameters():
        parameter.requires_grad_(False)
    return refiner


def _aggregate(rows: Sequence[Mapping[str, Any]], *, metrics_key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        result[split] = {}
        for history in (4, 8):
            selected = [
                {**row, "metrics": row[metrics_key]}
                for row in rows
                if str(row["split"]) == split and int(row["history_frames"]) == history
            ]
            result[split][f"H{history}"] = common.aggregate_generated_rows(selected)
    return result


def _evaluate_one(ctx: Any, *, arm: str, scope: str, refiner_path: Path | None) -> dict[str, Any]:
    parent_path, _decision, parent = _load_parent(ctx)
    refiner = None if refiner_path is None else _load_refiner_for_evaluation(
        ctx, refiner_path, arm=arm, parent_path=parent_path, parent=parent
    )
    output_root = ctx.output_dir / ROUND / f"evaluation_{scope}" / arm
    rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    draws = tuple(int(value) for value in (
        ctx.cfg["evaluation"]["quick_draws"] if scope == "quick" else ctx.cfg["evaluation"]["final_draws"]
    ))
    try:
        for split, dataset, item in common._evaluation_items(ctx, scope):
            history_coordinates: list[Tensor] = []
            for history in (4, 8):
                coordinates: list[Tensor] = []
                for draw in draws:
                    indices = (int(item["dataset_index"]),)
                    common._sync(ctx.device)
                    started = time.perf_counter()
                    raw, decoded, coordinate, observed, generation, noise_sha = _free_decode(
                        ctx, parent, dataset, indices, history, draw=draw
                    )
                    observed_frames = observed_frame_mask(observed)
                    if refiner is None:
                        refined = raw
                        correction = raw.new_zeros(())
                    else:
                        refined_output = refiner(
                            raw,
                            decoded.h.float(),
                            decoded.v.float(),
                            frame_mask=decoded.frame_mask,
                            time_ps=decoded.time_ps,
                            atom_mask=coordinate.loss_mask,
                            observed_frames=observed_frames,
                            abid=coordinate.abid,
                        )
                        refined = refined_output.x_refined
                        correction = refined_output.delta_x.float().square().mean().sqrt()
                    common._sync(ctx.device)
                    elapsed = time.perf_counter() - started
                    observed_atom = _atom_observed_mask(observed_frames, coordinate.abid)
                    if not torch.equal(refined[observed_atom], coordinate.x.float()[observed_atom]):
                        raise RuntimeError("Round 3 evaluation refiner altered observed output")
                    raw_metrics = common.reassessment_trajectory_metrics(raw, coordinate.x.float(), coordinate, history)
                    refined_metrics = common.reassessment_trajectory_metrics(refined, coordinate.x.float(), coordinate, history)
                    raw_metrics["future"]["amplitude_error_angstrom"] = common._amplitude_error(raw_metrics)
                    refined_metrics["future"]["amplitude_error_angstrom"] = common._amplitude_error(refined_metrics)
                    name = f"{split}__{item['sample_id']}__H{history}__draw{draw}".replace("/", "_")
                    prediction_root = output_root / "predictions"
                    raw_path = prediction_root / f"{name}__raw.npz"
                    refined_path = raw_path if refiner is None else prediction_root / f"{name}__refined.npz"
                    common._save_prediction(raw_path, raw)
                    if refiner is not None:
                        common._save_prediction(refined_path, refined)
                    rows.append({
                        "schema": f"{common.SCHEMA}.round3.generation_row.v1",
                        "round": 3,
                        "label": arm,
                        "split": split,
                        "sample_id": str(item["sample_id"]),
                        "system": str(item["system"]),
                        "replica": str(item["replica"]),
                        "window": int(item["window"]),
                        "history_frames": history,
                        "steps": int(ctx.cfg["evaluation"]["euler_steps"]),
                        "draw": draw,
                        "noise_sha256": noise_sha,
                        "parent_checkpoint": str(parent_path),
                        "parent_checkpoint_sha256": parent.checkpoint_sha256,
                        "parent_model_state_hash": parent.model_state_hash,
                        "parent_adapter_state_hash": parent.adapter_state_hash,
                        "refiner_checkpoint": None if refiner_path is None else str(refiner_path),
                        "refiner_checkpoint_sha256": None if refiner_path is None else common.sha256_file(refiner_path),
                        "refiner_state_hash": None if refiner is None else module_state_hash(refiner),
                        "generation": {**generation, "wall_seconds_including_decode_and_refiner": elapsed},
                        "raw_metrics": raw_metrics,
                        "refined_metrics": refined_metrics,
                        "metrics": refined_metrics,
                        "raw_prediction_coordinates": str(raw_path),
                        "refined_prediction_coordinates": str(refined_path),
                        "refiner_correction_l2_angstrom": float(correction.detach().cpu()),
                        "observed_coordinates_exact": True,
                        "test_payload_opened": False,
                    })
                    coordinates.append(refined.detach())
                if len(coordinates) > 1:
                    diversity_rows.append({
                        "split": split,
                        "sample_id": str(item["sample_id"]),
                        "system": str(item["system"]),
                        "history_frames": history,
                        "diversity": common.future_diversity(torch.stack(coordinates), coordinate, history),
                    })
                history_coordinates.extend(coordinates)
        rows_path = output_root / "generation_rows.jsonl"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("".join(json.dumps(common._safe(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (output_root / "diversity.json").write_text(json.dumps(common._safe({"rows": diversity_rows}), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        latency = [float(row["generation"]["wall_seconds_including_decode_and_refiner"]) for row in rows]
        result = {
            "schema": f"{common.SCHEMA}.round3.evaluation.v1",
            "status": "PASS",
            "round": 3,
            "label": arm,
            "scope": scope,
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": parent.checkpoint_sha256,
            "parent_model_state_hash": parent.model_state_hash,
            "adapter_identity": parent.adapter is parent.model.adapter,
            "parent_adapter_state_hash": parent.adapter_state_hash,
            "refiner_checkpoint": None if refiner_path is None else str(refiner_path),
            "refiner_checkpoint_sha256": None if refiner_path is None else common.sha256_file(refiner_path),
            "refiner_state_hash": None if refiner is None else module_state_hash(refiner),
            "refiner_contract": None if refiner is None else refiner.contract(),
            "parameter_count": int(sum(parameter.numel() for parameter in parent.model.parameters()) + (0 if refiner is None else sum(parameter.numel() for parameter in refiner.parameters()))),
            "row_count": len(rows),
            "aggregates": _aggregate(rows, metrics_key="refined_metrics"),
            "raw_aggregates": _aggregate(rows, metrics_key="raw_metrics"),
            "diversity_rows": len(diversity_rows),
            "generation_latency_seconds": {
                "mean": float(np.mean(latency)),
                "p90": float(np.quantile(latency, 0.9)),
                "includes_source": False,
                "includes_euler_decode_refiner": True,
            },
            "rows": str(rows_path),
            "test_payload_opened": False,
        }
        common._write_json(output_root / "summary.json", result)
        return result
    finally:
        del parent
        if refiner is not None:
            del refiner
        torch.cuda.empty_cache()


def _evaluate(ctx: Any, scope: str) -> dict[str, Any]:
    checkpoints = _checkpoints(ctx)
    results = {
        "parent": _evaluate_one(ctx, arm="parent", scope=scope, refiner_path=None),
        "local": _evaluate_one(ctx, arm="local", scope=scope, refiner_path=checkpoints["local"]),
        "cross_block": _evaluate_one(ctx, arm="cross_block", scope=scope, refiner_path=checkpoints["cross_block"]),
    }
    return {
        "schema": f"{common.SCHEMA}.round3.evaluation_pair.v1",
        "scope": scope,
        "arms": results,
        "sequential_loading": True,
        "raw_and_refined_saved": True,
        "test_payload_opened": False,
    }


def _system_boundary_error(summary: Mapping[str, Any], history: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in summary["aggregates"]["valid"][f"H{history}"]["system_rows"]:
        metrics = row["future"]
        values = (
            metrics.get("displacement.groups.block_boundary.prediction_displacement_angstrom.mean"),
            metrics.get("displacement.groups.within_block.prediction_displacement_angstrom.mean"),
            metrics.get("displacement.groups.block_boundary.target_displacement_angstrom.mean"),
            metrics.get("displacement.groups.within_block.target_displacement_angstrom.mean"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            continue
        prediction_within = float(values[1])
        target_within = float(values[3])
        if prediction_within <= 1.0e-8 or target_within <= 1.0e-8:
            continue
        prediction_ratio = float(values[0]) / prediction_within
        target_ratio = float(values[2]) / target_within
        if prediction_ratio <= 0.0 or target_ratio <= 0.0:
            continue
        result[str(row["system"])] = abs(math.log(prediction_ratio / target_ratio))
    return result


def _boundary_comparison(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    improvement: dict[str, float] = {}
    directions: dict[str, int] = {}
    baseline: dict[str, float] = {}
    for history in (4, 8):
        first = _system_boundary_error(control, history)
        second = _system_boundary_error(candidate, history)
        shared = sorted(set(first) & set(second))
        if len(shared) < 1:
            raise RuntimeError(f"Round 3 has no applicable boundary systems at H{history}")
        first_mean = float(np.mean([first[key] for key in shared]))
        second_mean = float(np.mean([second[key] for key in shared]))
        key = f"H{history}"
        baseline[key] = first_mean
        improvement[key] = (first_mean - second_mean) / max(abs(first_mean), 1.0e-12)
        directions[key] = sum(second[item] < first[item] for item in shared)
    return improvement, directions, baseline


def _motion_values(summary: Mapping[str, Any], history: int) -> dict[str, float]:
    """Return finite prediction RMSF values, including a genuine value of zero."""

    report = motion_arm_report(summary, history, label="round3")
    return {
        system: float(item["prediction"])
        for system, item in report["values"].items()
        if item.get("prediction") is not None
    }


def _guardrails(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for history in (4, 8):
        parent_bond = common._metric(parent, history, "bond_rmse")
        candidate_bond = common._metric(candidate, history, "bond_rmse")
        parent_contact = common._metric(parent, history, "contact_f1")
        candidate_contact = common._metric(candidate, history, "contact_f1")
        parent_amp = common._metric(parent, history, "amplitude_error_angstrom")
        candidate_amp = common._metric(candidate, history, "amplitude_error_angstrom")
        first_motion = motion_arm_report(parent, history, label="parent")
        second_motion = motion_arm_report(candidate, history, label="candidate")
        motion = paired_motion_report(
            second_motion,
            first_motion,
            candidate_label="candidate",
            reference_label="parent",
            expected_systems=sorted(
                set(first_motion["actual_systems"]) | set(second_motion["actual_systems"])
            ),
            threshold=MOTION_RATIO_THRESHOLD,
        )
        result[f"H{history}"] = {
            "bond_relative_change": (candidate_bond - parent_bond) / max(abs(parent_bond), 1.0e-12),
            "contact_f1_change": candidate_contact - parent_contact,
            "amplitude_relative_change": (candidate_amp - parent_amp) / max(abs(parent_amp), 1.0e-12),
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
    return result


def _pass_primary(improvement: Mapping[str, float], directions: Mapping[str, int], cfg: Mapping[str, Any]) -> bool:
    threshold = float(cfg["decision"]["primary_relative_improvement"])
    tolerance = float(cfg["decision"]["opposite_trend_tolerance"])
    minimum = int(cfg["decision"]["minimum_systems_same_direction"])
    return (
        float(np.mean(list(improvement.values()))) >= threshold
        and min(improvement.values()) > -tolerance
        and min(directions.values()) >= minimum
    )


def _pass_guardrails(guards: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return all(
        float(values["bond_relative_change"]) <= float(cfg["decision"]["bond_relative_worsening"])
        and float(values["contact_f1_change"]) >= -float(cfg["decision"]["contact_f1_absolute_drop"])
        and float(values["amplitude_relative_change"]) <= float(cfg["decision"]["amplitude_relative_worsening"])
        and values.get("motion_status") == "PASS"
        and values["median_motion_ratio"] is not None
        and float(values["median_motion_ratio"]) >= MOTION_RATIO_THRESHOLD
        for values in guards.values()
    )


def _evidence(ctx: Any, scope: str, checkpoints: Mapping[str, Path], parent_path: Path) -> dict[str, Any]:
    result = {}
    for arm in EVALUATION_ARMS:
        path = ctx.output_dir / ROUND / f"evaluation_{scope}" / arm / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Round 3 {scope} evaluation is missing for {arm}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("parent_checkpoint_sha256") != common.sha256_file(parent_path):
            raise RuntimeError(f"Round 3 {scope} evaluation is stale for {arm}")
        if arm in checkpoints and summary.get("refiner_checkpoint_sha256") != common.sha256_file(checkpoints[arm]):
            raise RuntimeError(f"Round 3 {scope} refiner evaluation is stale for {arm}")
        result[arm] = summary
    return result


def _report(ctx: Any, decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ctx.output_dir / ROUND / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, history in zip(axes, (4, 8)):
        local = _system_boundary_error(evidence["local"], history)
        cross = _system_boundary_error(evidence["cross_block"], history)
        pairs = [(local[key], cross[key]) for key in sorted(set(local) & set(cross))]
        if pairs:
            first, second = zip(*pairs)
            bound = max((*first, *second, 1.0e-8))
            axis.scatter(first, second)
            axis.plot((0.0, bound), (0.0, bound), linestyle="--", color="black")
        axis.set(title=f"H{history}", xlabel="L boundary-ratio error", ylabel="T boundary-ratio error")
    figure.suptitle("Round 3 per-system cross-block comparison")
    figure.tight_layout()
    figure.savefig(plot_dir / "boundary_pair.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, history in zip(axes, (4, 8)):
        labels = []
        values = []
        for arm in EVALUATION_ARMS:
            metrics = evidence[arm]["aggregates"]["valid"][f"H{history}"]["system_equal"]
            labels.append(arm)
            values.append(float(metrics["displacement.groups.block_boundary.prediction_displacement_angstrom.mean"]))
        axis.bar(labels, values)
        axis.set(title=f"H{history}", ylabel="block-boundary displacement (A)")
    figure.suptitle("Round 3 absolute boundary motion")
    figure.tight_layout()
    figure.savefig(plot_dir / "boundary_motion.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for arm in TRAIN_ARMS:
        path = ctx.output_dir / ROUND / arm / "train_history.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        axis.plot([row["added_successful_updates"] for row in rows], [row["loss"] for row in rows], label=arm)
    axis.set(title="Round 3 matched refiner training", xlabel="added successful update", ylabel="refiner loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "training_loss.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Round 3: local versus cross-block temporal refinement",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        f"- Parent checkpoint SHA256: `{decision['parent_checkpoint_sha256']}`",
        f"- Added refiner updates per arm: {decision['successful_updates_per_arm']}",
        "",
        "| H | T vs L E_boundary improvement | T direction | L vs P E_boundary improvement | L direction |",
        "|---|---:|---:|---:|---:|",
    ]
    for history in (4, 8):
        key = f"H{history}"
        lines.append(
            f"| {key} | {decision['cross_vs_local']['relative_improvement'][key]:.6f} | "
            f"{decision['cross_vs_local']['systems_same_direction'][key]} | "
            f"{decision['local_vs_parent']['relative_improvement'][key]:.6f} | "
            f"{decision['local_vs_parent']['systems_same_direction'][key]} |"
        )
    lines.extend(("", decision["reason"], "", f"Remaining risk: {decision['remaining_risk']}", ""))
    (ctx.output_dir / ROUND / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _decide(ctx: Any) -> dict[str, Any]:
    checkpoints = _checkpoints(ctx)
    parent_path, _parent_decision = _round2_parent_path(ctx)
    quick = _evidence(ctx, "quick", checkpoints, parent_path)
    local_improvement, local_direction, local_baseline = _boundary_comparison(quick["parent"], quick["local"])
    cross_improvement, cross_direction, cross_baseline = _boundary_comparison(quick["local"], quick["cross_block"])
    quick_promising = _pass_primary(local_improvement, local_direction, ctx.cfg) or _pass_primary(cross_improvement, cross_direction, ctx.cfg)
    final_root = ctx.output_dir / ROUND / "evaluation_final"
    if quick_promising and not all((final_root / arm / "summary.json").is_file() for arm in EVALUATION_ARMS):
        result = {
            "schema": f"{common.SCHEMA}.decision.v1",
            "round": 3,
            "status": "NEEDS_FINAL_EVALUATION",
            "parent_checkpoint": str(parent_path),
            "unique_variable": "same coordinate-supervised refiner, block-local versus cross-block adjacent-frame mask",
            "local_vs_parent": {"relative_improvement": local_improvement, "systems_same_direction": local_direction, "baseline_error": local_baseline},
            "cross_vs_local": {"relative_improvement": cross_improvement, "systems_same_direction": cross_direction, "baseline_error": cross_baseline},
            "next_command": "evaluate-round3 --scope final, then decide-round3",
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "decision.json", result)
        return result
    evidence = _evidence(ctx, "final", checkpoints, parent_path) if quick_promising else quick
    scope = "final" if quick_promising else "quick"
    local_improvement, local_direction, local_baseline = _boundary_comparison(evidence["parent"], evidence["local"])
    cross_improvement, cross_direction, cross_baseline = _boundary_comparison(evidence["local"], evidence["cross_block"])
    cross_parent_improvement, cross_parent_direction, cross_parent_baseline = _boundary_comparison(evidence["parent"], evidence["cross_block"])
    local_guard = _guardrails(evidence["parent"], evidence["local"])
    cross_guard = _guardrails(evidence["parent"], evidence["cross_block"])
    local_primary = _pass_primary(local_improvement, local_direction, ctx.cfg)
    cross_primary = _pass_primary(cross_improvement, cross_direction, ctx.cfg)
    local_guard_pass = _pass_guardrails(local_guard, ctx.cfg)
    cross_guard_pass = _pass_guardrails(cross_guard, ctx.cfg)
    local_pass = local_primary and local_guard_pass
    cross_pass = (
        cross_primary
        and min(cross_parent_improvement.values()) > -float(ctx.cfg["decision"]["opposite_trend_tolerance"])
        and cross_guard_pass
    )
    guardrail_tradeoff = (local_primary and not local_guard_pass) or (cross_primary and not cross_guard_pass)
    train = json.loads((ctx.output_dir / ROUND / "train_summary.json").read_text(encoding="utf-8"))
    positive = max(
        float(np.mean(list(local_improvement.values()))),
        float(np.mean(list(cross_improvement.values()))),
    ) > 0.0
    if cross_pass:
        status, selected_arm, reason = (
            "KEEP_CROSS_BLOCK",
            "cross_block",
            "cross-block adjacency improved E_boundary relative to the equal local refiner without parent guardrail failure",
        )
    elif local_pass:
        status, selected_arm, reason = (
            "KEEP_LOCAL_ONLY",
            "local",
            "local refinement improved E_boundary relative to P; cross-block adjacency was not supported",
        )
    elif guardrail_tradeoff:
        status, selected_arm, reason = (
            "TRADEOFF",
            "parent",
            "a refiner met the boundary threshold but violated a registered geometry or motion guardrail; retain P",
        )
    elif positive and not bool(train.get("common_extension_applied")):
        status, selected_arm, reason = (
            "NEEDS_COMMON_EXTENSION",
            None,
            "boundary evidence is positive but below the pre-registered paired threshold; use the one common extension",
        )
    elif positive:
        status, selected_arm, reason = (
            "INCONCLUSIVE",
            "parent",
            "the matched extension remained heterogeneous; retain the unrefined parent",
        )
    else:
        status, selected_arm, reason = (
            "KEEP_PARENT",
            "parent",
            "neither local nor cross-block refinement improved the free-generation boundary metric",
        )
    selected_refiner = None if selected_arm in (None, "parent") else checkpoints[selected_arm]
    result = {
        "schema": f"{common.SCHEMA}.decision.v1",
        "round": 3,
        "status": status,
        "evaluation_scope": scope,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": common.sha256_file(parent_path),
        "unique_variable": "same coordinate-supervised refiner, block-local versus cross-block adjacent-frame mask",
        "primary_metric": "system-equal abs(log((boundary/within)_prediction / (boundary/within)_MD))",
        "near_static_policy": "exclude non-positive or near-zero ratio denominators from E_boundary; retain absolute displacement metrics in rows and plots",
        "primary_pass": {"local_vs_parent": local_primary, "cross_vs_local": cross_primary},
        "guardrails_pass": {"local_vs_parent": local_guard_pass, "cross_vs_parent": cross_guard_pass},
        "guardrail_tradeoff": guardrail_tradeoff,
        "local_vs_parent": {"relative_improvement": local_improvement, "systems_same_direction": local_direction, "baseline_error": local_baseline, "guardrails": local_guard},
        "cross_vs_local": {"relative_improvement": cross_improvement, "systems_same_direction": cross_direction, "baseline_error": cross_baseline},
        "cross_vs_parent": {"relative_improvement": cross_parent_improvement, "systems_same_direction": cross_parent_direction, "baseline_error": cross_parent_baseline, "guardrails": cross_guard},
        "selected_arm": selected_arm,
        "selected_checkpoint": str(parent_path),
        "selected_checkpoint_sha256": common.sha256_file(parent_path),
        "selected_refiner_checkpoint": None if selected_refiner is None else str(selected_refiner),
        "selected_refiner_checkpoint_sha256": None if selected_refiner is None else common.sha256_file(selected_refiner),
        "refiner_checkpoints": {arm: {"path": str(path), "sha256": common.sha256_file(path)} for arm, path in checkpoints.items()},
        "initialization_hash": train["initialization_hash"],
        "successful_updates_per_arm": train["successful_updates_per_arm"],
        "future_atom_frame_tokens_per_arm": train["added_future_atom_frame_tokens_per_arm"],
        "gpu_hours": train["gpu_hours"],
        "config_hash": train["config_hash"],
        "data_hash": train["data_hash"],
        "codec_state_hash": train["codec_state_hash"],
        "statistics_hash": train["statistics_hash"],
        "code": train["code"],
        "reason": reason,
        "remaining_risk": "one training seed; target-coupled endpoint post-training is not a proof of autoregressive rollout stability",
        "next_command": "train-round3 --resume; then evaluate-round3 --scope quick and decide-round3" if status == "NEEDS_COMMON_EXTENSION" else None,
        "test_payload_opened": False,
    }
    common._write_json(ctx.output_dir / ROUND / "decision.json", result)
    if status not in ("NEEDS_COMMON_EXTENSION", "NEEDS_FINAL_EVALUATION"):
        _report(ctx, result, evidence)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_architecture_sequential_v1.yaml")
    parser.add_argument("--stage", required=True, choices=("profile-round3", "smoke-round3", "train-round3", "evaluate-round3", "decide-round3"))
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
    ctx = common._load_context(cfg, output_dir, torch.device(args.device))
    if not (output_dir / "selection.json").is_file():
        raise FileNotFoundError("Round 3 requires the already frozen sequential selection")
    if args.stage == "profile-round3":
        result = _profile(ctx)
    elif args.stage == "smoke-round3":
        result = _smoke(ctx)
    elif args.stage == "train-round3":
        result = _train(ctx, resume=bool(args.resume))
    elif args.stage == "evaluate-round3":
        result = _evaluate(ctx, args.scope)
    else:
        result = _decide(ctx)
    print(json.dumps(common._safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
