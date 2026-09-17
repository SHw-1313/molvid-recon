#!/usr/bin/env python
"""Run Round 4: paired clean versus error-corrupted observed history training.

The selected Round-2 DiT is the parent because Round 3 retained no refiner.
Both arms preserve the calibrated future-bond auxiliary and all source, dt,
tokenization, optimizer, and RF choices.  The only training-time difference is
whether the observed H4/H8 coordinates are independently perturbed by the
pre-registered 50/25/25 percent (0/0.02/0.05 Angstrom) distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.dit_architecture_sequential_v1 import load_sequential_checkpoint
from module.dit_geometry_supervision import FutureBondAuxiliary
from module.dit_history_corruption import (
    DEFAULT_PROBABILITIES,
    DEFAULT_SIGMAS_ANGSTROM,
    HistoryConditionedLatents,
    combine_clean_target_with_condition,
    make_history_corruption_views,
    sample_history_sigmas,
)
from module.latent_flow_source import build_observed_center
from module.state_detail_latent_adapter import DiTLatentBatch
from scripts.run_state_detail_codec_v2_t1 import TrajectoryCappedBatchSampler
from scripts.run_state_detail_dit_pilot import _encode_batch
from trainer.dit_trainer import DiTTrainer, module_state_hash

from scripts import run_dit_architecture_round2 as round2
from scripts import run_dit_architecture_sequential_v1 as common


ROUND = "round4"
ARMS = ("control", "candidate")
FFN_NORM_SOURCE = "post_adaln"
# Keep the common schema expected by the shared independent evaluation loader.
CHECKPOINT_SCHEMA = "pvb.dit.architecture_sequential.v1.checkpoint.v1"


@dataclass(frozen=True)
class PreparedRound4Batch:
    coordinate_batch: Any
    target_view: Any
    observed: DiTLatentBatch
    source_center: Any
    corruption: HistoryConditionedLatents


@dataclass(frozen=True)
class RolloutTrack:
    system: str
    replica: str
    first_window: int
    sample_ids: tuple[str, str]
    global_x: Tensor
    global_time_ps: Tensor
    dt_ps: float
    template: Any


def _settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = cfg.get(ROUND)
    if not isinstance(value, Mapping):
        raise ValueError("round4 configuration is missing")
    required = (
        "max_atom_frame_tokens",
        "validation_max_atom_frame_tokens",
        "evaluation_recovery_reserve_gpu_hours",
        "corruption",
        "rollout",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"round4 configuration lacks {missing}")
    result = dict(value)
    corruption = result["corruption"]
    rollout = result["rollout"]
    if not isinstance(corruption, Mapping) or not isinstance(rollout, Mapping):
        raise ValueError("round4 corruption and rollout settings must be mappings")
    result["corruption"] = dict(corruption)
    result["rollout"] = dict(rollout)
    if int(result["max_atom_frame_tokens"]) < 56672:
        raise ValueError("round4 token cap cannot reject the largest frozen train clip")
    if int(result["validation_max_atom_frame_tokens"]) < 56672:
        raise ValueError("round4 validation cap cannot reject the largest frozen validation clip")
    if int(result["validation_max_atom_frame_tokens"]) > int(result["max_atom_frame_tokens"]):
        raise ValueError("round4 validation cap cannot exceed the profiled train cap")
    if tuple(float(value) for value in corruption.get("probabilities", ())) != DEFAULT_PROBABILITIES:
        raise ValueError("round4 corruption probabilities must be [0.5, 0.25, 0.25]")
    if tuple(float(value) for value in corruption.get("sigmas_angstrom", ())) != DEFAULT_SIGMAS_ANGSTROM:
        raise ValueError("round4 corruption sigmas must be [0.0, 0.02, 0.05]")
    if int(corruption.get("seed", -1)) < 0:
        raise ValueError("round4 corruption requires a non-negative independent seed")
    if int(rollout.get("history_frames", -1)) != 8:
        raise ValueError("round4 rollout fixes H8")
    if int(rollout.get("segments", -1)) != 3 or tuple(int(value) for value in rollout.get("draws", ())) != (0, 1):
        raise ValueError("round4 rollout fixes three generated segments and two draws")
    if int(rollout.get("clip_len", -1)) != 16 or int(rollout.get("window_stride_frames", -1)) != 16:
        raise ValueError("round4 rollout requires two contiguous 16-frame source clips")
    if int(rollout.get("steps", -1)) != int(cfg["evaluation"]["euler_steps"]):
        raise ValueError("round4 rollout must use the frozen Euler-step count")
    if float(result["evaluation_recovery_reserve_gpu_hours"]) < 0.0:
        raise ValueError("round4 evaluation/recovery reserve must be non-negative")
    return result


def _code_provenance() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    files = (
        "scripts/run_dit_architecture_round4.py",
        "scripts/run_dit_architecture_round2.py",
        "scripts/run_dit_architecture_sequential_v1.py",
        "module/dit_history_corruption.py",
        "module/dit_geometry_supervision.py",
        "evaluation/dit_architecture_sequential_v1.py",
        "config/dit_architecture_sequential_v1.yaml",
        "config/dit_architecture_sequential_v1.neibu.yaml",
        "tests/test_dit_architecture_round4_cuda.py",
    )
    hashes = {
        relative: common.sha256_file(PROJECT_ROOT / relative)
        if (PROJECT_ROOT / relative).is_file()
        else None
        for relative in files
    }
    return {"head": head, "files": hashes, "source_hash": common._canonical_hash(hashes)}


def _seed(ctx: Any) -> int:
    return int(ctx.cfg["evaluation"]["training_seed_base"]) + 4


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
    cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    total = 0
    for update in range(int(updates)):
        indices, _ = common._next_batch(sampler, cursor, cache)
        total += common._batch_tokens(specs, indices, histories[update % len(histories)])
    return total


def _round2_parent_path(ctx: Any) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    round2_path = ctx.output_dir / "round2" / "decision.json"
    round3_path = ctx.output_dir / "round3" / "decision.json"
    if not round2_path.is_file() or not round3_path.is_file():
        raise FileNotFoundError("round4 requires completed Round 2 and Round 3 decisions")
    round2_decision = json.loads(round2_path.read_text(encoding="utf-8"))
    round3_decision = json.loads(round3_path.read_text(encoding="utf-8"))
    if round2_decision.get("status") != "KEEP" or round2_decision.get("selected_arm") != "candidate":
        raise RuntimeError("round4 requires the selected Round 2 geometry-supervised candidate")
    if round3_decision.get("status") != "KEEP_PARENT" or round3_decision.get("selected_arm") != "parent":
        raise RuntimeError("this R4 run is authorized only after Round 3 retained no refiner")
    path = Path(str(round3_decision.get("selected_checkpoint", "")))
    if not path.is_file():
        raise RuntimeError("Round 3's retained parent checkpoint is unavailable")
    digest = common.sha256_file(path)
    if digest != round3_decision.get("selected_checkpoint_sha256"):
        raise RuntimeError("Round 3 parent checkpoint changed after its decision")
    if path != Path(str(round2_decision.get("selected_checkpoint", ""))):
        raise RuntimeError("Round 3 parent does not match the selected Round 2 checkpoint")
    if digest != round2_decision.get("selected_checkpoint_sha256"):
        raise RuntimeError("Round 2 selected checkpoint hash disagrees with Round 3 parent")
    return path, round2_decision, round3_decision


def _load_parent(ctx: Any) -> tuple[Path, Mapping[str, Any], dict[str, Tensor], str, dict[str, Any]]:
    path, _round2_decision, _round3_decision = _round2_parent_path(ctx)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_contract", {}).get("ffn_norm_source") != FFN_NORM_SOURCE:
        raise RuntimeError("round4 requires the selected post-AdaLN Round 2 lineage")
    geometry = dict(payload.get("config", {}).get("metadata", {}).get("geometry_supervision", {}))
    if geometry.get("enabled") is not True or float(geometry.get("lambda_bond", 0.0)) <= 0.0:
        raise RuntimeError("round4 must retain the selected Round 2 future-bond auxiliary")
    if geometry.get("future_target_clean") is not True:
        raise RuntimeError("Round 2 geometry contract did not certify clean future targets")
    state, state_hash = common._state_from_payload(
        ctx.cfg, payload, ctx.device, ffn_norm_source=FFN_NORM_SOURCE
    )
    return path, payload, state, state_hash, geometry


def _new_trainer(
    ctx: Any,
    *,
    state: Mapping[str, Tensor],
    state_hash: str,
    max_steps: int,
    arm: str,
    geometry: Mapping[str, Any],
) -> DiTTrainer:
    trainer = common._new_trainer(
        ctx,
        ffn_norm_source=FFN_NORM_SOURCE,
        max_steps=int(max_steps),
        initial_state=state,
        initial_hash=state_hash,
        arm=arm,
        round_name=ROUND,
    )
    trainer.config.metadata["geometry_supervision"] = dict(geometry)
    trainer.config.metadata["history_corruption"] = {
        "enabled": arm == "candidate",
        "schema": "pvb.dit.architecture_sequential.v1.history_corruption.v1",
        "distribution": "per_clip_50pct_clean_25pct_sigma0.02_25pct_sigma0.05",
        "observed_coordinates_only": True,
        "future_target_clean": True,
        "rng_stream": "independent_of_rf_tau_epsilon",
        "cache_policy": "disabled_online_corruption",
    }
    trainer.config.metadata["temporal_refiner"] = None
    return trainer


def _clean_sigmas(batch: Any) -> Tensor:
    return torch.zeros((int(batch.batch_size),), device=batch.x.device, dtype=batch.x.dtype)


def _prepare_round4_batch(
    ctx: Any,
    dataset: Any,
    indices: Sequence[int],
    adapter: Any,
    history: int,
    *,
    sigmas: Tensor,
    epsilon: Tensor,
) -> PreparedRound4Batch:
    """Encode a clean target once and inject condition fields only at H4/H8."""

    clean_latent, _batch_cpu, coordinate = _encode_batch(
        dataset,
        indices,
        codec=ctx.codec,
        adapter=adapter,
        data_hash=ctx.data_hash,
        device=ctx.device,
    )
    clean_target = adapter.pack(
        clean_latent,
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
        origin_from_latent=True,
        loss_mask=coordinate.loss_mask,
    )
    views = make_history_corruption_views(
        coordinate,
        history_frames=int(history),
        sigma_per_sample_angstrom=sigmas,
        epsilon=epsilon,
    )
    # The zero path intentionally reuses the exact clean target fields.  This
    # makes sigma=0 the same numerical condition as the clean control rather
    # than merely an approximately equal re-encode.
    if bool(torch.all(sigmas == 0.0)):
        condition_latent = clean_target
    else:
        with torch.no_grad():
            condition_raw = ctx.codec.model.encode(views.condition_view)
        condition_latent = adapter.pack(
            condition_raw,
            codec_hash=ctx.codec.codec_state_hash,
            data_hash=ctx.data_hash,
            origin_from_latent=True,
            loss_mask=views.condition_view.loss_mask,
        )
    corruption = combine_clean_target_with_condition(
        clean_target,
        condition_latent,
        views=views,
        history_frames=int(history),
    )
    center = common._source_center(
        ctx,
        views.condition_view,
        corruption.observed,
        adapter,
        int(history),
    )
    return PreparedRound4Batch(
        coordinate_batch=coordinate,
        target_view=views.target_view,
        observed=corruption.observed,
        source_center=center,
        corruption=corruption,
    )


def _candidate_corruption(
    batch_size: int,
    shape: torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    settings: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    corruption = settings["corruption"]
    sigmas = sample_history_sigmas(
        int(batch_size),
        generator=generator,
        device=device,
        dtype=dtype,
        probabilities=tuple(float(value) for value in corruption["probabilities"]),
        sigmas_angstrom=tuple(float(value) for value in corruption["sigmas_angstrom"]),
    )
    epsilon = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return sigmas, epsilon


def _geometry_auxiliary(
    ctx: Any,
    trainer: DiTTrainer,
    target_view: Any,
    history: int,
    geometry: Mapping[str, Any],
) -> FutureBondAuxiliary:
    return FutureBondAuxiliary(
        adapter=trainer.adapter,
        statistics=ctx.statistics_device,
        codec=ctx.codec.model,
        coordinate_batch=target_view,
        history_frames=int(history),
        lambda_bond=float(geometry["lambda_bond"]),
        tau_threshold=float(geometry["tau_threshold"]),
    )


def _run_pair_update(
    ctx: Any,
    *,
    trainers: Mapping[str, DiTTrainer],
    dataset: Any,
    indices: Sequence[int],
    history: int,
    rf_generators: Mapping[str, torch.Generator],
    corruption_generator: torch.Generator,
    geometry: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    settings = _settings(ctx.cfg)
    prepared: dict[str, PreparedRound4Batch] = {}
    # Both arms read the same source records and use the same H.  Only the
    # candidate consumes its separate corruption RNG stream.
    for arm in ARMS:
        if arm == "control":
            # The fresh clean coordinate batch is obtained inside preparation;
            # use a deterministic zero epsilon that cannot consume an RNG draw.
            probe = _encode_batch(
                dataset,
                indices,
                codec=ctx.codec,
                adapter=trainers[arm].adapter,
                data_hash=ctx.data_hash,
                device=ctx.device,
            )[2]
            sigmas = _clean_sigmas(probe)
            epsilon = torch.zeros_like(probe.x)
            # Reuse the already-read clip only through the deterministic public
            # preparation path.  It has no cache and does not mutate the dataset.
            del probe
        else:
            probe = _encode_batch(
                dataset,
                indices,
                codec=ctx.codec,
                adapter=trainers[arm].adapter,
                data_hash=ctx.data_hash,
                device=ctx.device,
            )[2]
            sigmas, epsilon = _candidate_corruption(
                int(probe.batch_size),
                probe.x.shape,
                dtype=probe.x.dtype,
                device=ctx.device,
                generator=corruption_generator,
                settings=settings,
            )
            del probe
        prepared[arm] = _prepare_round4_batch(
            ctx,
            dataset,
            indices,
            trainers[arm].adapter,
            int(history),
            sigmas=sigmas,
            epsilon=epsilon,
        )
    rows: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        item = prepared[arm]
        row = trainers[arm].train_step(
            item.observed,
            generator=rf_generators[arm],
            source_center=item.source_center,
            auxiliary_objective=_geometry_auxiliary(
                ctx, trainers[arm], item.target_view, int(history), geometry
            ),
        )
        rows[arm] = row
    if not torch.equal(rf_generators["control"].get_state(), rf_generators["candidate"].get_state()):
        raise RuntimeError("Round 4 RF tau/epsilon generators diverged")
    metadata = prepared["candidate"].corruption.views.metadata()
    metadata["control_sigma_all_zero"] = bool(torch.all(prepared["control"].corruption.views.sigma_per_sample_angstrom == 0.0))
    return rows, metadata


def _prior_training_hours(ctx: Any) -> float:
    result = 0.0
    for prior in ("round1", "round2", "round3"):
        path = ctx.output_dir / prior / "train_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"round4 requires {prior} training evidence")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"round4 refuses non-passing {prior} training evidence")
        result += float(summary.get("gpu_hours", 0.0))
    return result


def _profile_generation(ctx: Any, trainer: DiTTrainer, indices: Sequence[int], history: int) -> dict[str, Any]:
    probe = _encode_batch(
        ctx.data.train,
        (int(indices[0]),),
        codec=ctx.codec,
        adapter=trainer.adapter,
        data_hash=ctx.data_hash,
        device=ctx.device,
    )[2]
    prepared = _prepare_round4_batch(
        ctx,
        ctx.data.train,
        (int(indices[0]),),
        trainer.adapter,
        int(history),
        sigmas=_clean_sigmas(probe),
        epsilon=torch.zeros_like(probe.x),
    )
    del probe
    normalized = ctx.statistics_device.normalize(prepared.observed)
    sample_id = str(prepared.coordinate_batch.sample_id[0])
    noise, metadata = common.make_fixed_noise(
        normalized,
        seed=common.generation_seed(int(ctx.cfg["evaluation"]["generation_seed"]), sample_id, int(history), 0),
    )
    common._sync(ctx.device)
    started = time.perf_counter()
    generated, generation = common.generate_fixed_noise_latent(
        trainer.model,
        trainer.adapter,
        prepared.observed,
        ctx.statistics_device,
        noise=noise,
        steps=int(ctx.cfg["evaluation"]["euler_steps"]),
        source_center=prepared.source_center,
        source_mode="conditional",
    )
    decoded = ctx.codec.model.decode(generated).x_hat.float()
    metrics = common.reassessment_trajectory_metrics(
        decoded, prepared.target_view.x.float(), prepared.target_view, int(history)
    )
    common._sync(ctx.device)
    if not bool(torch.isfinite(decoded).all()) or not bool(generation["observed_clamp_exact"]):
        raise RuntimeError("round4 profile generation produced non-finite output or a clamp failure")
    return {
        "seconds": time.perf_counter() - started,
        "sample_id": sample_id,
        "history_frames": int(history),
        "bond_rmse": metrics["future"]["bond_rmse"],
        "contact_f1": metrics["future"].get("contact_f1"),
        "noise_sha256": metadata["noise_sha256"],
        "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
    }


def _profile(ctx: Any) -> dict[str, Any]:
    path = ctx.output_dir / ROUND / "budget.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS":
            raise RuntimeError("a non-passing round4 budget exists; refusing to overwrite it")
        parent_path, _, _ = _round2_parent_path(ctx)
        if existing.get("parent_checkpoint_sha256") != common.sha256_file(parent_path):
            raise RuntimeError("existing round4 budget uses a different parent")
        if existing.get("round4_settings_hash") != common._canonical_hash(_settings(ctx.cfg)):
            raise RuntimeError("existing round4 budget uses different settings")
        return existing
    settings = _settings(ctx.cfg)
    parent_path, parent, state, state_hash, geometry = _load_parent(ctx)
    total = int(ctx.cfg["budget"]["profile_warmup_updates"]) + int(ctx.cfg["budget"]["profile_measured_updates"])
    trainers = {
        arm: _new_trainer(
            ctx, state=state, state_hash=state_hash, max_steps=int(parent["step"]) + total,
            arm=arm, geometry=geometry,
        )
        for arm in ARMS
    }
    for trainer in trainers.values():
        common._restore_parent(trainer, parent, expected_codec_hash=ctx.codec.codec_state_hash, expected_statistics_hash=ctx.statistics.hash)
    common.assert_independent_trainers(trainers["control"], trainers["candidate"])
    frozen = trainers["control"].frozen_state_hashes()
    sampler = _sampler(ctx, seed=_seed(ctx))
    cursor = {"epoch": 0, "batch_index": 0}
    cache: dict[str, Any] = {}
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    rf_generators = {arm: torch.Generator(device=ctx.device).manual_seed(_seed(ctx)) for arm in ARMS}
    corruption_generator = torch.Generator(device=ctx.device).manual_seed(int(settings["corruption"]["seed"]))
    warmup = int(ctx.cfg["budget"]["profile_warmup_updates"])
    pair_seconds: list[float] = []
    corruption_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(ctx.device)
    try:
        for update in range(total):
            indices, _ = common._next_batch(sampler, cursor, cache)
            history = histories[update % len(histories)]
            common._sync(ctx.device)
            started = time.perf_counter()
            _rows, corruption = _run_pair_update(
                ctx, trainers=trainers, dataset=ctx.data.train, indices=indices, history=history,
                rf_generators=rf_generators, corruption_generator=corruption_generator, geometry=geometry,
            )
            common._sync(ctx.device)
            if update >= warmup:
                pair_seconds.append(time.perf_counter() - started)
                corruption_rows.append(corruption)
        generation = _profile_generation(ctx, trainers["control"], (int(indices[0]),), histories[-1])
        if any(trainer.frozen_state_hashes() != frozen for trainer in trainers.values()):
            raise RuntimeError("round4 profile altered frozen codec or frame encoder")
        p90 = float(np.quantile(pair_seconds, 0.9))
        prior_hours = _prior_training_hours(ctx)
        total_seconds = float(ctx.cfg["budget"]["gpu_hours_total"]) * 3600.0
        reserve = float(settings["evaluation_recovery_reserve_gpu_hours"]) * 3600.0
        available = total_seconds - prior_hours * 3600.0 - reserve
        choices = []
        selected = None
        for updates in (int(value) for value in ctx.cfg["budget"]["round_candidate_updates"]):
            extension = int(math.ceil(updates * float(ctx.cfg["budget"]["inconclusive_extension_fraction"])))
            estimate = p90 * (updates + extension)
            choices.append({
                "added_updates_per_arm": updates,
                "extension_updates_per_arm": extension,
                "estimated_pair_gpu_seconds_including_extension": estimate,
                "feasible": estimate <= available,
            })
            if selected is None and estimate <= available:
                selected = updates
        if selected is None:
            raise RuntimeError("round4 cannot fit even the smallest paired continuation budget")
        extension = int(math.ceil(selected * float(ctx.cfg["budget"]["inconclusive_extension_fraction"])))
        result = {
            "schema": f"{common.SCHEMA}.round4.budget.v1",
            "status": "PASS",
            "frozen_before_training": True,
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": common.sha256_file(parent_path),
            "parent_model_state_hash": state_hash,
            "round4_settings": settings,
            "round4_settings_hash": common._canonical_hash(settings),
            "geometry_supervision": geometry,
            "profile": {
                "warmup_updates": warmup,
                "measured_updates": len(pair_seconds),
                "pair_seconds": pair_seconds,
                "pair_p90_seconds_per_update": p90,
                "generation": generation,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
                "online_corruption_cache": "disabled",
                "corruption_samples": corruption_rows,
            },
            "prior_round_training_seconds": prior_hours * 3600.0,
            "gpu_seconds_total": total_seconds,
            "evaluation_recovery_reserve_seconds": reserve,
            "future_rounds_training_reserve_seconds": 0.0,
            "available_round4_training_seconds": available,
            "choices": choices,
            "selected_round4_added_updates": selected,
            "selected_round4_extension_added_updates": extension,
            "selected_round4_future_atom_frame_tokens_per_arm": _tokens_for_updates(ctx, selected, _seed(ctx)),
            "selected_round4_extended_total_updates": selected + extension,
            "selected_round4_extended_total_future_atom_frame_tokens_per_arm": _tokens_for_updates(ctx, selected + extension, _seed(ctx)),
            "device": common._cuda_info(ctx.device),
            "code": _code_provenance(),
            "test_payload_opened": False,
        }
        result["selected_round4_extension_future_atom_frame_tokens_per_arm"] = (
            int(result["selected_round4_extended_total_future_atom_frame_tokens_per_arm"])
            - int(result["selected_round4_future_atom_frame_tokens_per_arm"])
        )
        common._write_json(path, result)
        common._write_json(ctx.output_dir / ROUND / "profile.json", result)
        return result
    finally:
        del trainers
        torch.cuda.empty_cache()


def _load_budget(ctx: Any) -> dict[str, Any]:
    path = ctx.output_dir / ROUND / "budget.json"
    if not path.is_file():
        raise FileNotFoundError("round4 requires a frozen passing profile/budget")
    budget = json.loads(path.read_text(encoding="utf-8"))
    parent_path, _, _ = _round2_parent_path(ctx)
    if budget.get("status") != "PASS" or budget.get("frozen_before_training") is not True:
        raise RuntimeError("round4 budget is not a frozen passing budget")
    if budget.get("parent_checkpoint_sha256") != common.sha256_file(parent_path):
        raise RuntimeError("round4 budget parent differs from selected R2 parent")
    if budget.get("round4_settings_hash") != common._canonical_hash(_settings(ctx.cfg)):
        raise RuntimeError("round4 settings differ from frozen budget")
    return budget


def _exposure_plan(ctx: Any, budget: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    base = int(budget["selected_round4_added_updates"])
    extension = int(budget["selected_round4_extension_added_updates"])
    summary_path = ctx.output_dir / ROUND / "train_summary.json"
    decision_path = ctx.output_dir / ROUND / "decision.json"
    prior: dict[str, Any] = {}
    apply_extension = False
    if resume and summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else {}
        if decision.get("status") != "NEEDS_COMMON_EXTENSION":
            raise RuntimeError("completed round4 can resume only for its one common extension")
        if prior.get("status") != "PASS" or bool(prior.get("common_extension_applied")):
            raise RuntimeError("round4 extension provenance is invalid")
        if int(prior.get("added_successful_updates_per_arm", -1)) != base:
            raise RuntimeError("round4 base exposure differs from its frozen budget")
        apply_extension = True
    return {
        "base_delta": base,
        "extension_delta": extension,
        "target_delta": base + (extension if apply_extension else 0),
        "common_extension": apply_extension,
        "prior_summary": prior,
    }


def _checkpoint_payload(
    trainer: DiTTrainer,
    *,
    arm: str,
    parent_path: Path,
    parent_sha256: str,
    parent_model_hash: str,
    cursor: Mapping[str, int],
    rf_generator: torch.Generator,
    corruption_generator: torch.Generator | None,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["sequential"] = {
        "schema": CHECKPOINT_SCHEMA,
        "round": ROUND,
        "arm": arm,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_model_state_hash": parent_model_hash,
        "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        "training_generator_state": rf_generator.get_state(),
        "corruption_generator_state": None if corruption_generator is None else corruption_generator.get_state(),
        "added_future_atom_frame_tokens": int(added_tokens),
        "added_successful_updates": int(added_updates),
        "schedule_hash": str(schedule_hash),
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
    rf_generator: torch.Generator,
    corruption_generator: torch.Generator | None,
    added_tokens: int,
    added_updates: int,
    schedule_hash: str,
    unique_variable: Mapping[str, Any],
    final: bool,
) -> Path:
    payload = _checkpoint_payload(
        trainer, arm=arm, parent_path=parent_path, parent_sha256=parent_sha256,
        parent_model_hash=parent_model_hash, cursor=cursor, rf_generator=rf_generator,
        corruption_generator=corruption_generator, added_tokens=added_tokens,
        added_updates=added_updates, schedule_hash=schedule_hash, unique_variable=unique_variable,
    )
    name = "checkpoint_final.pt" if final else f"checkpoint_step{trainer.step:06d}.pt"
    path = directory / name
    common._atomic_torch_save(path, payload)
    common._atomic_torch_save(directory / "latest.pt", payload)
    return path


def _resume_paths(ctx: Any) -> dict[str, Path]:
    root = ctx.output_dir / ROUND
    latest = {arm: root / arm / "latest.pt" for arm in ARMS}
    if not all(path.is_file() for path in latest.values()):
        raise FileNotFoundError("round4 resume requires both arm checkpoints")

    def exposure(path: Path, arm: str) -> tuple[Any, ...]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sequential = payload.get("sequential", {})
        if sequential.get("schema") != CHECKPOINT_SCHEMA or sequential.get("round") != ROUND or sequential.get("arm") != arm:
            raise RuntimeError(f"round4 resume checkpoint contract mismatch for {arm}")
        if not isinstance(sequential.get("training_generator_state"), Tensor):
            raise RuntimeError("round4 checkpoint lacks RF generator state")
        if arm == "candidate" and not isinstance(sequential.get("corruption_generator_state"), Tensor):
            raise RuntimeError("round4 candidate checkpoint lacks independent corruption RNG")
        return (
            int(payload.get("successful_optimizer_updates", -1)),
            int(sequential.get("added_successful_updates", -1)),
            int(sequential.get("added_future_atom_frame_tokens", -1)),
            dict(sequential.get("cursor", {})),
        )

    values = {arm: exposure(path, arm) for arm, path in latest.items()}
    if values["control"] == values["candidate"]:
        return latest
    names = None
    for arm in ARMS:
        current = {path.name for path in (root / arm).glob("checkpoint_step*.pt") if path.is_file()}
        names = current if names is None else names & current
    for name in sorted(names or (), reverse=True):
        paths = {arm: root / arm / name for arm in ARMS}
        current = {arm: exposure(path, arm) for arm, path in paths.items()}
        if current["control"] == current["candidate"]:
            return paths
    raise RuntimeError("round4 has no common atomic paired checkpoint")


def _unique_variable(settings: Mapping[str, Any], geometry: Mapping[str, Any], extension: bool) -> dict[str, Any]:
    return {
        "ffn_norm_source": FFN_NORM_SOURCE,
        "geometry_supervision": dict(geometry),
        "training_history": {
            "control": "clean_observed_coordinates",
            "candidate": "per_clip_50pct_clean_25pct_isotropic_0.02A_25pct_isotropic_0.05A",
            "future_target": "clean",
            "condition_and_source_origin": "condition_view_observed_frame0_centroid",
            "corruption_rng": "independent_of_rf_tau_epsilon",
            "cache": "disabled",
        },
        "round4_settings_hash": common._canonical_hash(settings),
        "common_extension": bool(extension),
    }


def _restore_resume(
    ctx: Any,
    trainers: Mapping[str, DiTTrainer],
    rf_generators: Mapping[str, torch.Generator],
    corruption_generator: torch.Generator,
    parent_step: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    paths = _resume_paths(ctx)
    sequential: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        payload = trainers[arm].load_checkpoint(paths[arm], map_location=ctx.device)
        sequential[arm] = payload["sequential"]
        rf_generators[arm].set_state(sequential[arm]["training_generator_state"].detach().cpu())
    corruption_generator.set_state(sequential["candidate"]["corruption_generator_state"].detach().cpu())
    if not torch.equal(rf_generators["control"].get_state(), rf_generators["candidate"].get_state()):
        raise RuntimeError("round4 resumed RF generators differ")
    if sequential["control"]["cursor"] != sequential["candidate"]["cursor"]:
        raise RuntimeError("round4 resumed sampler cursors differ")
    added = int(sequential["control"]["added_successful_updates"])
    if int(sequential["candidate"]["added_successful_updates"]) != added:
        raise RuntimeError("round4 resumed added update counts differ")
    tokens = {arm: int(sequential[arm]["added_future_atom_frame_tokens"]) for arm in ARMS}
    if tokens["control"] != tokens["candidate"]:
        raise RuntimeError("round4 resumed token exposures differ")
    if any(trainer.successful_updates != parent_step + added for trainer in trainers.values()):
        raise RuntimeError("round4 resumed model step differs from paired exposure")
    cursor = {"epoch": int(sequential["control"]["cursor"]["epoch"]), "batch_index": int(sequential["control"]["cursor"]["batch_index"])}
    return cursor, tokens, added


def _train(ctx: Any, *, resume: bool) -> dict[str, Any]:
    smoke_path = ctx.output_dir / ROUND / "smoke.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("round4 requires a passing real-clip CUDA smoke")
    budget = _load_budget(ctx)
    exposure = _exposure_plan(ctx, budget, resume=resume)
    settings = _settings(ctx.cfg)
    parent_path, parent, state, state_hash, geometry = _load_parent(ctx)
    parent_sha = common.sha256_file(parent_path)
    parent_step = int(parent["step"])
    target_step = parent_step + int(exposure["target_delta"])
    trainers = {
        arm: _new_trainer(ctx, state=state, state_hash=state_hash, max_steps=target_step, arm=arm, geometry=geometry)
        for arm in ARMS
    }
    for trainer in trainers.values():
        common._restore_parent(trainer, parent, expected_codec_hash=ctx.codec.codec_state_hash, expected_statistics_hash=ctx.statistics.hash)
    common.assert_independent_trainers(trainers["control"], trainers["candidate"])
    frozen = trainers["control"].frozen_state_hashes()
    rf_generators = {arm: torch.Generator(device=ctx.device).manual_seed(_seed(ctx)) for arm in ARMS}
    corruption_generator = torch.Generator(device=ctx.device).manual_seed(int(settings["corruption"]["seed"]))
    sampler = _sampler(ctx, seed=_seed(ctx))
    cursor = {"epoch": 0, "batch_index": 0}
    cache: dict[str, Any] = {}
    specs = ctx.data.train.clip_spec_table()
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    tokens = {arm: 0 for arm in ARMS}
    start_added = 0
    if resume and any((ctx.output_dir / ROUND / arm / "latest.pt").is_file() for arm in ARMS):
        cursor, tokens, start_added = _restore_resume(ctx, trainers, rf_generators, corruption_generator, parent_step)
        for arm in ARMS:
            common._truncate_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", step_key="added_successful_updates", maximum_step=start_added)
            common._truncate_jsonl(ctx.output_dir / ROUND / arm / "validation_history.jsonl", step_key="added_successful_updates", maximum_step=start_added)
    elif any((ctx.output_dir / ROUND / arm / "latest.pt").exists() for arm in ARMS):
        raise RuntimeError("fresh round4 training refuses existing arm checkpoints")
    compute = {arm: 0.0 for arm in ARMS}
    corruption_counts = {"clean_samples": 0, "sigma_0_02_samples": 0, "sigma_0_05_samples": 0}
    schedule_hash = ""
    torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    try:
        while trainers["control"].successful_updates < target_step:
            added = trainers["control"].successful_updates - parent_step
            if trainers["candidate"].successful_updates - parent_step != added:
                raise RuntimeError("round4 arms have unequal successful updates")
            indices, schedule_hash = common._next_batch(sampler, cursor, cache)
            history = histories[added % len(histories)]
            token_count = common._batch_tokens(specs, indices, history)
            common._sync(ctx.device)
            pair_started = time.perf_counter()
            rows, corruption = _run_pair_update(
                ctx, trainers=trainers, dataset=ctx.data.train, indices=indices, history=history,
                rf_generators=rf_generators, corruption_generator=corruption_generator, geometry=geometry,
            )
            common._sync(ctx.device)
            pair_elapsed = time.perf_counter() - pair_started
            # The pair is serialized on one audited GPU; preserve actual wall
            # cost while retaining equal logical per-arm exposure accounting.
            compute["control"] += pair_elapsed / 2.0
            compute["candidate"] += pair_elapsed / 2.0
            for name in corruption_counts:
                corruption_counts[name] += int(corruption[name])
            for arm in ARMS:
                tokens[arm] += token_count
                row = rows[arm]
                row.update({
                    "round": 4,
                    "arm": arm,
                    "history_frames": int(history),
                    "batch_tokens": int(token_count),
                    "added_future_atom_frame_tokens": int(tokens[arm]),
                    "added_successful_updates": trainers[arm].successful_updates - parent_step,
                    "parent_step": parent_step,
                    "lambda_bond": float(geometry["lambda_bond"]),
                    "history_corruption": corruption if arm == "candidate" else {"mode": "clean", "sigma_all_zero": True},
                })
                common._append_jsonl(ctx.output_dir / ROUND / arm / "train_history.jsonl", row)
            current = trainers["control"].successful_updates - parent_step
            if current % int(ctx.cfg["schedule"]["validation_interval"]) == 0:
                for arm in ARMS:
                    # Release inactive allocator slabs from the paired optimizer
                    # states before the same selected validation clips are batched
                    # under the registered memory cap.
                    torch.cuda.empty_cache()
                    validation = common._validation_rf(
                        ctx, trainers[arm], step=trainers[arm].successful_updates,
                        max_atom_frame_tokens=int(settings["validation_max_atom_frame_tokens"]),
                    )
                    validation.update({"arm": arm, "added_successful_updates": current, "validation_condition": "clean_history_diagnostic"})
                    common._append_jsonl(ctx.output_dir / ROUND / arm / "validation_history.jsonl", validation)
            if current % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0:
                for arm in ARMS:
                    _save_checkpoint(
                        ctx.output_dir / ROUND / arm, trainers[arm], arm=arm, parent_path=parent_path,
                        parent_sha256=parent_sha, parent_model_hash=state_hash, cursor=cursor,
                        rf_generator=rf_generators[arm], corruption_generator=corruption_generator if arm == "candidate" else None,
                        added_tokens=tokens[arm], added_updates=current, schedule_hash=schedule_hash,
                        unique_variable=_unique_variable(settings, geometry, bool(exposure["common_extension"])), final=False,
                    )
        expected_tokens = int(
            budget["selected_round4_extended_total_future_atom_frame_tokens_per_arm"]
            if exposure["common_extension"] else budget["selected_round4_future_atom_frame_tokens_per_arm"]
        )
        if tokens["control"] != tokens["candidate"] or tokens["control"] != expected_tokens:
            raise RuntimeError("round4 pair differs from frozen equal token exposure")
        checkpoints = {
            arm: _save_checkpoint(
                ctx.output_dir / ROUND / arm, trainers[arm], arm=arm, parent_path=parent_path,
                parent_sha256=parent_sha, parent_model_hash=state_hash, cursor=cursor,
                rf_generator=rf_generators[arm], corruption_generator=corruption_generator if arm == "candidate" else None,
                added_tokens=tokens[arm], added_updates=int(exposure["target_delta"]), schedule_hash=schedule_hash,
                unique_variable=_unique_variable(settings, geometry, bool(exposure["common_extension"])), final=True,
            )
            for arm in ARMS
        }
        if any(trainer.frozen_state_hashes() != frozen for trainer in trainers.values()):
            raise RuntimeError("round4 training changed frozen codec or frame encoder")
        elapsed = time.perf_counter() - started
        prior = dict(exposure["prior_summary"])
        result = {
            "schema": f"{common.SCHEMA}.round4.training.v1",
            "status": "PASS",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": parent_sha,
            "parent_model_state_hash": state_hash,
            "initialization_hash": state_hash,
            "unique_variable": "clean observed history versus online independently corrupted observed history; clean future RF and bond targets",
            "geometry_supervision": geometry,
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
            "successful_optimizer_updates": {arm: trainer.successful_updates for arm, trainer in trainers.items()},
            "added_future_atom_frame_tokens_per_arm": expected_tokens,
            "updates_this_invocation_per_arm": int(exposure["target_delta"]) - start_added,
            "wall_seconds_this_invocation": elapsed,
            "wall_seconds": float(prior.get("wall_seconds", 0.0)) + elapsed,
            "gpu_hours_this_invocation": elapsed / 3600.0,
            "gpu_hours": float(prior.get("gpu_hours", 0.0)) + elapsed / 3600.0,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
            "arm_compute_seconds_this_invocation": compute,
            "arm_compute_seconds": {arm: float(prior.get("arm_compute_seconds", {}).get(arm, 0.0)) + compute[arm] for arm in ARMS},
            "history_corruption_counts": corruption_counts,
            "online_corruption_cache": "disabled",
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
    finally:
        del trainers
        torch.cuda.empty_cache()


def _smoke(ctx: Any) -> dict[str, Any]:
    settings = _settings(ctx.cfg)
    parent_path, parent, state, state_hash, geometry = _load_parent(ctx)
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    items = list(selection["smoke"])
    trainers = {
        arm: _new_trainer(ctx, state=state, state_hash=state_hash, max_steps=int(parent["step"]) + len(items), arm=arm, geometry=geometry)
        for arm in ARMS
    }
    for trainer in trainers.values():
        common._restore_parent(trainer, parent, expected_codec_hash=ctx.codec.codec_state_hash, expected_statistics_hash=ctx.statistics.hash)
    common.assert_independent_trainers(trainers["control"], trainers["candidate"])
    dataset = ClipMMapDataset(Path(selection["smoke_root"]) / "train")
    rf_generators = {arm: torch.Generator(device=ctx.device).manual_seed(_seed(ctx)) for arm in ARMS}
    corruption_generator = torch.Generator(device=ctx.device).manual_seed(int(settings["corruption"]["seed"]))
    histories = tuple(int(value) for value in ctx.cfg["schedule"]["history_order"])
    rows = []
    corruptions = []
    try:
        for offset, item in enumerate(items):
            history = histories[offset % len(histories)]
            train_rows, corruption = _run_pair_update(
                ctx, trainers=trainers, dataset=dataset, indices=(int(item["dataset_index"]),), history=history,
                rf_generators=rf_generators, corruption_generator=corruption_generator, geometry=geometry,
            )
            corruptions.append(corruption)
            for evaluation_history in (4, 8):
                for arm in ARMS:
                    probe = _encode_batch(dataset, (int(item["dataset_index"]),), codec=ctx.codec, adapter=trainers[arm].adapter, data_hash=ctx.data_hash, device=ctx.device)[2]
                    prepared = _prepare_round4_batch(
                        ctx, dataset, (int(item["dataset_index"]),), trainers[arm].adapter, evaluation_history,
                        sigmas=_clean_sigmas(probe), epsilon=torch.zeros_like(probe.x),
                    )
                    del probe
                    normalized = ctx.statistics_device.normalize(prepared.observed)
                    noise, noise_meta = common.make_fixed_noise(
                        normalized,
                        seed=common.generation_seed(20260914, str(item["sample_id"]), evaluation_history, 0),
                    )
                    generated, generation = common.generate_fixed_noise_latent(
                        trainers[arm].model, trainers[arm].adapter, prepared.observed, ctx.statistics_device,
                        noise=noise, steps=8, source_center=prepared.source_center, source_mode="conditional",
                    )
                    decoded = ctx.codec.model.decode(generated).x_hat.float()
                    rows.append({
                        "sample_id": str(item["sample_id"]), "system": str(item["system"]), "replica": str(item["replica"]),
                        "training_history_frames": history, "history_frames": evaluation_history, "arm": arm,
                        "finite": bool(torch.isfinite(decoded).all()), "observed_clamp_exact": bool(generation["observed_clamp_exact"]),
                        "noise_sha256": noise_meta["noise_sha256"], "train_loss": train_rows[arm]["loss"],
                    })
        passed = bool(rows) and all(row["finite"] and row["observed_clamp_exact"] for row in rows) and all(
            trainer.successful_updates == int(parent["step"]) + len(items) for trainer in trainers.values()
        )
        result = {
            "schema": f"{common.SCHEMA}.round4.smoke.v1", "status": "PASS" if passed else "FAIL",
            "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": common.sha256_file(parent_path),
            "systems": list(ctx.cfg["evaluation"]["smoke"]["systems"]), "trajectory_count": len(items),
            "optimizer_updates_per_arm": len(items), "batching": "serial_single_clip", "rows": rows,
            "candidate_corruption": corruptions, "ranking_use": False, "device": common._cuda_info(ctx.device),
            "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "smoke.json", result)
        return result
    finally:
        dataset.close()
        del trainers
        torch.cuda.empty_cache()


def _checkpoints(ctx: Any) -> dict[str, Path]:
    path = ctx.output_dir / ROUND / "train_summary.json"
    if not path.is_file():
        raise FileNotFoundError("round4 training summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("round4 training did not pass")
    result = {}
    for arm in ARMS:
        checkpoint = Path(summary["checkpoints"][arm]["path"])
        if common.sha256_file(checkpoint) != summary["checkpoints"][arm]["sha256"]:
            raise RuntimeError(f"round4 {arm} checkpoint changed")
        result[arm] = checkpoint
    return result


def _block_positions_cuda(x: Tensor, block_id: Tensor) -> Tensor:
    result = torch.empty_like(x)
    for identifier in torch.unique(block_id):
        members = block_id == identifier
        selected = x[:, members]
        # Boolean-index assignment does not broadcast a [T, 1, 3] block
        # center to the selected atom axis on CUDA. Materialize that one
        # broadcast explicitly so generated rollout conditions retain the
        # canonical repeated per-block positions.
        result[:, members] = selected.mean(dim=1, keepdim=True).expand_as(selected)
    return result


def _inverse_record_coordinates(record: Mapping[str, Any], *, device: torch.device) -> Tensor:
    alignment = record.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("rollout clip lacks its recorded preprocessing alignment")
    x = torch.as_tensor(record["x"], device=device, dtype=torch.float32)
    rotation = torch.as_tensor(alignment["rotation"], device=device, dtype=torch.float32)
    translation = torch.as_tensor(alignment["translation"], device=device, dtype=torch.float32)
    center = torch.as_tensor(alignment["center_angstrom"], device=device, dtype=torch.float32)
    if x.shape[0] != 16 or rotation.shape != (16, 3, 3) or translation.shape != (16, 3) or center.shape != (3,):
        raise ValueError("rollout alignment has an unexpected shape")
    centered = x + center.view(1, 1, 3) - translation.unsqueeze(1)
    return torch.einsum("tnc,tdc->tnd", centered, rotation)


def _load_rollout_track(ctx: Any, *, system: str, replica: str, first_window: int) -> RolloutTrack:
    settings = _settings(ctx.cfg)["rollout"]
    ids = {str(row[0]): int(index) for index, row in enumerate(ctx.data.valid._index)}
    sample_ids = tuple(f"{system}_{replica}_w{int(first_window + offset):06d}" for offset in range(2))
    if any(sample_id not in ids for sample_id in sample_ids):
        raise FileNotFoundError(f"missing contiguous valid rollout windows: {sample_ids}")
    records = [ctx.data.valid[ids[sample_id]] for sample_id in sample_ids]
    reference = records[0]
    for record in records[1:]:
        for key in ("atype", "btype", "block_id", "component_id", "atom_source_index", "bond_index", "atom_identity"):
            if not np.array_equal(np.asarray(record[key]), np.asarray(reference[key])):
                raise ValueError(f"contiguous rollout windows disagree on topology {key}")
    raw = [_inverse_record_coordinates(record, device=ctx.device) for record in records]
    local_times = [torch.as_tensor(record["time_ps"], device=ctx.device, dtype=torch.float32) for record in records]
    deltas = [torch.as_tensor(record["delta_time_ps"], device=ctx.device, dtype=torch.float32) for record in records]
    if any(value.numel() != 15 or not bool(torch.allclose(value, value[:1].expand_as(value), rtol=1.0e-6, atol=1.0e-5)) for value in deltas):
        raise ValueError("rollout requires constant physical dt")
    dt = float(deltas[0][0].detach().cpu())
    if not all(abs(float(value[0].detach().cpu()) - dt) <= 1.0e-5 for value in deltas):
        raise ValueError("rollout windows have inconsistent dt")
    global_times = [
        local_times[offset] + float(first_window + offset) * float(settings["window_stride_frames"]) * dt
        for offset in range(2)
    ]
    if not bool(torch.isclose(global_times[1][0], global_times[0][-1] + dt, rtol=1.0e-6, atol=1.0e-4)):
        raise ValueError("rollout windows are not contiguous under the registered stride")
    batch_cpu = collate_clip_records([reference])
    ctx.codec.model.prepare_batch(batch_cpu)
    template = batch_cpu.to(ctx.device, non_blocking=True)
    return RolloutTrack(
        system=str(system), replica=str(replica), first_window=int(first_window), sample_ids=sample_ids,
        global_x=torch.cat(raw, dim=0), global_time_ps=torch.cat(global_times, dim=0), dt_ps=dt, template=template,
    )


def _rollout_condition_batch(track: RolloutTrack, prefix: Tensor) -> Any:
    if prefix.shape != (8, track.global_x.shape[1], 3):
        raise ValueError("round4 rollout prefix must contain exactly eight atom frames")
    x = torch.cat((prefix, prefix[-1:].expand(8, -1, -1)), dim=0)
    return replace(track.template, x=x, bpos=_block_positions_cuda(x, track.template.block_id))


def _rollout_metric_batch(track: RolloutTrack, start: int) -> Any:
    x = track.global_x[int(start):int(start) + 16]
    if x.shape[0] != 16:
        raise ValueError("rollout metric window exceeds its verified 32-frame track")
    return replace(track.template, x=x, bpos=_block_positions_cuda(x, track.template.block_id))


def _prepare_inference(ctx: Any, adapter: Any, coordinate: Any, history: int) -> tuple[DiTLatentBatch, Any]:
    with torch.no_grad():
        latent = ctx.codec.model.encode(coordinate)
    target = adapter.pack(
        latent, codec_hash=ctx.codec.codec_state_hash, data_hash=ctx.data_hash,
        origin_from_latent=True, loss_mask=coordinate.loss_mask,
    )
    observed = common._observed_batch(target, coordinate, adapter=adapter, history_frames=int(history))
    center = common._source_center(ctx, coordinate, observed, adapter, int(history))
    return observed, center


def _rollout_seed(ctx: Any, track: RolloutTrack, segment: int, draw: int) -> int:
    identity = f"{track.system}_{track.replica}_w{track.first_window:06d}_r4_rollout_segment{int(segment)}"
    return common.generation_seed(int(ctx.cfg["evaluation"]["generation_seed"]), identity, 8, int(draw))


def _run_rollout_arm(ctx: Any, *, arm: str, checkpoint: Path, tracks: Sequence[RolloutTrack]) -> list[dict[str, Any]]:
    loaded = load_sequential_checkpoint(
        checkpoint, device=ctx.device, expected_data_hash=ctx.data_hash,
        expected_codec_hash=ctx.codec.codec_state_hash, expected_statistics_hash=ctx.statistics.hash,
    )
    rows: list[dict[str, Any]] = []
    try:
        for track in tracks:
            for draw in (int(value) for value in _settings(ctx.cfg)["rollout"]["draws"]):
                by_prefix: dict[str, list[Tensor]] = {"true_prefix": [], "generated_prefix": []}
                for prefix_kind in ("true_prefix", "generated_prefix"):
                    generated_prefix = track.global_x[:8]
                    for segment in range(3):
                        prefix = track.global_x[segment * 8:(segment + 1) * 8] if prefix_kind == "true_prefix" else generated_prefix
                        condition = _rollout_condition_batch(track, prefix)
                        observed, center = _prepare_inference(ctx, loaded.adapter, condition, 8)
                        normalized = ctx.statistics_device.normalize(observed)
                        seed = _rollout_seed(ctx, track, segment, draw)
                        noise, noise_metadata = common.make_fixed_noise(normalized, seed=seed)
                        generated, generation = common.generate_fixed_noise_latent(
                            loaded.model, loaded.adapter, observed, ctx.statistics_device,
                            noise=noise, steps=int(_settings(ctx.cfg)["rollout"]["steps"]),
                            source_center=center, source_mode="conditional",
                        )
                        decoded = ctx.codec.model.decode(generated).x_hat.float()
                        if not bool(torch.isfinite(decoded).all()) or not bool(generation["observed_clamp_exact"]):
                            raise RuntimeError("round4 rollout generation failed finite/clamp checks")
                        # Coordinate-domain clamp makes the supplied prefix exactly
                        # the next segment's history; only the generated future is
                        # propagated for generated-prefix rollouts.
                        prediction = torch.cat((prefix, decoded[8:]), dim=0)
                        metric_batch = _rollout_metric_batch(track, segment * 8)
                        metrics = common.reassessment_trajectory_metrics(
                            prediction, metric_batch.x.float(), metric_batch, 8
                        )
                        future = prediction[8:].detach()
                        by_prefix[prefix_kind].append(future)
                        if prefix_kind == "generated_prefix":
                            generated_prefix = future
                        rows.append({
                            "schema": f"{common.SCHEMA}.round4.rollout_segment.v1",
                            "split": "valid", "arm": arm, "system": track.system, "replica": track.replica,
                            "window": track.first_window, "sample_id": track.sample_ids[0], "draw": draw,
                            "prefix_kind": prefix_kind, "segment": segment, "history_frames": 8,
                            "steps": int(_settings(ctx.cfg)["rollout"]["steps"]), "seed": seed,
                            "checkpoint": str(checkpoint), "checkpoint_sha256": loaded.checkpoint_sha256,
                            "model_state_hash": loaded.model_state_hash, "adapter_is_model_adapter": loaded.adapter is loaded.model.adapter,
                            "noise_sha256": noise_metadata["noise_sha256"], "generation": generation,
                            "global_frame_interval": [segment * 8, segment * 8 + 16],
                            "target_future_global_interval": [segment * 8 + 8, segment * 8 + 16],
                            "prefix_source": "true_previous_eight" if prefix_kind == "true_prefix" else "own_previous_generated_eight",
                            "metrics": metrics,
                        })
                if not torch.equal(by_prefix["true_prefix"][0], by_prefix["generated_prefix"][0]):
                    raise RuntimeError("round4 true/generated-prefix segment 1 diverged despite identical history and epsilon")
        return rows
    finally:
        del loaded
        torch.cuda.empty_cache()


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rollout_summary(rows: Sequence[Mapping[str, Any]], checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    expected = len(ARMS) * 8 * 2 * 2 * 3
    if len(rows) != expected:
        raise RuntimeError(f"round4 rollout produced {len(rows)} rows; expected {expected}")
    lookup: dict[tuple[str, str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["arm"]), str(row["system"]), int(row["draw"]), int(row["segment"]), str(row["prefix_kind"]))
        if key in lookup:
            raise RuntimeError("round4 rollout has duplicate arm/system/draw/segment/prefix row")
        lookup[key] = row
    per_system: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    section_groups: dict[str, dict[str, dict[str, list[float]]]] = {arm: {} for arm in ARMS}
    systems = sorted({str(row["system"]) for row in rows})
    for arm in ARMS:
        for prefix in ("true_prefix", "generated_prefix"):
            section_groups[arm][prefix] = {str(segment): {"bond_rmse": [], "contact_f1": []} for segment in range(3)}
        for system in systems:
            extras = []
            contact_extras = []
            true_bonds = []
            generated_bonds = []
            for draw in (0, 1):
                for segment in range(3):
                    true = lookup[(arm, system, draw, segment, "true_prefix")]["metrics"]["future"]
                    generated = lookup[(arm, system, draw, segment, "generated_prefix")]["metrics"]["future"]
                    for prefix, metric in (("true_prefix", true), ("generated_prefix", generated)):
                        bond = metric.get("bond_rmse")
                        contact = metric.get("contact_f1")
                        if isinstance(bond, (int, float)) and math.isfinite(float(bond)):
                            section_groups[arm][prefix][str(segment)]["bond_rmse"].append(float(bond))
                        if isinstance(contact, (int, float)) and math.isfinite(float(contact)):
                            section_groups[arm][prefix][str(segment)]["contact_f1"].append(float(contact))
                    if segment > 0:
                        bond_true, bond_generated = true.get("bond_rmse"), generated.get("bond_rmse")
                        if not isinstance(bond_true, (int, float)) or not isinstance(bond_generated, (int, float)):
                            raise RuntimeError("round4 rollout lacks finite future bond RMSE")
                        extras.append(max(0.0, float(bond_generated) - float(bond_true)))
                        true_bonds.append(float(bond_true))
                        generated_bonds.append(float(bond_generated))
                        contact_true, contact_generated = true.get("contact_f1"), generated.get("contact_f1")
                        if isinstance(contact_true, (int, float)) and isinstance(contact_generated, (int, float)):
                            contact_extras.append(max(0.0, float(contact_true) - float(contact_generated)))
            per_system[arm][system] = {
                "E_roll_bond": _mean(extras),
                "extra_contact_f1_error": _mean(contact_extras),
                "true_prefix_bond_rmse_sections_2_3": _mean(true_bonds),
                "generated_prefix_bond_rmse_sections_2_3": _mean(generated_bonds),
                "draw_count": 2,
                "section_count": 2,
            }
    aggregate = {
        arm: {
            "E_roll_bond": _mean([float(row["E_roll_bond"]) for row in per_system[arm].values() if row["E_roll_bond"] is not None]),
            "extra_contact_f1_error": _mean([float(row["extra_contact_f1_error"]) for row in per_system[arm].values() if row["extra_contact_f1_error"] is not None]),
            "true_prefix_bond_rmse_sections_2_3": _mean([float(row["true_prefix_bond_rmse_sections_2_3"]) for row in per_system[arm].values() if row["true_prefix_bond_rmse_sections_2_3"] is not None]),
            "generated_prefix_bond_rmse_sections_2_3": _mean([float(row["generated_prefix_bond_rmse_sections_2_3"]) for row in per_system[arm].values() if row["generated_prefix_bond_rmse_sections_2_3"] is not None]),
        }
        for arm in ARMS
    }
    groups = {
        arm: {
            prefix: {
                segment: {metric: _mean(values) for metric, values in metrics.items()}
                for segment, metrics in segments.items()
            }
            for prefix, segments in prefixes.items()
        }
        for arm, prefixes in section_groups.items()
    }
    return {
        "schema": f"{common.SCHEMA}.round4.rollout.v1", "status": "PASS", "system_count": len(systems),
        "systems": systems, "draws": [0, 1], "history_frames": 8, "segments": 3,
        "primary_metric": "mean_over_systems_draws_sections_2_3 max(0, bond_generated_prefix - bond_true_prefix)",
        "per_system": per_system, "aggregate": aggregate, "raw_section_groups": groups,
        "checkpoints": {arm: {"path": str(path), "sha256": common.sha256_file(path)} for arm, path in checkpoints.items()},
        "first_segment_true_and_generated_prefix_exact": True, "test_payload_opened": False,
    }


def _evaluate_rollout(ctx: Any, checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    selection = json.loads((ctx.output_dir / "selection.json").read_text(encoding="utf-8"))
    items = list(selection["quick_valid"])
    if len(items) != 8:
        raise RuntimeError("round4 rollout requires exactly the fixed eight validation systems")
    tracks = [
        _load_rollout_track(ctx, system=str(item["system"]), replica=str(item["replica"]), first_window=int(item["window"]))
        for item in items
    ]
    if len({track.system for track in tracks}) != 8:
        raise RuntimeError("round4 rollout selection must contain eight distinct validation systems")
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        rows.extend(_run_rollout_arm(ctx, arm=arm, checkpoint=checkpoints[arm], tracks=tracks))
    output = ctx.output_dir / ROUND / "evaluation_quick" / "rollout"
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "segment_rows.jsonl"
    rows_path.write_text("".join(json.dumps(common._safe(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = _rollout_summary(rows, checkpoints)
    summary.update({"rows": str(rows_path), "rows_sha256": common.sha256_file(rows_path), "window_contract": {
        "first_windows": {track.system: track.first_window for track in tracks}, "source_clip_count": 2,
        "source_clip_frames": 16, "global_frames": 32, "window_stride_frames": 16,
        "model_segment_condition": "eight supplied prefix frames plus repeat-last future fill", "artificial_history_noise": False,
    }})
    common._write_json(output / "summary.json", summary)
    return summary


def _evaluate_clean(ctx: Any, checkpoints: Mapping[str, Path], scope: str) -> dict[str, Any]:
    return {
        arm: common._evaluate_checkpoint(ctx, checkpoint=checkpoint, label=arm, round_name=ROUND, scope=scope)
        for arm, checkpoint in checkpoints.items()
    }


def _evaluate(ctx: Any, scope: str) -> dict[str, Any]:
    checkpoints = _checkpoints(ctx)
    clean = _evaluate_clean(ctx, checkpoints, scope)
    rollout = _evaluate_rollout(ctx, checkpoints) if scope == "quick" else None
    if scope == "final":
        rollout_path = ctx.output_dir / ROUND / "evaluation_quick" / "rollout" / "summary.json"
        if not rollout_path.is_file():
            raise FileNotFoundError("round4 final clean evaluation requires current quick rollout evidence")
        rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
        for arm, checkpoint in checkpoints.items():
            if rollout.get("checkpoints", {}).get(arm, {}).get("sha256") != common.sha256_file(checkpoint):
                raise RuntimeError("round4 quick rollout is stale for the current checkpoint")
    return {
        "schema": f"{common.SCHEMA}.round4.evaluation_pair.v1", "scope": scope,
        "arms": clean, "rollout": rollout, "sequential_loading": True,
        "test_payload_opened": False,
    }


def _evidence(ctx: Any, scope: str, checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    result = {}
    for arm, checkpoint in checkpoints.items():
        path = ctx.output_dir / ROUND / f"evaluation_{scope}" / arm / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"round4 {scope} clean evaluation is missing for {arm}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("checkpoint_sha256") != common.sha256_file(checkpoint):
            raise RuntimeError(f"round4 {scope} clean evaluation is stale for {arm}")
        result[arm] = summary
    return result


def _rollout_evidence(ctx: Any, checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    path = ctx.output_dir / ROUND / "evaluation_quick" / "rollout" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError("round4 rollout evidence is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or int(summary.get("system_count", 0)) != 8:
        raise RuntimeError("round4 rollout evidence did not pass for all eight systems")
    for arm, checkpoint in checkpoints.items():
        if summary.get("checkpoints", {}).get(arm, {}).get("sha256") != common.sha256_file(checkpoint):
            raise RuntimeError("round4 rollout evidence is stale")
    return summary


def _clean_guardrails(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for history in (4, 8):
        control, candidate = evidence["control"], evidence["candidate"]
        control_bond = common._metric(control, history, "bond_rmse")
        candidate_bond = common._metric(candidate, history, "bond_rmse")
        control_contact = common._metric(control, history, "contact_f1")
        candidate_contact = common._metric(candidate, history, "contact_f1")
        control_amp = common._metric(control, history, "amplitude_error_angstrom")
        candidate_amp = common._metric(candidate, history, "amplitude_error_angstrom")
        result[f"H{history}"] = {
            "bond_relative_change": (candidate_bond - control_bond) / max(abs(control_bond), 1.0e-12),
            "contact_f1_change": candidate_contact - control_contact,
            "amplitude_relative_change": (candidate_amp - control_amp) / max(abs(control_amp), 1.0e-12),
        }
    return result


def _guards_pass(guards: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    return all(
        float(row["bond_relative_change"]) <= float(cfg["decision"]["bond_relative_worsening"])
        and float(row["contact_f1_change"]) >= -float(cfg["decision"]["contact_f1_absolute_drop"])
        and float(row["amplitude_relative_change"]) <= float(cfg["decision"]["amplitude_relative_worsening"])
        for row in guards.values()
    )


def _rollout_comparison(rollout: Mapping[str, Any]) -> dict[str, Any]:
    per_system = rollout["per_system"]
    systems = sorted(set(per_system["control"]) & set(per_system["candidate"]))
    if len(systems) != 8:
        raise RuntimeError("round4 rollout comparison lacks paired eight-system evidence")
    control = {system: float(per_system["control"][system]["E_roll_bond"]) for system in systems}
    candidate = {system: float(per_system["candidate"][system]["E_roll_bond"]) for system in systems}
    mean_control = float(np.mean(list(control.values())))
    mean_candidate = float(np.mean(list(candidate.values())))
    true_control = float(rollout["aggregate"]["control"]["true_prefix_bond_rmse_sections_2_3"])
    true_candidate = float(rollout["aggregate"]["candidate"]["true_prefix_bond_rmse_sections_2_3"])
    return {
        "relative_improvement": (mean_control - mean_candidate) / max(abs(mean_control), 1.0e-12),
        "systems_same_direction": sum(candidate[system] < control[system] for system in systems),
        "control_mean_E_roll_bond": mean_control,
        "candidate_mean_E_roll_bond": mean_candidate,
        "per_system_control": control,
        "per_system_candidate": candidate,
        "true_prefix_bond_relative_change": (true_candidate - true_control) / max(abs(true_control), 1.0e-12),
        "true_prefix_bond_control": true_control,
        "true_prefix_bond_candidate": true_candidate,
        "extra_contact_f1_error_control": rollout["aggregate"]["control"].get("extra_contact_f1_error"),
        "extra_contact_f1_error_candidate": rollout["aggregate"]["candidate"].get("extra_contact_f1_error"),
    }


def _report(ctx: Any, decision: Mapping[str, Any]) -> None:
    comparison = decision["rollout_comparison"]
    lines = [
        "# Round 4: error-corrupted observed-history training", "",
        f"- Decision: `{decision['status']}`", f"- Selected arm: `{decision['selected_arm']}`",
        f"- Parent checkpoint SHA256: `{decision['parent_checkpoint_sha256']}`",
        f"- Primary E_roll_bond relative improvement: {comparison['relative_improvement']:.6f}",
        f"- Systems with lower extra generated-prefix bond degradation: {comparison['systems_same_direction']}/8", "",
        "| H | clean bond relative change | contact F1 change | amplitude relative change |",
        "|---|---:|---:|---:|",
    ]
    for history in (4, 8):
        guard = decision["clean_guardrails"][f"H{history}"]
        lines.append(f"| H{history} | {guard['bond_relative_change']:.6f} | {guard['contact_f1_change']:.6f} | {guard['amplitude_relative_change']:.6f} |")
    lines.extend(("", f"- True-prefix rollout bond relative change: {comparison['true_prefix_bond_relative_change']:.6f}", "", decision["reason"], "", f"Remaining risk: {decision['remaining_risk']}", ""))
    (ctx.output_dir / ROUND / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _decide(ctx: Any) -> dict[str, Any]:
    checkpoints = _checkpoints(ctx)
    quick = _evidence(ctx, "quick", checkpoints)
    rollout = _rollout_evidence(ctx, checkpoints)
    comparison = _rollout_comparison(rollout)
    threshold = float(ctx.cfg["decision"]["primary_relative_improvement"])
    tolerance = float(ctx.cfg["decision"]["opposite_trend_tolerance"])
    minimum = int(ctx.cfg["decision"]["minimum_systems_same_direction"])
    primary = comparison["relative_improvement"] >= threshold and comparison["systems_same_direction"] >= minimum
    no_opposite = comparison["relative_improvement"] > -tolerance
    final_root = ctx.output_dir / ROUND / "evaluation_final"
    if primary and not all((final_root / arm / "summary.json").is_file() for arm in ARMS):
        result = {
            "schema": f"{common.SCHEMA}.decision.v1", "round": 4, "status": "NEEDS_FINAL_EVALUATION",
            "parent_checkpoint": str(_round2_parent_path(ctx)[0]),
            "unique_variable": "clean versus online error-corrupted observed training history; clean future target",
            "rollout_comparison": comparison,
            "next_command": "evaluate-round4 --scope final, then decide-round4", "test_payload_opened": False,
        }
        common._write_json(ctx.output_dir / ROUND / "decision.json", result)
        return result
    evidence = _evidence(ctx, "final", checkpoints) if primary else quick
    scope = "final" if primary else "quick"
    guards = _clean_guardrails(evidence)
    guards_pass = _guards_pass(guards, ctx.cfg)
    true_prefix_confounded = comparison["true_prefix_bond_relative_change"] > float(ctx.cfg["decision"]["bond_relative_worsening"])
    train = json.loads((ctx.output_dir / ROUND / "train_summary.json").read_text(encoding="utf-8"))
    if primary and guards_pass and not true_prefix_confounded:
        status, selected, reason = "KEEP", "candidate", "candidate reduced generated-prefix extra bond degradation without clean single-segment or true-prefix regression"
    elif primary:
        status, selected, reason = "TRADEOFF", "control", "rollout primary improved but clean guardrails or true-prefix rollout geometry regressed"
    elif comparison["relative_improvement"] > 0.0 and not bool(train.get("common_extension_applied")):
        status, selected, reason = "NEEDS_COMMON_EXTENSION", None, "positive but sub-threshold/heterogeneous paired rollout evidence receives the one common extension"
    elif comparison["relative_improvement"] > 0.0:
        status, selected, reason = "INCONCLUSIVE", "control", "the one matched extension remained sub-threshold or heterogeneous; retain the simpler clean-history control"
    else:
        status, selected, reason = "REJECT", "control", "error-corrupted history training did not reduce paired generated-prefix extra bond degradation"
    parent_path, _, _ = _round2_parent_path(ctx)
    result = {
        "schema": f"{common.SCHEMA}.decision.v1", "round": 4, "status": status, "evaluation_scope": scope,
        "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": common.sha256_file(parent_path),
        "unique_variable": "same selected R2 parent and geometry auxiliary; only online observed-history coordinate corruption differs",
        "primary_metric": "E_roll_bond: mean max(0, bond_generated_prefix - bond_true_prefix) over rollout sections 2 and 3",
        "rollout_comparison": comparison, "clean_guardrails": guards, "clean_guardrails_pass": guards_pass,
        "true_prefix_rollout_confounded": true_prefix_confounded, "selected_arm": selected,
        "selected_checkpoint": None if selected is None else str(checkpoints[selected]),
        "selected_checkpoint_sha256": None if selected is None else common.sha256_file(checkpoints[selected]),
        "checkpoints": {arm: {"path": str(path), "sha256": common.sha256_file(path)} for arm, path in checkpoints.items()},
        "initialization_hash": train["initialization_hash"], "successful_updates_per_arm": train["added_successful_updates_per_arm"],
        "future_atom_frame_tokens_per_arm": train["added_future_atom_frame_tokens_per_arm"], "gpu_hours": train["gpu_hours"],
        "config_hash": train["config_hash"], "data_hash": train["data_hash"], "codec_state_hash": train["codec_state_hash"],
        "statistics_hash": train["statistics_hash"], "code": train["code"], "reason": reason,
        "remaining_risk": "one seed and a 32-frame diagnostic; this does not establish long-rollout stability or a scaling law",
        "next_command": "train-round4 --resume; then evaluate-round4 --scope quick and decide-round4" if status == "NEEDS_COMMON_EXTENSION" else None,
        "test_payload_opened": False,
    }
    common._write_json(ctx.output_dir / ROUND / "decision.json", result)
    if status not in ("NEEDS_COMMON_EXTENSION", "NEEDS_FINAL_EVALUATION"):
        _report(ctx, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_architecture_sequential_v1.yaml")
    parser.add_argument("--stage", required=True, choices=("profile-round4", "smoke-round4", "train-round4", "evaluate-round4", "decide-round4"))
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
    try:
        if not (output_dir / "selection.json").is_file():
            raise FileNotFoundError("round4 requires the frozen sequential selection")
        if args.stage == "profile-round4":
            result = _profile(ctx)
        elif args.stage == "smoke-round4":
            result = _smoke(ctx)
        elif args.stage == "train-round4":
            result = _train(ctx, resume=bool(args.resume))
        elif args.stage == "evaluate-round4":
            result = _evaluate(ctx, args.scope)
        else:
            result = _decide(ctx)
        print(json.dumps(common._safe(result), indent=2, sort_keys=True))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
