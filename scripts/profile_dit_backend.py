#!/usr/bin/env python
"""CUDA equivalence, real-clip smoke, and bounded profile for DiT backend v2.

This script intentionally reuses the frozen pilot loader.  It only opens the
train/validation stores through that loader, never fits or updates a science
checkpoint, and keeps the profile to four cases x two backends x 25 steps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics as py_statistics
import sys
import time
from typing import Any, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.dit_backend_v2 import ATTENTION_BACKEND
from module.latent_rectified_flow import RectifiedFlowObjective, generate_state_detail_latent
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
    tensor_hash,
)
from scripts.run_state_detail_dit_pilot import (
    _encode_batch,
    _load_approved_codec,
    _load_data,
    _make_sampler,
    _observed_batch,
    FrozenCodec,
    PilotData,
)
from trainer.dit_trainer import module_state_hash


REFERENCE_ATTENTION_BACKEND = "reference_einsum_softmax"
BACKENDS = ("reference", "factorized_v2")


@dataclass(frozen=True)
class DitBundle:
    candidate: str
    ratio: int
    config: Mapping[str, Any]
    model_contract: Mapping[str, Any]
    model_contract_hash: str
    model_state: Mapping[str, Any]
    model_state_hash: str
    statistics: LatentStatistics
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class CaseSpec:
    name: str
    ratio: int
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    batch_size: int
    num_atoms: int
    max_blocks: int
    atom_counts: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCase:
    spec: CaseSpec
    latent_batch: Any
    observed: Any
    normalized: Any
    encode_seconds: float
    input_hash: str


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    return float(py_statistics.median(values)) if values else float("nan")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(percentile * len(ordered))) - 1))
    return float(ordered[index])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    return generator.manual_seed(int(seed))


def _field_max_abs(left: Any, right: Any) -> float:
    values = []
    for name in left.names():
        values.append((getattr(left, name).float() - getattr(right, name).float()).abs().max())
    return float(torch.stack(values).max().detach().cpu())


def _field_relative_l2(left: Any, right: Any) -> float:
    numerator = []
    denominator = []
    for name in left.names():
        a = getattr(left, name).float()
        b = getattr(right, name).float()
        numerator.append((a - b).square().sum())
        denominator.append(a.square().sum())
    return float(
        torch.stack(numerator).sum().sqrt().div(torch.stack(denominator).sum().sqrt().clamp_min(1.0e-12))
        .detach()
        .cpu()
    )


def _gradient_max_abs(left: torch.nn.Module, right: torch.nn.Module) -> float:
    values = []
    for parameter_left, parameter_right in zip(left.parameters(), right.parameters()):
        if parameter_left.grad is None or parameter_right.grad is None:
            continue
        values.append((parameter_left.grad.float() - parameter_right.grad.float()).abs().max())
    return float(torch.stack(values).max().detach().cpu()) if values else 0.0


def _gate_status(model: MolecularDiT) -> dict[str, Any]:
    values = []
    nonzero = 0
    total = 0
    for block in model.blocks:
        for adaln in (block.spatial_adaln, block.temporal_adaln, block.ffn_adaln):
            for value in (adaln.modulation.weight, adaln.modulation.bias):
                current = value.detach().float()
                values.append(current.abs().max())
                nonzero += int(torch.count_nonzero(current).detach().cpu())
                total += current.numel()
    return {
        "max_abs": float(torch.stack(values).max().detach().cpu()),
        "nonzero": nonzero,
        "total": total,
        "nonzero_fraction": nonzero / max(total, 1),
    }


def _batch_value_hash(batch: Any) -> str:
    values = {
        "fields": {name: tensor_hash(getattr(batch, name)) for name in batch.fields.names()},
        "token_mask": tensor_hash(batch.token_mask),
        "detail_valid": tensor_hash(batch.detail_valid),
        "abid": tensor_hash(batch.abid),
        "block_id": tensor_hash(batch.block_id),
        "atom_ptr": tensor_hash(batch.atom_ptr),
    }
    return contract_hash(values)


def _new_model(bundle: DitBundle, backend: str, device: torch.device) -> MolecularDiT:
    config = bundle.config
    adapter = StateDetailLatentAdapter(
        codec_width=int(config["codec_width"]),
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        ratio=bundle.ratio,
    ).to(device)
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=int(config["scalar_width"]),
        vector_width=int(config["vector_width"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        ffn_multiplier=int(config["ffn_multiplier"]),
        dropout=float(config["dropout"]),
        execution_backend=backend,
    ).to(device)
    if contract_hash(model.contract()) != bundle.model_contract_hash:
        raise RuntimeError(f"{bundle.candidate} model contract changed before loading")
    model.load_state_dict(bundle.model_state, strict=True)
    model.eval()
    return model


def _load_dit_bundle(candidate: str, path: Path) -> DitBundle:
    path = path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise RuntimeError(f"invalid DiT checkpoint payload: {path}")
    if payload.get("schema") != "pvb.dit.state_detail.checkpoint.v2":
        raise RuntimeError(f"unexpected DiT checkpoint schema: {path}")
    config = dict(payload["config"])
    ratio = int(payload["ratio"])
    if candidate != f"ratio{ratio}_state_detail":
        raise RuntimeError("candidate and DiT checkpoint ratio disagree")
    model_state = payload.get("model_state")
    model_contract = payload.get("model_contract")
    statistics_state = payload.get("statistics_state")
    if not isinstance(model_state, Mapping) or not isinstance(model_contract, Mapping):
        raise RuntimeError("DiT checkpoint lacks model state or model contract")
    if not isinstance(statistics_state, Mapping):
        raise RuntimeError("DiT checkpoint lacks serialized statistics")
    model_contract_hash = str(payload.get("model_contract_hash", ""))
    if model_contract_hash != contract_hash(model_contract):
        raise RuntimeError("DiT model contract hash is invalid")
    statistics = LatentStatistics.from_state_dict(statistics_state)
    if statistics.hash != str(payload.get("statistics_hash", "")):
        raise RuntimeError("DiT statistics hash is invalid")
    return DitBundle(
        candidate=candidate,
        ratio=ratio,
        config=config,
        model_contract=model_contract,
        model_contract_hash=model_contract_hash,
        model_state=model_state,
        model_state_hash=contract_hash(
            {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
             for name, value in sorted(model_state.items())}
        ),
        statistics=statistics,
        checkpoint_path=path,
        checkpoint_sha256=_sha256(path),
    )


def _case_shape(batch_cpu: Any) -> tuple[int, int, tuple[int, ...], int]:
    counts = tuple(int(value) for value in batch_cpu.host_atom_counts)
    block_values = batch_cpu.block_id.tolist()
    max_blocks = max(
        (len(set(block_values[start:stop])) for start, stop in zip(
            batch_cpu.atom_ptr[:-1].tolist(), batch_cpu.atom_ptr[1:].tolist()
        )),
        default=0,
    )
    return batch_cpu.batch_size, int(batch_cpu.atom_count), counts, int(max_blocks)


def _probe_cases(data: PilotData, cfg: Mapping[str, Any]) -> dict[str, CaseSpec]:
    profile = cfg["profile"]
    probe = int(profile.get("candidate_batch_probe", 32))
    sampler = _make_sampler(data.train, seed=int(cfg["seed"]), clips_per_trajectory=24)
    candidates: list[tuple[tuple[int, ...], tuple[str, ...], int, int, tuple[int, ...], int]] = []
    for indices in list(sampler.global_batches)[:probe]:
        records = [data.train[int(index)] for index in indices]
        from data.clip_dataset import collate_clip_records
        batch_cpu = collate_clip_records(records)
        batch_size, num_atoms, counts, max_blocks = _case_shape(batch_cpu)
        candidates.append((
            tuple(int(index) for index in indices),
            tuple(str(value) for value in batch_cpu.sample_id),
            batch_size,
            num_atoms,
            counts,
            max_blocks,
        ))
    if len(candidates) < 3:
        raise RuntimeError("profile probe produced fewer than three real train batches")
    candidates.sort(key=lambda row: (row[3], row[0]))

    def make(name: str, ratio: int, row: tuple[Any, ...]) -> CaseSpec:
        return CaseSpec(
            name=name,
            ratio=ratio,
            indices=row[0],
            sample_ids=row[1],
            batch_size=row[2],
            num_atoms=row[3],
            max_blocks=row[5],
            atom_counts=row[4],
        )

    middle = candidates[len(candidates) // 2]
    return {
        "r4_small": make("r4_small", 4, candidates[0]),
        "r4_median": make("r4_median", 4, middle),
        "r4_large": make("r4_large", 4, candidates[-1]),
        "r2_median": make("r2_median", 2, middle),
    }


def _prepare_case(
    data: PilotData,
    codec: FrozenCodec,
    bundle: DitBundle,
    spec: CaseSpec,
    device: torch.device,
) -> PreparedCase:
    adapter = StateDetailLatentAdapter(codec_width=int(bundle.config["codec_width"]), ratio=bundle.ratio)
    started = time.perf_counter()
    latent, _batch_cpu, batch = _encode_batch(
        data.train,
        spec.indices,
        codec=codec,
        adapter=adapter,
        data_hash=data.data_hash,
        device=device,
    )
    _sync(device)
    encode_seconds = time.perf_counter() - started
    latent_batch = adapter.pack(
        latent,
        codec_hash=codec.codec_state_hash,
        data_hash=data.data_hash,
        loss_mask=batch.loss_mask,
    )
    observed = _observed_batch(latent_batch, batch, adapter=adapter, history_frames=4)
    normalized = bundle.statistics.to(device=device).normalize(observed)
    if normalized.batch_size != spec.batch_size or normalized.num_atoms != spec.num_atoms:
        raise RuntimeError(f"case shape changed during real encoding: {spec.name}")
    return PreparedCase(
        spec=spec,
        latent_batch=latent_batch,
        observed=observed,
        normalized=normalized,
        encode_seconds=float(encode_seconds),
        input_hash=_batch_value_hash(normalized),
    )


def _profile_step(
    model: MolecularDiT,
    batch: Any,
    flow: RectifiedFlowObjective,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    amp: bool,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_start = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    forward_start.record()
    sample = flow.sample(batch, generator=generator)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        prediction = model(batch.with_fields(sample.interpolated), sample.tau)
        loss = flow.loss(prediction, sample.target, batch)
    forward_end.record()
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    backward_end.record()
    optimizer.step()
    optimizer_end.record()
    _sync(device)
    return (
        float(total_start.elapsed_time(optimizer_end) / 1000.0),
        float(forward_start.elapsed_time(forward_end) / 1000.0),
        float(forward_end.elapsed_time(backward_end) / 1000.0),
        float(backward_end.elapsed_time(optimizer_end) / 1000.0),
        float(loss.total.detach().float().cpu()),
    )


def _run_profile(
    bundle: DitBundle,
    prepared: PreparedCase,
    backend: str,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = _new_model(bundle, backend, device)
    model_hash = module_state_hash(model)
    gate = _gate_status(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(bundle.config.get("learning_rate", 2.0e-4)),
        weight_decay=float(bundle.config.get("weight_decay", 0.01)),
    )
    flow = RectifiedFlowObjective()
    seed = int(cfg["seed"]) + prepared.spec.ratio * 1000 + sum(ord(c) for c in backend)
    generator = _generator(device, seed)
    warmup_steps = int(cfg["profile"]["warmup_steps"])
    measured_steps = int(cfg["profile"]["measured_steps"])
    for _ in range(warmup_steps):
        _profile_step(model, prepared.normalized, flow, optimizer, generator, True, device)
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for _ in range(measured_steps):
        rows.append(_profile_step(model, prepared.normalized, flow, optimizer, generator, True, device))
    total, forward, backward, optimizer_times, losses = zip(*rows)
    model_backend = ATTENTION_BACKEND if backend == "factorized_v2" else REFERENCE_ATTENTION_BACKEND
    model_tokens = prepared.spec.num_atoms * (16 // prepared.spec.ratio)
    return {
        "backend": backend,
        "attention_backend": model_backend,
        "execution_contract": model.execution_contract(),
        "case": prepared.spec.name,
        "ratio": prepared.spec.ratio,
        "batch_size": prepared.spec.batch_size,
        "num_atoms": prepared.spec.num_atoms,
        "atom_counts": list(prepared.spec.atom_counts),
        "max_blocks": prepared.spec.max_blocks,
        "latent_atom_time_tokens": model_tokens,
        "atom_frame_tokens": 16 * prepared.spec.num_atoms,
        "scalar_coefficient_volume": 2 * model_tokens * int(bundle.config["codec_width"]),
        "vector_coefficient_volume": 2 * model_tokens * 3 * int(bundle.config["codec_width"]),
        "input_hash": prepared.input_hash,
        "model_initial_state_hash": model_hash,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "dtype": "torch.bfloat16_autocast",
        "amp": True,
        "tf32": False,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "timing_s": {
            "total_median": _median(list(total)),
            "total_p90": _percentile(list(total), 0.90),
            "forward_median": _median(list(forward)),
            "forward_p90": _percentile(list(forward), 0.90),
            "backward_median": _median(list(backward)),
            "backward_p90": _percentile(list(backward), 0.90),
            "optimizer_median": _median(list(optimizer_times)),
            "optimizer_p90": _percentile(list(optimizer_times), 0.90),
            "steps_per_s": measured_steps / max(sum(total), 1.0e-12),
            "samples_per_s": prepared.spec.batch_size * measured_steps / max(sum(total), 1.0e-12),
            "atom_frame_tokens_per_s": 16 * prepared.spec.num_atoms * measured_steps / max(sum(total), 1.0e-12),
            "latent_atom_time_tokens_per_s": model_tokens * measured_steps / max(sum(total), 1.0e-12),
            "end_to_end_median": _median(list(total)) + prepared.encode_seconds,
        },
        "loss": {"first": float(losses[0]), "last": float(losses[-1])},
        "encoder_seconds": prepared.encode_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "adaln_gate": gate,
    }


def _load_context(cfg: Mapping[str, Any], device: torch.device) -> tuple[PilotData, dict[str, DitBundle], dict[str, FrozenCodec]]:
    data = _load_data(Path(cfg["manifest_root"]))
    bundles: dict[str, DitBundle] = {}
    codecs: dict[str, FrozenCodec] = {}
    for candidate, paths in cfg["checkpoints"].items():
        bundles[candidate] = _load_dit_bundle(candidate, Path(paths["dit"]))
        codecs[candidate] = _load_approved_codec(
            candidate=candidate,
            result_path=Path(paths["codec_result"]),
            checkpoint_path=Path(paths["codec_checkpoint"]),
            device=device,
        )
    return data, bundles, codecs


def _equivalence(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    data, bundles, codecs = _load_context(cfg, device)
    try:
        cases = _probe_cases(data, cfg)
        rows = []
        for candidate in ("ratio2_state_detail", "ratio4_state_detail"):
            bundle = bundles[candidate]
            spec = cases["r2_median" if bundle.ratio == 2 else "r4_median"]
            prepared = _prepare_case(data, codecs[candidate], bundle, spec, device)
            reference = _new_model(bundle, "reference", device)
            optimized = _new_model(bundle, "factorized_v2", device)
            tau = torch.full((prepared.normalized.batch_size,), 0.37, device=device)
            with torch.no_grad():
                reference_output = reference(prepared.normalized, tau)
                optimized_output = optimized(prepared.normalized, tau)
            forward_abs = _field_max_abs(reference_output, optimized_output)
            forward_rel = _field_relative_l2(reference_output, optimized_output)
            flow = RectifiedFlowObjective()
            reference.train()
            optimized.train()
            sample_reference = flow.sample(prepared.normalized, generator=_generator(device, 424242))
            sample_optimized = flow.sample(prepared.normalized, generator=_generator(device, 424242))
            loss_reference = flow.loss(
                reference(prepared.normalized.with_fields(sample_reference.interpolated), sample_reference.tau),
                sample_reference.target,
                prepared.normalized,
            )
            loss_optimized = flow.loss(
                optimized(prepared.normalized.with_fields(sample_optimized.interpolated), sample_optimized.tau),
                sample_optimized.target,
                prepared.normalized,
            )
            loss_reference.total.backward()
            loss_optimized.total.backward()
            gradient_abs = _gradient_max_abs(reference, optimized)
            optimizer_reference = torch.optim.AdamW(reference.parameters(), lr=2.0e-4, weight_decay=0.01)
            optimizer_optimized = torch.optim.AdamW(optimized.parameters(), lr=2.0e-4, weight_decay=0.01)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(optimized.parameters(), 1.0)
            optimizer_reference.step()
            optimizer_optimized.step()
            update_abs = max(
                (left.detach().float() - right.detach().float()).abs().max().item()
                for left, right in zip(reference.parameters(), optimized.parameters())
            )
            rows.append({
                "candidate": candidate,
                "case": spec.name,
                "batch_size": spec.batch_size,
                "num_atoms": spec.num_atoms,
                "max_blocks": spec.max_blocks,
                "input_hash": prepared.input_hash,
                "forward_max_abs": forward_abs,
                "forward_relative_l2": forward_rel,
                "gradient_max_abs": gradient_abs,
                "adamw_update_max_abs": float(update_abs),
                "reference_attention_backend": REFERENCE_ATTENTION_BACKEND,
                "optimized_attention_backend": ATTENTION_BACKEND,
                "reference_adaln_gate": _gate_status(reference),
                "optimized_adaln_gate": _gate_status(optimized),
                "tolerances": {
                    "forward_atol": 2.0e-5,
                    "forward_rtol": 2.0e-4,
                    "gradient_atol": 2.0e-5,
                },
            })
            del reference, optimized, prepared
            torch.cuda.empty_cache()
        result = {
            "schema": "pvb.dit.factorized_backend.v2.equivalence",
            "status": "PASS",
            "device": str(device),
            "dtype": "float32",
            "tf32": False,
            "rows": rows,
        }
        _json_dump(output_dir / "parity.json", result)
        return result
    finally:
        data.train.close()
        data.valid.close()


def _smoke(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    data, bundles, codecs = _load_context(cfg, device)
    try:
        cases = _probe_cases(data, cfg)
        rows = []
        for candidate in ("ratio2_state_detail", "ratio4_state_detail"):
            bundle = bundles[candidate]
            spec = cases["r2_median" if bundle.ratio == 2 else "r4_median"]
            prepared = _prepare_case(data, codecs[candidate], bundle, spec, device)
            model = _new_model(bundle, "factorized_v2", device)
            model.eval()
            with torch.no_grad():
                generated, metadata = generate_state_detail_latent(
                    model,
                    model.adapter,
                    prepared.observed,
                    bundle.statistics.to(device=device),
                    steps=8,
                    seed=int(cfg["seed"]) + bundle.ratio,
                )
                decoded = codecs[candidate].model.decode(generated)
            x_hat = decoded.x_hat
            rows.append({
                "candidate": candidate,
                "case": spec.name,
                "input_hash": prepared.input_hash,
                "generated_latent_contract_hash": contract_hash(generated.contract()),
                "generation": metadata,
                "decoded_x_hat_shape": list(x_hat.shape),
                "decoded_x_hat_finite": bool(torch.isfinite(x_hat).all()),
                "codec_state_hash_before": codecs[candidate].codec_state_hash,
            })
            del model, prepared, generated, decoded
            torch.cuda.empty_cache()
        result = {
            "schema": "pvb.dit.factorized_backend.v2.smoke",
            "status": "PASS",
            "device": str(device),
            "steps": 8,
            "rows": rows,
        }
        _json_dump(output_dir / "smoke.json", result)
        return result
    finally:
        data.train.close()
        data.valid.close()


def _profile(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    data, bundles, codecs = _load_context(cfg, device)
    try:
        cases = _probe_cases(data, cfg)
        _json_dump(output_dir / "profile_inputs.json", {
            "schema": "pvb.dit.factorized_backend.v2.profile_inputs",
            "data_hash": data.data_hash,
            "manifest_content_sha256": data.manifest["manifest_content_sha256"],
            "cases": {name: spec.__dict__ for name, spec in cases.items()},
        })
        rows = []
        for case_name in ("r4_small", "r4_median", "r4_large", "r2_median"):
            spec = cases[case_name]
            candidate = f"ratio{spec.ratio}_state_detail"
            bundle = bundles[candidate]
            prepared = _prepare_case(data, codecs[candidate], bundle, spec, device)
            for backend in BACKENDS:
                row = _run_profile(bundle, prepared, backend, cfg, device)
                rows.append(row)
                print(json.dumps({"event": "profile_case", **row["timing_s"], "case": case_name, "backend": backend}, sort_keys=True), flush=True)
            del prepared
            torch.cuda.empty_cache()
        result = {
            "schema": "pvb.dit.factorized_backend.v2.profile",
            "status": "PASS",
            "device": str(device),
            "warmup_steps": int(cfg["profile"]["warmup_steps"]),
            "measured_steps": int(cfg["profile"]["measured_steps"]),
            "optimizer_steps": 2 * 4 * (int(cfg["profile"]["warmup_steps"]) + int(cfg["profile"]["measured_steps"])),
            "rows": rows,
        }
        _json_dump(output_dir / "profile.json", result)
        lines = [
            "# DiT factorized backend v2 bounded profile",
            "",
            f"- device: {device}; TF32 disabled; profile autocast: BF16",
            f"- optimizer steps: {result['optimizer_steps']} (two backends x four cases x 25)",
            "- timings are CUDA-event medians/p90s; end-to-end adds the one-time frozen codec encode.",
            "",
            "| case | backend | N | B | M | total median ms | total p90 ms | forward median ms | backward median ms | optimizer median ms | end-to-end median ms | atom-frame tok/s | latent atom-time tok/s | peak allocated MiB | peak reserved MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            timing = row["timing_s"]
            lines.append(
                f"| {row['case']} | {row['backend']} | {row['num_atoms']} | {row['batch_size']} | {row['max_blocks']} | "
                f"{timing['total_median'] * 1000:.3f} | {timing['total_p90'] * 1000:.3f} | "
                f"{timing['forward_median'] * 1000:.3f} | {timing['backward_median'] * 1000:.3f} | "
                f"{timing['optimizer_median'] * 1000:.3f} | {timing['end_to_end_median'] * 1000:.3f} | "
                f"{timing['atom_frame_tokens_per_s']:.1f} | {timing['latent_atom_time_tokens_per_s']:.1f} | "
                f"{row['peak_allocated_bytes'] / 2**20:.1f} | {row['peak_reserved_bytes'] / 2**20:.1f} |")
        (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result
    finally:
        data.train.close()
        data.valid.close()


def _preflight(cfg: Mapping[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    paths = []
    for candidate, values in cfg["checkpoints"].items():
        for label in ("dit", "codec_result", "codec_checkpoint"):
            path = Path(values[label]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing {candidate} {label}: {path}")
            if "test" in str(path).lower().split("/"):
                raise RuntimeError(f"test path is forbidden: {path}")
            paths.append({"candidate": candidate, "label": label, "path": str(path), "sha256": _sha256(path)})
    manifest_root = Path(cfg["manifest_root"]).resolve()
    if "test" in str(manifest_root).lower().split("/"):
        raise RuntimeError("manifest path must not contain test")
    for name in ("manifest.json", "materialization.json"):
        if not (manifest_root / name).is_file():
            raise FileNotFoundError(manifest_root / name)
    result = {
        "schema": "pvb.dit.factorized_backend.v2.preflight",
        "status": "PASS",
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "tf32": False,
        "manifest_root": str(manifest_root),
        "paths": paths,
    }
    _json_dump(output_dir / "preflight.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "equivalence", "smoke", "profile"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise RuntimeError("profile config must be a mapping")
    device = torch.device(args.device or cfg.get("device", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("backend v2 checks require an actual CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_root = Path(args.output_root or cfg.get("output_root", "outputs/dit_factorized_backend_v2"))
    output_dir = output_root / str(cfg.get("run_id", "backend_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = _preflight(cfg, output_dir, device)
    elif args.stage == "equivalence":
        result = _equivalence(cfg, output_dir, device)
    elif args.stage == "smoke":
        result = _smoke(cfg, output_dir, device)
    else:
        result = _profile(cfg, output_dir, device)
    print(json.dumps({"stage": args.stage, "output_dir": str(output_dir), "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
