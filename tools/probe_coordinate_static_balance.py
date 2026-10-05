"""Train-only CUDA backward measurements on a fixed coordinate checkpoint."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.gradient_balance import gradient_snapshot, gradient_comparison, require_static_boundaries
from molvid.flow.coordinate import CoordinateFlowModel, coordinate_flow_loss
from molvid.latent.types import QuerySpec
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint


def corrupted_structure(clean, noise, sigma):
    """Only corrupted coordinates and their centroids become static context."""
    x = clean.x + sigma * noise
    _, group = torch.unique(clean.block_id, return_inverse=True)
    counts = torch.bincount(group).to(x)
    sums = x.new_zeros(1, counts.numel(), 3).index_add_(1, group, x)
    return replace(clean, x=x, bpos=(sums / counts[None, :, None])[:, group])


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA mismatch")
    candidates = load_ca_cache(args.cache, "train")
    by_system = {}
    for item in sorted(candidates, key=lambda i: i["record"]["sample_id"]):
        by_system.setdefault(item["metadata"]["system"], item)
    systems = sorted(by_system, key=lambda s: hashlib.sha256(("coordinate-static-balance-v1:" + s).encode()).hexdigest())[:16]
    if len(systems) != 16:
        raise ValueError("need 16 distinct training systems")
    chosen = [by_system[s] for s in systems]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require_final_coordinate_checkpoint(saved)
    runtime = Path(__file__).resolve().parents[1] / "molvid"
    for name, digest in saved["contract"]["runtime_source_sha256"].items():
        if sha256_file(runtime / name) != digest:
            raise ValueError(f"trained model runtime differs: {name}")
    device = configure_device("cuda", deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.manual_seed(20261005)
    parent, _, _ = load_parent(device)
    model = CoordinateFlowModel(parent).to(device).eval()
    del parent
    if model.contract() != saved["contract"]["model"]:
        raise ValueError("coordinate model contract differs")
    model.load_state_dict(saved["model"], strict=True)
    before = saved["model"]
    del saved["optimizer"], saved["rng"]
    protocol = dict(purpose="train-only static/dynamic gradient compatibility, not effectiveness",
        checkpoint_sha256=args.checkpoint_sha256, checkpoint_cursor=2048, systems=systems,
        dynamic_samples=[i["record"]["sample_id"] for i in chosen],
        static_pairing="independent training system at cyclic offset7; frame7",
        histories="alternating H4/H8 by selected system index", flow_times=[.25, .75],
        static_corruption_sigma_A=[.05, .25, .5], source_sigma_A=1.,
        expected_dynamic_backwards=32, expected_static_backwards=96,
        model_mode="eval, autograd enabled", precision="FP32", optimizer_updates=0,
        static_context="corrupted structure only; clean target restricted to RF loss/interpolation",
        budget="two-hour, GPU6 only", test_opened=False,
        limits=["single trained seed71 checkpoint", "16 train systems, not all192",
                "gradient cosine is local, not forecast improvement", "MD snapshots, not external static structures",
                "no selected loss weight or automatic task promotion"])
    freeze_run_identity(args.output, inputs=[args.checkpoint, args.cache / "complete.json", args.cache / "protocol.json"],
                        purpose=protocol["purpose"], protocol=protocol)
    atomic_write_json(args.output / "protocol.json", protocol)
    started = time.monotonic()
    rows = []
    with ExecutionEvidence(args.output):
        for index, item in enumerate(chosen):
            if time.monotonic() - started > 7200:
                raise TimeoutError("two-hour gradient probe budget")
            cpu = collate_clip_records([item["record"]])
            history = 4 if index % 2 == 0 else 8
            observed_cpu = slice_clip_frames(cpu, 0, history)
            model.prepare_batch(observed_cpu)
            observed = observed_cpu.to(device)
            target = cpu.x[history:].to(device)
            query = QuerySpec(cpu.time_ps[:, history:].to(device), cpu.frame_mask[:, history:].to(device))
            static_item = chosen[(index + 7) % len(chosen)]
            static_cpu = slice_clip_frames(collate_clip_records([static_item["record"]]), 7, 8)
            model.prepare_batch(static_cpu)
            clean = static_cpu.to(device)
            static_noise = torch.randn(clean.x.shape, device=device,
                generator=torch.Generator(device=device).manual_seed(3000 + index))
            for flow_time in (.25, .75):
                model.zero_grad(set_to_none=True)
                progress = target.new_full((1,), flow_time)
                condition = model.prepare_condition(observed)
                source = model.source(condition=condition, query=query,
                    generator=torch.Generator(device=device).manual_seed(2000 + index))
                prediction = model(torch.lerp(source, target, progress[:, None, None]), condition=condition,
                    query=query, flow_time=progress, checkpoint_blocks=True)
                loss = coordinate_flow_loss(prediction, target - source, batch=observed, query=query)
                loss.backward()
                dynamic_loss = float(loss.detach())
                dynamic = gradient_snapshot(model)
                if any(n.startswith("target_teacher.") for n in dynamic):
                    raise RuntimeError("frozen teacher got dynamic gradients")
                del condition, source, prediction, loss
                for sigma in (.05, .25, .5):
                    model.zero_grad(set_to_none=True)
                    corrupted = corrupted_structure(clean, static_noise, sigma)
                    static_condition = model.prepare_condition(corrupted, task="static")
                    state = torch.lerp(corrupted.x, clean.x, progress[:, None, None])
                    prediction = model.forward_static(state, condition=static_condition, flow_time=progress)
                    static_query = QuerySpec(torch.zeros_like(clean.time_ps), clean.frame_mask)
                    loss = coordinate_flow_loss(prediction, clean.x - corrupted.x, batch=clean, query=static_query)
                    loss.backward()
                    static = gradient_snapshot(model)
                    require_static_boundaries(static)
                    rows.append(dict(dynamic_system=systems[index], static_system=systems[(index + 7) % 16],
                        history_frames=history, flow_time=flow_time, sigma_A=sigma,
                        dynamic_loss_A2=dynamic_loss, static_loss_A2=float(loss.detach()),
                        groups=gradient_comparison(dynamic, static)))
                    del corrupted, static_condition, state, prediction, loss, static
                del dynamic
            atomic_write_json(args.output / "rows.json", rows)
            print(json.dumps(dict(completed_systems=index + 1, paired_rows=len(rows), seconds=time.monotonic()-started)), flush=True)
            del observed, target, query, clean, static_noise
        model.zero_grad(set_to_none=True)
        if any(not torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items()):
            raise RuntimeError("read-only gradient probe changed model state")
        aggregates = {}
        for sigma in (.05, .25, .5):
            subset = [r for r in rows if r["sigma_A"] == sigma]
            aggregates[str(sigma)] = {}
            for group in ("dit", "readout"):
                values = [r["groups"][group] for r in subset]
                aggregates[str(sigma)][group] = dict(
                    mean_cosine=float(np.mean([v["cosine"] for v in values])),
                    negative_cosines=sum(v["cosine"] < 0 for v in values), pairs=len(values),
                    median_norm_ratio=float(np.median([v["static_to_dynamic_norm"] for v in values])),
                    max_norm_ratio=max(v["static_to_dynamic_norm"] for v in values),
                    mean_static_active_cosine=float(np.mean([v["static_active_cosine"] for v in values])))
        atomic_write_json(args.output / "summary.json", dict(status="completed", optimizer_updates=0,
            dynamic_backwards=32, static_backwards=96, systems=16, static_boundaries_passed=True,
            model_state_bitwise_unchanged=True, seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), aggregates=aggregates,
            effectiveness_tested=False, test_opened=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("/workspace/molvid-ca-detail-20261005/runs/ca_detail_20261005/cache"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
