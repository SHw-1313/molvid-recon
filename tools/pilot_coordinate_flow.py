"""Bounded real-ATLAS full-model CUDA gate, not a quality benchmark."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate import CoordinateFlowModel, coordinate_flow_loss
from molvid.latent.types import QuerySpec
from molvid.provenance import freeze_run_identity
from molvid.runtime import configure_device, atomic_write_json, sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, atomic_torch_save, RELEASE


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    device = configure_device("cuda", deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.06)
    torch.manual_seed(901)
    candidates = load_ca_cache(args.cache, "train")
    item = min(candidates, key=lambda i: (len(i["record"]["atype"]), i["record"]["sample_id"]))
    record = item["record"]
    parent, _, release = load_parent(device)
    model = CoordinateFlowModel(parent).train()
    del parent
    cpu = collate_clip_records([record])
    observed_cpu = slice_clip_frames(cpu, 0, 4)
    model.prepare_batch(observed_cpu)
    observed = observed_cpu.to(device)
    # Only two query frames for a memory-bounded engineering gate.
    target = cpu.x[4:6].to(device)
    query = QuerySpec(cpu.time_ps[:, 4:6].to(device), cpu.frame_mask[:, 4:6].to(device))
    generator = torch.Generator(device=device).manual_seed(902)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    protocol = dict(purpose="real_coordinate_flow_engineering_gate_not_quality",
                    system=item["metadata"]["system"], sample_id=record["sample_id"],
                    split="train", atoms=observed.atom_count, history=4, query_frames=2,
                    planned_updates=4, static_backward_only=True, actual_source_steps=2,
                    memory_fraction=.06, co_resident_with_own_capacity_evaluation=True,
                    throughput_comparison_valid=False, parent=release["selected_checkpoint"],
                    contract=model.contract(), warm_start=model.warm_start_report,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    freeze_run_identity(args.output, inputs=[args.cache / "complete.json", RELEASE,
                                             Path(release["selected_checkpoint"]["path"])],
                        purpose=protocol["purpose"], protocol=protocol)
    atomic_write_json(args.output / "protocol.json", protocol)
    started = time.monotonic()
    rows = []
    with ExecutionEvidence(args.output):
        for step in range(1, 5):
            optimizer.zero_grad(set_to_none=True)
            condition = model.prepare_condition(observed)
            source = model.source(condition=condition, query=query, generator=generator)
            progress = torch.rand(1, device=device, generator=generator)
            state = torch.lerp(source, target, progress[:, None, None])
            predicted = model(state, condition=condition, query=query, flow_time=progress)
            loss = coordinate_flow_loss(predicted, target - source, batch=observed, query=query)
            loss.backward()
            grads = {name: sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
                     for name, module in (("dit", model.dit), ("history", model.history_encoder), ("readout", model.readout))}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite coordinate-flow gradients")
            optimizer.step()
            rows.append(dict(step=step, loss=float(loss), gradients=grads, grad_norm=float(norm),
                             seconds=time.monotonic() - started))
            atomic_write_json(args.output / "updates.json", rows)
            print(json.dumps(rows[-1]), flush=True)
        if not all(rows[-1]["gradients"][name] > 0 for name in ("dit", "history", "readout")):
            raise RuntimeError("real coordinate-flow path did not train all intended modules")
        del condition, source, state, predicted, loss
        optimizer.zero_grad(set_to_none=True)
        # This gate is genuinely differentiable through current-coordinate
        # teacher graph construction. It makes NO additional optimizer update.
        condition = model.prepare_condition(observed)
        source = model.source(condition=condition, query=query, generator=generator).requires_grad_(True)
        endpoint = model.rollout(source, condition=condition, query=query, steps=2)
        endpoint.square().mean().backward()
        if source.grad is None or not torch.isfinite(source.grad).all():
            raise RuntimeError("actual-source coordinate gradient failed")
        rollout_grad = float(source.grad.norm())
        del condition, source, endpoint
        optimizer.zero_grad(set_to_none=True)
        clean_static = slice_clip_frames(observed, 0, 1)
        corrupted = replace(clean_static, x=clean_static.x + .2 * torch.randn(
            clean_static.x.shape, generator=generator, device=device))
        static = model.prepare_condition(corrupted, task="static")
        value = model.forward_static(corrupted.x, condition=static, flow_time=torch.tensor([.4], device=device))
        (value - (clean_static.x - corrupted.x)).square().mean().backward()
        if any(p.grad is not None for p in model.history_encoder.parameters()):
            raise RuntimeError("static loss entered history")
        static_grad = sum(float(p.grad.norm()) for p in model.dit.parameters() if p.grad is not None)
        if static_grad <= 0 or any(p.grad is not None for p in model.target_teacher.parameters()):
            raise RuntimeError("static trunk/frozen encoder gradient boundary failed")
        optimizer.zero_grad(set_to_none=True)
        del static, value
        checkpoint = args.output / "coordinate_step_0004.pt"
        atomic_torch_save(checkpoint, dict(schema="molvid.coordinate_flow.prototype.checkpoint.v1",
                                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         contract=model.contract(), parent=release["selected_checkpoint"],
                                         completed_updates=4, warm_start=model.warm_start_report))
        model.eval()
        with torch.no_grad():
            condition = model.prepare_condition(observed)
            fixed_source = observed.x[-1:].expand(2, -1, -1).clone()
            reference = model.rollout(fixed_source, condition=condition, query=query, steps=2)
        del condition, optimizer, model
        torch.cuda.empty_cache()
        fresh_parent, _, _ = load_parent(device)
        fresh = CoordinateFlowModel(fresh_parent).eval()
        del fresh_parent
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["contract"] != fresh.contract():
            raise RuntimeError("checkpoint contract differs")
        fresh.load_state_dict(saved["model"], strict=True)
        fresh.prepare_batch(observed_cpu)
        with torch.no_grad():
            recovered = fresh.rollout(fixed_source, condition=fresh.prepare_condition(observed), query=query, steps=2)
        error = float((reference - recovered).abs().max())
        if not torch.equal(reference, recovered):
            raise RuntimeError(f"fresh checkpoint forecast differs: {error}")
        atomic_write_json(args.output / "summary.json", dict(status="completed", updates=4,
            systems=1, split="train", static_updates=0, rollout_backward_updates=0,
            actual_source_input_grad_norm=rollout_grad, static_trunk_grad_norm=static_grad,
            checkpoint_sha256=sha256_file(checkpoint), reload_max_error_angstrom=error,
            optimizer_resume_tested=False, geometry_motion_effectiveness_tested=False,
            seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            test_opened=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/workspace/molvid-ca-detail-20261005/runs/ca_detail_20261005/cache"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
