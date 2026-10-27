"""Two real TRAIN Q1 checks only: largest H4 RF and small irregular-history aux.

Run through run_aux.py kernel_data_max; no optimizer, training loop, or extra
sampling matrix. Frozen model/data/trainer modules are only imported.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.kernel_windows import KernelWindows
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.flow.local_geometry_loss import ligand_geometry_band_loss
from molvid.training.coordinate_state import LigandSourceGeometry, load_coordinate_weights

RUN = ROOT / "runs/flow_kernel_20261027"
PARENT = Path("/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt")


def memory_guard(physical_gpu, initialized=False):
    used_mib = float(subprocess.check_output([
        "nvidia-smi", "-i", str(physical_gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    own_mib = torch.cuda.memory_reserved() / 2**20 if initialized else 0.
    external_mib = used_mib - own_mib
    # Stop before the parent's approximate 48 GiB external-occupancy boundary.
    if external_mib >= 47 * 1024:
        raise RuntimeError(f"External GPU allocation approaches 48GiB: {external_mib:.1f}MiB")
    return dict(total_used_MiB=used_mib, own_reserved_MiB=own_mib,
                estimated_external_MiB=external_mib, stop_at_external_MiB=47 * 1024)


def gradient_summary(model):
    squares, head_squares, count = 0., 0., 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not bool(torch.isfinite(grad).all()):
            raise ValueError(f"nonfinite gradient: {name}")
        value = float(grad.square().sum())
        squares += value
        if name.startswith("head."):
            head_squares += value
        count += 1
    if squares <= 0 or head_squares <= 0:
        raise ValueError("Expected nonzero finite model and readout gradients")
    return dict(model_gradient_norm=squares**.5, head_gradient_norm=head_squares**.5,
                tensors_with_gradient=count, all_gradients_finite=True)


def peak():
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--cap-GiB", type=float, default=32.)
    parser.add_argument("--output", type=Path, default=RUN / "kernel_data_max.json")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "torch-ito":
        raise RuntimeError("existing torch-ito environment required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("one explicitly selected physical GPU is required")
    if args.output.exists():
        raise FileExistsError("preserve the existing check receipt")
    started = time.monotonic()
    result = dict(status="running", parent=str(PARENT), physical_gpu=args.physical_gpu,
                  cap_GiB=args.cap_GiB, optimizer_steps=0, sampling_trajectories=0)
    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        torch.set_num_threads(2)
        result["initial_gpu_memory"] = memory_guard(args.physical_gpu)
        cfg = json.loads((ROOT / "tasks/flow-kernel-20261027/configs/history_mix.json").read_text())
        manifest = json.loads(Path(cfg["kernel_training"]["manifest"]).read_text())
        rows = manifest["rows"]
        largest = max((r for r in rows if len(r["history_ages_delta"]) == 4), key=lambda r: r["atom_count"])
        irregular = [r for r in rows if r["domain"] == "complex" and len(r["history_ages_delta"]) == 4
                     and len({a - b for a, b in zip(r["history_ages_delta"], r["history_ages_delta"][1:])}) > 1]
        small = min(irregular, key=lambda r: r["atom_count"])
        data = KernelWindows(cfg["kernel_training"], cfg["data"], 71)
        def load(row, mode="history_mix"):
            data.load_state_dict(dict(cursor=row["plan_index"], plan_digest=data.digest))
            return data.next(mode)
        result["cpu_time_and_context_checks"] = []
        for row in (largest, small):
            for mode in ("h1", "history_mix"):
                batch, history, identity, band = load(row, mode)
                assert batch.frames == history + 1 and band.shape[0] == 16
                assert history == (1 if mode == "h1" else 4)
                assert bool(batch.frame_mask.all()) and bool(torch.isfinite(batch.x).all())
                relative_ps = batch.time_ps[0] - batch.time_ps[0, history - 1]
                expected_ps = torch.tensor([-age * row["delta_ps"] for age in identity["history_ages_delta"]]
                                           + [identity["target_lag_ps"]])
                torch.testing.assert_close(relative_ps, expected_ps, rtol=0, atol=0)
                absolute_ps = torch.tensor(identity["absolute_time_ps"], dtype=torch.float64)
                torch.testing.assert_close(absolute_ps - identity["absolute_t0_ps"], expected_ps.double(), rtol=0, atol=0)
                result["cpu_time_and_context_checks"].append(dict(system=row["system"],
                    plan_index=row["plan_index"], mode=mode, atoms=batch.atom_count,
                    history=history, target_frames=1, relative_time_ps=relative_ps.tolist(),
                    absolute_time_ps=identity["absolute_time_ps"], band_frames=16))
        torch.cuda.set_per_process_memory_fraction(args.cap_GiB * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(71)
        model = CoordinateStateDenoiser(cfg).cuda().train()
        report = load_coordinate_weights(model, PARENT)
        assert not report["newly_initialized_modules"] and model.checkpoint_query_blocks
        result["parameters"] = model.contract()["parameters"]
        batch, history, identity, band = load(largest)
        batch = batch.to("cuda")
        observed = slice_clip_frames(batch, 0, history)
        target, query_times = batch.x[history:], batch.time_ps[:, history:]
        assert target.shape[0] == 1 and observed.frames == 4
        torch.cuda.reset_peak_memory_stats()
        result["before_max_rf_memory"] = memory_guard(args.physical_gpu, True)
        stage = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition = model.prepare_condition(observed, task="dynamic")
            assert condition.task == "dynamic" and condition.history_h.shape[0] == 4
            assert condition.observed.frames == 4 and torch.equal(condition.reference_A, observed.x[-1])
            loss, metrics = coordinate_state_objective(model, condition, target, query_times,
                generator=torch.Generator(device="cuda").manual_seed(20261027))
        loss.backward()
        torch.cuda.synchronize()
        result["max_H4_RF"] = dict(identity=identity, atoms=batch.atom_count,
            history_frames=history, target_frames=1, bf16=True, loss=float(loss.detach()),
            seconds=time.monotonic() - stage, **gradient_summary(model), **peak(), metrics=metrics)
        assert torch.isfinite(loss)
        save()
        model.zero_grad(set_to_none=True)
        del batch, observed, target, query_times, condition, loss, band
        gc.collect()
        torch.cuda.empty_cache()

        batch, history, identity, support = load(small)
        batch = batch.to("cuda")
        observed = slice_clip_frames(batch, 0, history)
        target, query_times = batch.x[history:], batch.time_ps[:, history:]
        support = support.to("cuda").requires_grad_()
        auxiliary = LigandSourceGeometry(cfg["ligand_geometry"], 71 + 20261027, "cuda")
        result["before_aux_memory"] = memory_guard(args.physical_gpu, True)
        torch.cuda.reset_peak_memory_stats()
        stage = time.monotonic()
        seen = []
        def check_inputs(module, args_, kwargs):
            condition = kwargs["condition"]
            assert args_[0].shape == target.shape and condition.task == "dynamic"
            assert condition.observed.frames == history and torch.equal(condition.observed.x, observed.x)
            assert torch.equal(condition.reference_A, observed.x[-1])
            assert torch.equal(kwargs["query_time_ps"], query_times)
            seen.append(float(kwargs["progress"][0]))
        hook = model.register_forward_pre_hook(check_inputs, with_kwargs=True)
        with auxiliary.isolated_rng():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                endpoint = model.sample_truncated(observed, query_times, tail_intervals=4,
                                                  generator=auxiliary.generator)
                endpoint.retain_grad()
                source_calls = len(seen)
                raw_loss, geom = ligand_geometry_band_loss(endpoint, target, observed,
                    cfg["ligand_geometry"]["band"], band_reference_A=support)
                weighted = cfg["ligand_geometry"]["weight"] * raw_loss
            weighted.backward()
        hook.remove()
        torch.cuda.synchronize()
        ligand = observed.loss_mask & (observed.component_id > 0)
        ligand_grad = float(endpoint.grad[:, ligand].norm())
        assert source_calls == 35 and endpoint.shape == target.shape and torch.isfinite(endpoint).all()
        assert support.grad is None and torch.isfinite(endpoint.grad).all() and ligand_grad > 0
        assert not bool(endpoint.grad[:, observed.component_id == 0].any())
        assert geom["geometry_band_frames"] == 16 and geom["local_geometry_bonds"] > 0 and geom["local_geometry_angles"] > 0
        result["small_irregular_Q1_aux"] = dict(identity=identity, atoms=batch.atom_count,
            history_frames=history, target_frames=1, bf16=True, native_source_NFE=source_calls,
            backward_recompute_calls=len(seen) - source_calls, tail_intervals=4,
            raw_loss=float(raw_loss.detach()), weighted_loss=float(weighted.detach()),
            endpoint_ligand_gradient_norm=ligand_grad, endpoint_protein_gradient_zero=True,
            support_detached=True, all_forward_inputs_observed_only=True,
            seconds=time.monotonic() - stage, **gradient_summary(model), **peak(), metrics=geom)
        data.close()
        result.update(status="completed", sampling_trajectories=1, seconds=time.monotonic() - started)
        save()
        print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as error:
        result.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        save()
        raise


if __name__ == "__main__":
    main()
