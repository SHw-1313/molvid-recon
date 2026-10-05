"""Fixed train-clip learnability check; no validation or model selection."""
import argparse
import json
from pathlib import Path
import time
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.native_forecast import molvid_forecast
from molvid.flow.coordinate import CoordinateFlowModel, coordinate_flow_loss
from molvid.latent.types import QuerySpec
from molvid.provenance import freeze_run_identity
from molvid.runtime import configure_device, atomic_write_json, sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent, atomic_torch_save, RELEASE


def fit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    device = configure_device("cuda", deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.06)
    torch.manual_seed(903)
    items = load_ca_cache(args.cache, "train")
    item = min(items, key=lambda i: (len(i["record"]["atype"]), i["record"]["sample_id"]))
    parent, _, release = load_parent(device)
    cpu = collate_clip_records([item["record"]])
    observed_cpu = slice_clip_frames(cpu, 0, 4)
    observed = observed_cpu.to(device)
    target = cpu.x[4:6].to(device)
    query = QuerySpec(cpu.time_ps[:, 4:6].to(device), cpu.frame_mask[:, 4:6].to(device))
    bonds = cpu.bond_index.to(device)
    reference_lengths = (target[:, bonds[0]] - target[:, bonds[1]]).norm(dim=-1)

    def metrics(x):
        distance = (x[:, bonds[0]] - x[:, bonds[1]]).norm(dim=-1)
        return dict(coordinate_rmse_A=float((x - target).square().sum(-1).mean().sqrt()),
                    bond_rmse_A=float((distance - reference_lengths).square().mean().sqrt()),
                    displacement_rms_A=float((x - observed.x[-1:]).square().sum(-1).mean().sqrt()))

    model = CoordinateFlowModel(parent)
    protocol = dict(kind="train_clip_overfit_feasibility_not_generalization",
        system=item["metadata"]["system"], sample_id=item["record"]["sample_id"], atoms=cpu.atom_count,
        history=4, queries=2, query_dt_ps=100, updates=512, seed=903, source_seed=904,
        evaluation_seeds=[905, 906, 907], solver_steps=16, static_updates=0,
        objective="coordinate_RF_only", lr=dict(dit=5e-5, history=2.5e-5, readout=5e-4),
        no_checkpoint_selection=True, memory_fraction=.06, co_resident=True,
        contract=model.contract(), warm_start=model.warm_start_report,
        parent=release["selected_checkpoint"], test_opened=False)
    freeze_run_identity(args.output, inputs=[args.cache / "complete.json", RELEASE],
                        purpose="Coordinate flow fixed-budget learnability gate", protocol=protocol)
    atomic_write_json(args.output / "protocol.json", protocol)
    started = time.monotonic()
    with ExecutionEvidence(args.output):
        # Shared trunk: use no_grad, never freeze parent parameters here.
        with torch.no_grad():
            reference = molvid_forecast(parent, observed_cpu, query.time_ps, device=device, seed=905)
        baseline = metrics(reference)
        del parent, reference
        model.prepare_batch(observed_cpu)

        @torch.no_grad()
        def evaluate(label):
            model.eval()
            condition = model.prepare_condition(observed)
            rows = []
            for seed in (905, 906, 907):
                source = model.source(condition=condition, query=query,
                    generator=torch.Generator(device=device).manual_seed(seed))
                generated = model.rollout(source, condition=condition, query=query, steps=16)
                rows.append(dict(label=label, seed=seed, metrics=metrics(generated)))
            return rows

        comparisons = evaluate("initial_zero_velocity")
        groups = [{"params": model.dit.parameters(), "lr": 5e-5},
                  {"params": model.history_encoder.parameters(), "lr": 2.5e-5},
                  {"params": model.readout.parameters(), "lr": 5e-4}]
        if model.motion_encoder is not None:
            groups.append({"params": model.motion_encoder.parameters(), "lr": 2.5e-5})
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        generator = torch.Generator(device=device).manual_seed(904)
        model.train()
        with (args.output / "updates.jsonl").open("x") as log:
            for step in range(1, 513):
                if time.monotonic() - started > 1800:
                    raise TimeoutError("30 minute fixed learnability budget")
                optimizer.zero_grad(set_to_none=True)
                condition = model.prepare_condition(observed)
                source = model.source(condition=condition, query=query, generator=generator)
                progress = torch.rand(1, device=device, generator=generator)
                state = torch.lerp(source, target, progress[:, None, None])
                prediction = model(state, condition=condition, query=query, flow_time=progress)
                loss = coordinate_flow_loss(prediction, target - source, batch=observed, query=query)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite overfit gradient")
                optimizer.step()
                row = dict(step=step, loss=float(loss), grad_norm=float(norm), seconds=time.monotonic() - started)
                log.write(json.dumps(row) + "\n"); log.flush()
                if step % 32 == 0:
                    print(json.dumps(row), flush=True)
        del condition, source, state, prediction, loss
        optimizer.zero_grad(set_to_none=True)
        comparisons += evaluate("final512")
        checkpoint = args.output / "coordinate_step_0512.pt"
        atomic_torch_save(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            contract=model.contract(), completed_updates=512, parent=release["selected_checkpoint"],
            warm_start=model.warm_start_report, schema="molvid.coordinate_flow.prototype.checkpoint.v1"))
        atomic_write_json(args.output / "summary.json", dict(status="completed", updates=512, systems=1,
            train_only=True, generalization_tested=False, kinetics_tested=False, thermodynamics_tested=False,
            parent_unadapted_A1=baseline, persistence=metrics(observed.x[-1:].expand_as(target)),
            target=metrics(target), comparisons=comparisons, checkpoint_sha256=sha256_file(checkpoint),
            seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(), test_opened=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/workspace/molvid-ca-detail-20261005/runs/ca_detail_20261005/cache"))
    parser.add_argument("--output", type=Path, required=True)
    fit(parser.parse_args())
