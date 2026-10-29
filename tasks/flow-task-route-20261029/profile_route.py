"""Seven disposable RF updates: fixed-plan maximum, then joint/Q1 first+2 steady.

No ligand auxiliary or generated trajectory is run. Both adapter branches are
active in both arms, so one routed model profiles the common execution shape.
CUDA execution and process-wall accounting belong to the study runner.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.bridge_mixture import BridgeMixture
from molvid.data.kernel_windows import KernelWindows
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.training.coordinate_state import load_coordinate_weights

TASK = ROOT / 'tasks/flow-task-route-20261029'
RUN = ROOT / 'runs/flow_task_route_20261029'
PARENT = '/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocator-GiB', type=float, default=32.)
    parser.add_argument('--output', type=Path, default=RUN / 'profile.json')
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito':
        raise RuntimeError('Existing torch-ito required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(status='running', updates=[], auxiliary_performed=False,
                  auxiliary_small_prior_seconds=9.89, auxiliary_seconds_scope='prior 3M35 small check only; not maximum or mean')
    def save(): args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    save()
    data = mixture = None
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(args.allocator_GiB * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        cfg = json.loads((TASK / 'configs/routed.json').read_text())
        rows = json.loads(Path(cfg['kernel_training']['manifest']).read_text())['rows']
        h4 = sorted((r for r in rows if len(r['history_ages_delta']) == 4), key=lambda r:(r['atom_count'], r['plan_index']))
        largest, typical = h4[-1], h4[len(h4)//2]
        assert largest['atom_count'] == 11955
        data = KernelWindows(cfg['kernel_training'], cfg['data'], 71)
        def load(row):
            data.load_state_dict(dict(cursor=row['plan_index'], plan_digest=data.digest))
            batch, history, identity, _ = data.next('history_mix')
            return batch, history, [history], identity
        mixture = BridgeMixture(cfg['data'], 71)
        joint, task, _, _, joint_identity = mixture.next()
        assert task == 'dynamic'
        cases = [('max_plan_Q1', *load(largest), 'q1', 1),
                 ('typical_first_joint', joint, 4, list(range(4, joint.frames)), joint_identity, 'joint', 3),
                 ('typical_median_H4_Q1', *load(typical), 'q1', 3)]
        torch.manual_seed(71)
        model = CoordinateStateDenoiser(cfg).cuda().train()
        result['initialization'] = load_coordinate_weights(model, PARENT)
        result['contract'] = model.contract()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(20261029)
        for label, host, history, queries, identity, route, repeats in cases:
            batch = host.to('cuda')
            observed = slice_clip_frames(batch, 0, history)
            target, times = batch.x[queries], batch.time_ps[:, queries]
            for iteration in range(repeats):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    condition = model.prepare_condition(observed, route_task=route)
                    loss, metrics = coordinate_state_objective(model, condition, target, times, generator=generator)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['clip_grad'], error_if_nonfinite=True)
                assert norm > 0 and torch.isfinite(loss)
                assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('task_adapters.'))
                optimizer.step(); torch.cuda.synchronize()
                result['updates'].append(dict(label=label, iteration=iteration, steady=iteration>0,
                    sample=batch.sample_id[0], identity=identity, atoms=batch.atom_count,
                    history=history, query_frames=len(queries), route_task=route,
                    seconds=time.monotonic()-begin, loss=float(loss.detach()), gradient_norm=float(norm),
                    frozen_base_gradient_none=True, all_adapter_gradients_finite=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), metrics=metrics))
                save()
                del condition, loss
            del batch, observed, target, times
            gc.collect(); torch.cuda.empty_cache()
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if data is not None: data.close()
        if mixture is not None: mixture.close()
        result['wall_seconds'] = time.monotonic()-started
        save(); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
