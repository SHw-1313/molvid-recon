"""One actual maximum TRAIN window/update; not an average-throughput estimate."""
import argparse
import json
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import molvid.flow.coordinate_state as coordinate_module

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective

SYSTEMS = {'protein': ('atlas_6sup_A', 17002), 'complex': ('6D6U', 16999)}


def profile(args, result):
    config_path = ROOT/'tasks/flow-capacity-20261020/configs'/f'{args.arm}_md.json'
    settings = json.loads(config_path.read_text())
    result.update(config=str(config_path), initialization='fresh_recipe_initialization',
                  scheme=settings['scheme'], bf16=settings['bf16'], seed=71,
                  model_module=str(Path(coordinate_module.__file__).resolve()),
                  checkpoint_query_blocks=settings.get('checkpoint_query_blocks', False))
    if Path(coordinate_module.__file__).resolve() != ROOT/'molvid/flow/coordinate_state.py':
        raise RuntimeError('Model import resolved outside this capacity worktree')
    if settings['scheme'] != 'flow' or not settings['bf16']:
        raise ValueError('This profile requires the frozen bf16 Flow recipe')
    torch.set_num_threads(settings['cpu_threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; CPU model fallback is not allowed')
    system, expected_atoms = SYSTEMS[args.domain]
    mixture = BridgeMixture(settings['data'], seed=71)
    try:
        buckets = mixture.groups[args.domain][system]
        bucket = min(buckets, key=lambda name: float(name.split('_')[1][:-2]))
        store, index = buckets[bucket][0]
        record = store[index]
        if record['split'] != 'train':
            raise ValueError('Profile record must belong to TRAIN')
        host = collate_clip_records([record])
        if host.frames != 16 or host.atom_count != expected_atoms:
            raise ValueError(f'Expected 16 frames/{expected_atoms} atoms, got {host.frames}/{host.atom_count}')
        # Transfer before closing mmap-backed stores; do not retain a full trajectory.
        batch = host.to('cuda')
        result.update(system=system, sample_id=host.sample_id[0], time_bucket_id=bucket,
                      store_index=index, atoms=host.atom_count, history_frames=4,
                      query_frames=12, time_ps=host.time_ps[0].tolist(), split='train')
    finally:
        mixture.close()
    torch.manual_seed(71)
    torch.cuda.manual_seed_all(71)
    torch.cuda.reset_peak_memory_stats()
    model = CoordinateStateDenoiser(settings).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    # Same first-update warmup factor as the existing trainer, not a shortened schedule.
    learning_rate = settings['learning_rate']/settings['warmup_updates']
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    generator = torch.Generator(device='cuda').manual_seed(71+20261008)
    observed = slice_clip_frames(batch, 0, 4)
    target, times = batch.x[4:16], batch.time_ps[:,4:16]
    result.update(parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  learning_rate=learning_rate, model_contract=model.contract(),
                  device=torch.cuda.get_device_name(), visible_device=0,
                  peak_includes_model_optimizer_and_one_update=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        condition = model.prepare_condition(observed)
        loss, metrics = coordinate_state_objective(model, condition, target, times, generator=generator)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad'],
                                              error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter()-started
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite_gradients = all(bool(torch.isfinite(g).all()) for g in grads)
    finite_parameters = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    finite_optimizer = all(bool(torch.isfinite(v).all()) for state in optimizer.state.values()
                           for v in state.values() if torch.is_tensor(v))
    result.update(loss=float(loss.detach()), metrics=metrics, grad_norm_before_clip=float(grad_norm),
        gradients_present=len(grads), finite_gradients=finite_gradients,
        finite_parameters=finite_parameters, finite_optimizer=finite_optimizer,
        update_seconds=update_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), model_forward_calls=model.forward_calls,
        optimizer_steps=1, training=True, checkpointing='existing_train_grad_enabled_paths',
        scope='one_maximum_H4Q12_window_upper_bound_not_mean_throughput')
    if not bool(torch.isfinite(loss)) or not all((finite_gradients,finite_parameters,finite_optimizer)):
        raise RuntimeError('Nonfinite profile update')
    if model.forward_calls != 1:
        raise RuntimeError('Profile unexpectedly executed more than one objective forward')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('wide','collective'), default='wide')
    parser.add_argument('--domain', choices=('protein','complex'), default='protein')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/'runs/flow_capacity_20261020/profiles'/f'{args.arm}_{args.domain}.json'
    result = dict(schema='molvid.capacity.single_update_profile.v1', arm=args.arm, domain=args.domain,
                  passed=False, output=str(output))
    start = time.perf_counter()
    try:
        profile(args,result)
        result['passed'] = True
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        if torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        raise
    finally:
        result['process_seconds_until_report'] = time.perf_counter()-start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
