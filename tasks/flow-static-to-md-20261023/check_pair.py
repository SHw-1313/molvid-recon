"""CPU-only comparison of saved P1 dynamic streams and extra Q1 exposure.

No model, data loader or random generator is constructed. The two new-parent
arms must share the strict weights-only initialization report, not trained
weights. Old/new parent weights and the private replay state are not paired.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flow_static_to_md_20261023'
ARMS = ('old_static', 'new_static', 'new_static_replay')
STATE_KEYS = ('seed', 'exposure', 'mixture', 'generator_rng', 'torch_rng',
              'cuda_rng', 'python_rng', 'numpy_rng')
ROW_KEYS = ('step', 'task', 'sample', 'system', 'visit', 'domain',
            'clock_sampling', 'time_bucket_id', 'history', 'queries',
            'progress_mean', 'valid_atoms', 'query_frames')


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(v, w) for v, w in zip(a, b))
    return a == b


def finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check(run, step):
    filename = 'coordinate_final.pt' if step == 512 else f'checkpoint_step{step}.pt'
    paths = {arm: run / arm / filename for arm in ARMS}
    saved = {arm: torch.load(path, map_location='cpu', weights_only=False)
             for arm, path in paths.items()}
    rows = {arm: [row for line in (run / arm / 'updates.jsonl').read_text().splitlines()
                  if (row := json.loads(line))['step'] <= step] for arm in ARMS}
    old, new, replay = (saved[arm] for arm in ARMS)
    checks = {}
    for key in STATE_KEYS:
        checks[f'dynamic_{key}'] = all(equal(old[key], saved[arm][key]) for arm in ARMS[1:])
    checks['schema_step_seed_schedule'] = all(
        c['schema'] == 'molvid.coordinate_state.checkpoint.v1' and c['step'] == step
        and c['seed'] == 71 and c['settings']['updates'] == 512 for c in saved.values())
    recipes = [{k: v for k, v in c['settings'].items() if k != 'static_replay'} for c in saved.values()]
    checks['same_recipe_except_replay'] = all(equal(recipes[0], r) for r in recipes[1:])
    checks['history_recipe'] = all(r['history_sampling'] == 'alternating_optimizer_updates'
                                 and r['scheme'] == 'flow' and not r.get('reference_concat', False)
                                 and r.get('query_qk_mode') == 'group' for r in recipes)
    checks['sequential_complete_rows'] = all(
        [r['step'] for r in values] == list(range(1, step + 1)) for values in rows.values())
    mismatches = []
    for arm in ARMS[1:]:
        for a, b in zip(rows[ARMS[0]], rows[arm]):
            bad = [key for key in ROW_KEYS if key not in a or key not in b or not equal(a[key], b[key])]
            if bad:
                mismatches.append(dict(arm=arm, step=a['step'], keys=bad))
    checks['actual_dynamic_sample_clock_H_Q_progress'] = (
        checks['sequential_complete_rows'] and not mismatches)
    checks['dynamic_H4Q12_H8Q8'] = all(
        r['task'] == 'dynamic' and r['history'] == (4 if r['step'] % 2 else 8)
        and r['queries'] == list(range(r['history'], 16))
        and r['query_frames'] == len(r['queries']) and r['valid_atoms'] > 0
        for values in rows.values() for r in values)
    checks['dynamic_exposure_from_actual_rows'] = all(c['exposure'] == dict(
        updates=step, dynamic_examples=step, static_examples=0, target_frames=step * 10,
        atom_query_pairs=sum(r['valid_atoms'] * r['query_frames'] for r in rows[arm]))
        for arm, c in saved.items())
    checks['same_model_keys_and_shapes'] = all(
        c['model'].keys() == old['model'].keys()
        and all(v.shape == old['model'][k].shape for k, v in c['model'].items()) for c in saved.values())
    checks['finite_model_optimizer_and_training_logs'] = all(
        finite(c['model']) and finite(c['optimizer']) and finite(rows[arm]) for arm, c in saved.items())
    checks['optimizer_exactly_one_update_per_dynamic_step'] = all(
        bool(c['optimizer']['state']) and all(int(v['step']) == step for v in c['optimizer']['state'].values())
        for c in saved.values())
    warmup = recipes[0]['warmup_updates']
    fraction = max(0., (step - 1 - warmup) / max(1, 512 - warmup - 1))
    factor = step / warmup if step <= warmup else .1 + .45 * (1 + math.cos(math.pi * fraction))
    expected_lr = recipes[0]['learning_rate'] * factor
    checks['unshortened_512_schedule_lr'] = all(
        all(math.isclose(g['lr'], expected_lr, rel_tol=0, abs_tol=1e-15)
            for g in c['optimizer']['param_groups']) for c in saved.values())
    initial = {arm: c['initialization_report'] for arm, c in saved.items()}
    checks['new_arms_same_strict_parent_initialization'] = (
        equal(initial['new_static'], initial['new_static_replay'])
        and initial['new_static']['kind'] == 'coordinate_state_weights_only'
        and initial['new_static']['parent_step'] == 8192
        and initial['new_static']['newly_initialized_modules'] == [])
    checks['old_parent_is_completed_4096'] = (
        initial['old_static']['kind'] == 'coordinate_state_weights_only'
        and initial['old_static']['parent_step'] == 4096)
    checks['off_arms_have_no_replay_state'] = all(
        'static_replay' not in saved[arm] and 'replay_exposure' not in saved[arm]
        and all('replay' not in r for r in rows[arm]) for arm in ARMS[:2])
    replay_rows = [r for r in rows['new_static_replay'] if r['replay_applied']]
    calls = step // 4
    state = replay['static_replay']
    details = [r['replay'] for r in replay_rows]
    checks['replay_exactly_every_fourth_update'] = (
        [r['step'] for r in replay_rows] == list(range(4, step + 1, 4))
        and all(('replay' in r) == r['replay_applied'] for r in rows['new_static_replay']))
    checks['replay_Q1_recipe_and_weighted_objective'] = (
        state['schema'] == 'molvid.static_replay.v1'
        and equal(state['config'], replay['settings']['static_replay'])
        and (state['config']['every'], state['config']['weight'], state['config']['corruption_A']) == (4, .25, .2)
        and all(r['replay']['history'] == 1 and r['replay']['query_frames'] == 1
                and r['replay']['weight'] == .25 and r['replay']['corruption_A'] == .2
                and r['loss'] == r['dynamic_RF_loss'] + r['replay']['weighted_loss']
                and math.isclose(r['replay']['weighted_loss'], .25 * r['replay']['RF_loss'], rel_tol=1e-7)
                for r in replay_rows))
    expected_replay_exposure = dict(backward_calls=calls, static_examples=calls,
        target_frames=calls, atom_query_pairs=sum(r['valid_atoms'] for r in details))
    checks['extra_Q1_exposure_from_actual_rows'] = (
        replay['replay_exposure'] == state['exposure'] == expected_replay_exposure
        and all(r['exposure']['backward_calls'] == i and r['exposure']['target_frames'] == i
                for i, r in enumerate(details, 1)))
    replay_domains = Counter(r['domain'] for r in details)
    checks['replay_mixture_counts_from_rows'] = state['mixture']['counts'] == {
        'static:' + k: v for k, v in replay_domains.items()}
    if step == 512:
        checks['final_128_extra_Q1_50_25_25'] = (
            calls == 128 and replay_domains == {'protein': 64, 'atlas': 32, 'misato': 32})
    checks['private_replay_rng_saved_separately'] = (
        set(state['isolated_rng']) == {'python', 'numpy', 'torch_cpu', 'cuda'}
        and len(state['isolated_rng']['cuda']) == len(replay['cuda_rng'])
        and not equal(state['generator_rng'], replay['generator_rng']))
    different = sum(not equal(v, replay['model'][k]) for k, v in new['model'].items())
    return dict(passed=all(checks.values()), step=step, checks=checks,
        checkpoints={arm: str(path) for arm, path in paths.items()},
        dynamic_exposure=old['exposure'], dynamic_H_counts=dict(Counter(r['history'] for r in rows['old_static'])),
        dynamic_clock_counts=dict(Counter(r['time_bucket_id'] for r in rows['old_static'])),
        mismatched_dynamic_rows=mismatches, replay_exposure=replay['replay_exposure'],
        replay_domain_counts=dict(replay_domains), expected_learning_rate=expected_lr,
        parent_initialization=initial, new_vs_replay_different_state_tensors=different,
        scope='Exact saved dynamic RNG/mixture/exposure and logged sample/clock/H/Q/progress; '
              'progress_mean is the actual progress for the one-system training batch. '
              'Physical timestamps are identified by the same sample, time bucket and query indices; '
              'records and forward tensors are not reread. Same new-parent strict load report is '
              'initialization evidence; trained weights/optimizer values/private replay RNG may differ.',
        cuda_initialized=torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=(32, 128, 512), default=512)
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run in existing torch-ito with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    result = check(args.run_root, args.step)
    path = args.output or args.run_root / f'pair_check_step{args.step}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=result['passed'], step=args.step, output=str(path),
                         failed=[key for key, value in result['checks'].items() if not value])))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
