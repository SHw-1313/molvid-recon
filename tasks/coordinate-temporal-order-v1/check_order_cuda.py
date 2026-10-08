"""Exercise the changed interaction order on a real TRAIN window and source.

No optimizer updates or validation selection. Static equality, warm-start
semantics, dynamic gradients and complete coordinate sampling are the risks
specific to this ablation; scientific evidence comes from subsequent training.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.runtime import atomic_write_json
from molvid.training.coordinate_state import load_coordinate_weights


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scheme', choices=('flow', 'edm'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    torch.set_num_threads(4)
    arm = json.loads(args.protocol.read_text())['arms'][args.scheme]
    config = json.loads(Path(arm['recipe']).read_text())
    control_config = json.loads(Path(arm['source_recipe']).read_text())
    differences = {k for k in set(config)|set(control_config) if config.get(k)!=control_config.get(k)}
    if differences != {'schema', 'interaction_order'}:
        raise ValueError('This contrast changes interaction order only')
    mixture = BridgeMixture(config['data'], arm['seed'])
    try:
        batch, task, history, queries, identity = mixture.next()
    finally:
        mixture.close()
    if task != 'dynamic':
        raise ValueError('Use the actual first MD training example')
    batch = batch.to('cuda')
    observed = slice_clip_frames(batch, 0, history)
    target, times = batch.x[queries], batch.time_ps[:, queries]
    snapshot = observed.x.clone()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    candidate = CoordinateStateDenoiser(config).cuda()
    inherited = load_coordinate_weights(candidate, arm['warm_start'])
    control = CoordinateStateDenoiser(control_config).cuda()
    load_coordinate_weights(control, arm['warm_start'])
    if candidate.state_dict().keys()!=control.state_dict().keys() or any(
            not torch.equal(v, control.state_dict()[k]) for k,v in candidate.state_dict().items()):
        raise RuntimeError('Control/candidate parameter inheritance differs')
    rng = torch.Generator(device='cuda').manual_seed(417)
    singleton = slice_clip_frames(observed, 0, 1)
    corrupted = singleton.x + candidate.sigma_data*torch.randn(singleton.x.shape, device='cuda', generator=rng)
    _, group = torch.unique(torch.stack((singleton.abid, singleton.block_id), -1), dim=0, return_inverse=True)
    centers = corrupted.new_zeros(1, int(group.max())+1, 3).index_add_(1, group, corrupted)
    centers /= torch.bincount(group)[None, :, None]
    singleton = replace(singleton, x=corrupted, bpos=centers[:, group])
    candidate.eval(); control.eval()
    progress = corrupted.new_full((batch.batch_size,), .4 if args.scheme=='flow' else candidate.sigma_data)
    with torch.no_grad():
        a = candidate(corrupted, condition=candidate.prepare_condition(singleton, task='static'),
                      query_time_ps=singleton.time_ps, progress=progress)
        b = control(corrupted, condition=control.prepare_condition(singleton, task='static'),
                    query_time_ps=singleton.time_ps, progress=progress)
    static_error = float((a-b).abs().max())
    if not torch.allclose(a, b, atol=2e-4, rtol=2e-4):
        raise RuntimeError(f'Static bypass changed: {static_error}')
    del control, a, b
    candidate.train()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['bf16']):
        condition = candidate.prepare_condition(observed)
        loss, metrics = coordinate_state_objective(candidate, condition, target, times, generator=rng)
    loss.backward()
    gradients = {}
    for name, module in [('spatial', candidate.spatial), ('tokens', candidate.tokens),
                         ('temporal', candidate.temporal), ('head', candidate.head)]:
        values = [p.grad for p in module.parameters() if p.grad is not None]
        norm = sum(float(g.float().square().sum()) for g in values)**.5
        if not values or not all(bool(torch.isfinite(g).all()) for g in values) or norm==0:
            raise RuntimeError(f'Invalid dynamic gradient through {name}')
        gradients[name] = norm
    candidate.zero_grad(set_to_none=True)
    del condition, loss
    candidate.eval()
    states = []
    def record_state(module, arguments):
        current = arguments[0]
        states.append(float((current-observed.x[-1:]).square().mean()))
    handle = candidate.register_forward_pre_hook(record_state)
    with torch.no_grad():
        condition = candidate.prepare_condition(observed)
        calls, graphs = candidate.forward_calls, candidate.geometry_calls
        generated = candidate.sample(condition, times, generator=rng)
    handle.remove()
    nfe, rebuilt = candidate.forward_calls-calls, candidate.geometry_calls-graphs
    if nfe!=35 or rebuilt!=35*len(target) or len(set(states))<=1:
        raise RuntimeError('Incomplete source sampling or unchanged candidate coordinates')
    if generated.shape!=target.shape or not bool(torch.isfinite(generated).all()) or not torch.equal(observed.x, snapshot):
        raise RuntimeError('Invalid generated XYZ or modified observation')
    torch.cuda.synchronize()
    atomic_write_json(args.output, dict(status='completed', scheme=args.scheme, scientific_efficacy=False,
        optimizer_updates=0, TRAIN_identity=identity, history=history, queries=queries,
        initialization_report=inherited, static_max_difference_A=static_error,
        dynamic_gradient_norms=gradients, training_batch_metrics=metrics,
        complete_source_NFE=nfe, current_geometry_rebuilds=rebuilt,
        candidate_state_MSE_from_reference_A2_each_call=states,
        generated_shape=list(generated.shape), observation_unchanged=True,
        parameters=sum(p.numel() for p in candidate.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started,
        checkpoint_reload='Actual v2 trained checkpoint will be loaded by shared full-panel evaluator; not claimed by this pretraining exercise'))


if __name__ == '__main__':
    main()
