"""Free-running geometric bridge trajectories; only the first prefix is observed.

Coordinates are Angstrom, query timestamps are ps. Subsequent history is made
exclusively from generated coordinates. This evaluator does not assert a
Markov/semigroup law or equilibrium merely because it can recurse.
"""
from dataclasses import replace
import torch


@torch.no_grad()
def recursive_bridge_forecast(model, observed, query_time_ps, *, seed, block_frames=12, steps=10):
    """Return [Q,N,3] for one system, retaining the original history length.

The current bridge reads atom/topology fields and x, not bpos. For generated
history, bpos is derived block-centroid metadata, never copied clean targets.
This boundary is specific to this coordinate bridge, not a universal adapter
for models that require a different block-representative definition.
    """
    if observed.batch_size != 1 or query_time_ps.ndim != 2 or query_time_ps.shape[0] != 1:
        raise ValueError('This rollout evaluator handles one full system per call')
    if block_frames < 1 or query_time_ps.shape[1] < 1:
        raise ValueError('Positive query and block lengths required')
    device = next(model.parameters()).device
    history = observed.to(device)
    query_time_ps = query_time_ps.to(device)
    if not bool((query_time_ps[:, :1] > history.time_ps[:, -1:]).all()):
        raise ValueError('Queries must follow the observed prefix')
    if not bool((query_time_ps[:, 1:] > query_time_ps[:, :-1]).all()):
        raise ValueError('Physical query timestamps must be strictly increasing')
    keep = history.frames
    generator = torch.Generator(device=device).manual_seed(seed)
    _, group = torch.unique(torch.stack((history.abid, history.block_id), -1), dim=0, return_inverse=True)
    counts = torch.bincount(group)
    predictions = []
    for start in range(0, query_time_ps.shape[1], block_frames):
        times = query_time_ps[:, start:start+block_frames]
        condition = model.prepare_condition(history)
        prediction = model.sample(condition, times, steps=steps, generator=generator).float()
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(f'Nonfinite free-running block at query{start}')
        predictions.append(prediction)
        summed = prediction.new_zeros(len(prediction), counts.numel(), 3).index_add_(1, group, prediction)
        generated_bpos = (summed/counts[None, :, None])[:, group]
        next_times = torch.cat((history.time_ps, times), 1)[:, -keep:]
        history = replace(history,
            x=torch.cat((history.x, prediction), 0)[-keep:],
            bpos=torch.cat((history.bpos, generated_bpos), 0)[-keep:],
            time_ps=next_times, delta_time_ps=next_times[:, 1:]-next_times[:, :-1],
            frame_mask=torch.ones_like(next_times, dtype=torch.bool))
    return torch.cat(predictions, 0)
