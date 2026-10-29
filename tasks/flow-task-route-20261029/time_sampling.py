"""Finite physical short-block rollout and asynchronous RF integration.

All state is generated XYZ in Angstrom; physical query times remain ps.
The asynchronous integration parameter is dimensionless flow progress.
"""
from dataclasses import replace
import torch


def flow_heun(model, condition, times, noise, intervals):
    state = condition.reference_A[None]+model.rho_A*noise
    dt = 1./intervals
    for index in range(intervals):
        s = state.new_full((condition.observed.batch_size,), index/intervals)
        u = model(state, condition=condition, query_time_ps=times, progress=s)
        proposal = state+dt*u
        if index < intervals-1:
            u_next = model(proposal, condition=condition, query_time_ps=times, progress=s+dt)
            state = state+.5*dt*(u+u_next)
        else:
            state = proposal
    return state


def history_with_coordinates(observed, coordinates, times):
    """Retain topology; recompute bpos from generated/observed coordinates only."""
    keys = torch.stack((observed.abid, observed.component_id, observed.block_id), -1)
    _, group = torch.unique(keys, dim=0, return_inverse=True)
    centers = coordinates.new_zeros(len(coordinates), int(group.max())+1, 3)
    centers.index_add_(1, group, coordinates)
    centers /= torch.bincount(group)[None, :, None]
    return replace(observed, x=coordinates, bpos=centers[:, group], time_ps=times,
        delta_time_ps=times[:, 1:]-times[:, :-1], frame_mask=torch.ones_like(times, dtype=torch.bool))


@torch.no_grad()
def sample_short_blocks(model, condition, query_time_ps, *, intervals, generator):
    observed = condition.observed
    history = observed.frames
    if history not in (4, 8):
        raise ValueError('Short-block study retains native H4/H8 only')
    block_size = 4 if history == 4 else 8
    # Pre-draw the complete standard-normal tensor exactly like the control.
    noise = torch.randn((query_time_ps.shape[1], *condition.reference_A.shape),
                        device=condition.reference_A.device, generator=generator)
    history_x, history_t = observed.x, observed.time_ps
    pieces, block_lengths = [], []
    for begin in range(0, query_time_ps.shape[1], block_size):
        end = min(begin+block_size, query_time_ps.shape[1])
        current_condition = condition if begin == 0 else model.prepare_condition(
            history_with_coordinates(observed, history_x[-history:], history_t[:, -history:]),
            route_task=condition.route_task)
        part = flow_heun(model, current_condition, query_time_ps[:, begin:end],
                         noise[begin:end], intervals)
        pieces.append(part); block_lengths.append(end-begin)
        history_x = torch.cat((history_x, part), 0)[-history:]
        history_t = torch.cat((history_t, query_time_ps[:, begin:end]), 1)[:, -history:]
    frames = query_time_ps.shape[1]
    model.last_sampling_info = dict(mode='short_block', block_size=block_size,
        block_lengths=block_lengths, history_frames=history,
        total_NFE=len(pieces)*(2*intervals-1),
        effective_evaluations_per_frame=[2*intervals-1]*frames,
        effective_intervals_per_frame=[intervals]*frames,
        atom_frame_evaluations=(2*intervals-1)*frames*len(condition.reference_A),
        history_encoded_frames=len(pieces)*history,
        condition_policy='previous_generated_frames_only_no_target_or_future_alignment')
    return torch.cat(pieces, 0)

