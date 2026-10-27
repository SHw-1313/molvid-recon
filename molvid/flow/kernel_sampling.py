"""Observed-history transition kernel and generated-history rollout in physical ps.

Only observed coordinates and requested timestamps enter this interface. Q1 is
an actual dynamic model call with one future slot; no joint slicing, static
bypass, bridge endpoints, independent alignment or ground-truth history reset.
The kernel can later supply coarse anchors, but no bridge model is implemented.
"""
from dataclasses import replace
import torch


def update_history(observed, coordinates_A, times_ps):
    """Keep atom identities and a common coordinate frame, recomputing group centers."""
    if coordinates_A.ndim!=3 or coordinates_A.shape[1:]!=observed.x.shape[1:]:
        raise ValueError('History must have the same packed atoms and XYZ units')
    if times_ps.shape!=(observed.batch_size,len(coordinates_A)):
        raise ValueError('One physical timestamp per observed frame/system required')
    if not bool(torch.isfinite(times_ps).all() and (times_ps[:,1:]>times_ps[:,:-1]).all()):
        raise ValueError('Actual history timestamps must be finite and increasing')
    keys=torch.stack((observed.abid,observed.component_id,observed.block_id),-1)
    _,group=torch.unique(keys,dim=0,return_inverse=True)
    centers=coordinates_A.new_zeros(len(coordinates_A),int(group.max())+1,3)
    centers.index_add_(1,group,coordinates_A)
    centers/=torch.bincount(group)[None,:,None]
    return replace(observed,x=coordinates_A,bpos=centers[:,group],time_ps=times_ps,
        delta_time_ps=times_ps[:,1:]-times_ps[:,:-1],
        frame_mask=torch.ones_like(times_ps,dtype=torch.bool))


@torch.no_grad()
def sample_transition(model, observed, target_time_ps, *, generator):
    if model.scheme!='flow' or observed.frames not in (1,4):
        raise ValueError('Current kernel supports observed H1/H4 and native Flow')
    if target_time_ps.shape!=(observed.batch_size,1):
        raise ValueError('A transition generates exactly one future time')
    if not bool((target_time_ps[:,0]>observed.time_ps[:,-1]).all()):
        raise ValueError('Target time must follow the actual current state')
    condition=model.prepare_condition(observed,task='dynamic')
    return model.sample(condition,target_time_ps,generator=generator)


@torch.no_grad()
def rollout_transitions(model, observed, query_time_ps, *, generator):
    """Maintain latest H real/generated states; never invent missing fine history."""
    if query_time_ps.ndim!=2 or query_time_ps.shape[0]!=observed.batch_size or query_time_ps.shape[1]<1:
        raise ValueError('Expected nonempty physical query timestamps [B,Q]')
    if not bool((query_time_ps[:,1:]>query_time_ps[:,:-1]).all()):
        raise ValueError('Rollout target times must increase')
    history=observed.frames;current=observed;pieces=[];history_times=[]
    for frame in range(query_time_ps.shape[1]):
        times=query_time_ps[:,frame:frame+1]
        history_times.append(current.time_ps.detach().cpu().tolist())
        endpoint=sample_transition(model,current,times,generator=generator)
        pieces.append(endpoint)
        current=update_history(observed,torch.cat((current.x,endpoint),0)[-history:],
                               torch.cat((current.time_ps,times),1)[:,-history:])
    frames=len(pieces);nfe=2*model.settings['sampler_intervals']-1
    model.last_sampling_info=dict(mode='dynamic_q1_rollout',history_frames=history,
        actual_history_time_ps=history_times,query_time_ps=query_time_ps.detach().cpu().tolist(),
        total_NFE=frames*nfe,effective_evaluations_per_frame=[nfe]*frames,
        atom_frame_evaluations=frames*nfe*observed.atom_count,
        history_encoded_frames=frames*history,condition_policy='latest_generated_history_no_GT_future_no_realign')
    return torch.cat(pieces,0)
