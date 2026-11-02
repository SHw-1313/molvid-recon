"""Native RF rollout with independent retained-history and query block lengths."""
import math,sys
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tasks/flow-task-route-20261029'))
from time_sampling import flow_heun,history_with_coordinates
SETTINGS={'h4q4':(4,4),'h4q8':(4,8),'h8q4':(8,4),'h8q8':(8,8),'h12q12':(12,12)}
def observed_at_zero(batch,entry):
    """Change only time origin; actual dt and native metadata remain intact."""
    times=batch.time_ps-batch.time_ps[:,-1:]
    declared=times.new_tensor(entry['observed_time_ps'])[None]
    if not torch.equal(times,declared):raise ValueError('Observed physical clock differs from frozen manifest')
    return replace(batch,time_ps=times)

@torch.no_grad()
def sample_hq(model,observed,query_time_ps,setting,*,generator=None,noise=None):
    H,Q=SETTINGS[setting]
    if model.scheme!='flow' or model.settings['sampler_intervals']!=18 or observed.frames!=H:
        raise ValueError('Native18-interval RF and declared H required')
    if query_time_ps.shape!=(observed.batch_size,50):raise ValueError('Exactly50 scored physical queries required')
    delta=query_time_ps[:,0]-observed.time_ps[:,-1]
    full_times=observed.time_ps[:,-1:]+delta[:,None]*torch.arange(1,61,device=observed.x.device)[None]
    if not bool((delta>0).all()) or not torch.allclose(query_time_ps,full_times[:,:50],atol=1e-4,rtol=1e-6):raise ValueError('One real regular clock per system required')
    if noise is None:noise=torch.randn((60,observed.atom_count,3),device=observed.x.device,generator=generator)
    if noise.shape!=(60,observed.atom_count,3):raise ValueError('Common60-frame normal tensor required')
    raw_frames=math.ceil(50/Q)*Q
    history_x,history_t=observed.x,observed.time_ps
    parts=[];history_times=[]
    before=model.forward_calls
    for begin in range(0,raw_frames,Q):
        visible=observed if begin==0 else history_with_coordinates(observed,history_x[-H:],history_t[:,-H:])
        if not bool((visible.time_ps[:,-1]<full_times[:,begin]).all()):raise ValueError('No future may enter history')
        condition=model.prepare_condition(visible,task='dynamic',route_task='joint')
        part=flow_heun(model,condition,full_times[:,begin:begin+Q],noise[begin:begin+Q],18)
        history_times.append((visible.time_ps-observed.time_ps[:,-1:]).detach().cpu().tolist());parts.append(part)
        history_x=torch.cat((history_x,part))[-H:]
        history_t=torch.cat((history_t,full_times[:,begin:begin+Q]),1)[:,-H:]
    full=torch.cat(parts);actual=model.forward_calls-before
    if actual!=35*len(parts):raise RuntimeError('Native field count changed')
    model.last_sampling_info=dict(mode=setting,protocol=setting,history_frames=H,block_size=Q,
        block_lengths=[Q]*len(parts),scored_frames=50,generated_frames=raw_frames,total_NFE=actual,
        source_noise_shape=list(noise.shape),source_noise_draws=1,
        source_epsilon_prefix50_sum=float(noise[:50].double().sum()),source_epsilon_prefix50_square_sum=float(noise[:50].double().square().sum()),
        source_policy='one_common60_normal_draw_physical_prefix_shared_across_all_HQ',
        query_time_ps=(query_time_ps-observed.time_ps[:,-1:]).detach().cpu().tolist(),full_query_time_ps=(full_times[:,:raw_frames]-observed.time_ps[:,-1:]).detach().cpu().tolist(),
        block_history_time_ps=history_times,history_encoded_frames=H*len(parts),atom_frame_evaluations=35*raw_frames*observed.atom_count,
        effective_evaluations_per_frame=[35]*raw_frames,effective_intervals_per_frame=[18]*raw_frames,
        condition_policy='latest_H_observed_or_generated_only_no_MD_reset_no_realign',last_scored_block_frames=50%Q or Q,
        tail_policy='full_blocks_then_score_first50;partial_scored_tail_excluded_from_full_block_trends')
    model.last_sampling_artifacts=dict(full_coordinates_A=full.detach(),full_query_time_ps=(full_times[:,:raw_frames]-observed.time_ps[:,-1:]).detach())
    return full[:50]
