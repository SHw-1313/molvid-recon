"""One targeted real-TRAIN CUDA exercise for both coordinate-state recipes.

No scientific efficacy verdict; no optimizer updates or validation selection.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.runtime import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    configs={s:json.loads((args.recipes/f'{s}.json').read_text()) for s in ('flow','edm')}
    rows=json.loads((args.recipes/'rows.json').read_text())
    mixture=BridgeMixture(configs['flow']['data'],71)
    selected=[]
    # Packed records must share a native sampling bucket. Two distinct full
    # complexes exercise protein/ligand identities AND cross-sample isolation
    # without relabeling physical clocks or weakening the collator contract.
    systems=set()
    for row in sorted((x for x in rows if x['domain']=='complex' and x['time_bucket_id']=='dt_80ps'),
                      key=lambda x:x['atoms']):
        if row['system'] in systems:
            continue
        candidates=mixture.groups['complex'][row['system']][row['time_bucket_id']]
        for store,index in candidates:
            if store._clip_index_fields[index][0]==row['sample']:
                selected.append(store[index]);systems.add(row['system']);break
        if len(selected)==2:
            break
    if len(selected)!=2:raise RuntimeError('Two complete TRAIN systems required')
    batch=collate_clip_records(selected).to('cuda')
    observed=slice_clip_frames(batch,0,4)
    snapshot=observed.x.clone()
    target=batch.x[4:]
    times=batch.time_ps[:,4:]
    output={}
    began=time.monotonic()
    for scheme,config in configs.items():
        torch.manual_seed(71)
        model=CoordinateStateDenoiser(config).cuda().train()
        rng=torch.Generator(device='cuda').manual_seed(9001)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            condition=model.prepare_condition(observed)
            loss,metrics=coordinate_state_objective(model,condition,target,times,generator=rng)
        loss.backward()
        gradients={}
        for name,module in [('current_geometry',model.spatial),('temporal',model.temporal),
                            ('token_hierarchy',model.tokens),('coordinate_head',model.head)]:
            values=[p.grad for p in module.parameters() if p.grad is not None]
            norm=sum(float(g.float().square().sum()) for g in values)**.5
            if not values or not all(bool(torch.isfinite(g).all()) for g in values) or norm==0:
                raise RuntimeError(f'{scheme} invalid gradient path {name}')
            gradients[name]=norm
        model.zero_grad(set_to_none=True);model.eval()
        del condition,loss
        with torch.no_grad():
            condition=model.prepare_condition(observed)
            state=condition.reference_A[None]+torch.randn(target.shape,device='cuda',generator=rng)
            progress=torch.full((2,),.4 if scheme=='flow' else model.sigma_data,device='cuda')
            prediction=model(state,condition=condition,query_time_ps=times,progress=progress)
            rotation,_=torch.linalg.qr(torch.randn(3,3,device='cuda',generator=rng))
            if torch.det(rotation)<0:rotation[:,0]*=-1
            shift=state.new_tensor([3.,-4.,2.])
            rotated_observed=replace(observed,x=observed.x@rotation+shift,bpos=observed.bpos@rotation+shift)
            rotated_condition=model.prepare_condition(rotated_observed)
            rotated=model(state@rotation+shift,condition=rotated_condition,query_time_ps=times,progress=progress)
            equivariance=float((rotated-prediction@rotation).abs().max())
            if not torch.allclose(rotated,prediction@rotation,atol=2e-4,rtol=2e-4):
                raise RuntimeError(f'{scheme} joint equivariance error {equivariance}')
            # The other packed molecular system cannot affect sample0.
            other=observed.abid==1
            shifted=state.clone();shifted[:,other]+=7.
            shifted_observed=replace(observed,x=observed.x+other[None,:,None]*7.,
                                     bpos=observed.bpos+other[None,:,None]*7.)
            separated_condition=model.prepare_condition(shifted_observed)
            separated=model(shifted,condition=separated_condition,query_time_ps=times,progress=progress)
            isolation=float((separated[:,~other]-prediction[:,~other]).abs().max())
            repeated=model(state,condition=condition,query_time_ps=times,progress=progress)
            repeat_error=float((repeated[:,~other]-prediction[:,~other]).abs().max())
            # CUDA scatter reductions gave 4e-5 A repeat variation on an
            # identical FP32 input. Use the same absolute tolerance as the
            # joint-equivariance exercise, and preserve repeat error evidence.
            if repeat_error>2e-4 or not torch.allclose(separated[:,~other],prediction[:,~other],atol=2e-4,rtol=2e-5):
                raise RuntimeError(f'{scheme}: isolation max={isolation}, identical-input repeat max={repeat_error}, '
                                   f'prediction max={float(prediction[:,~other].abs().max())}; '
                                   'distinguish numerical reduction order from actual cross-system interaction')
            calls,graphs=model.forward_calls,model.geometry_calls
            sample=model.sample(condition,times,generator=rng)
            nfe=model.forward_calls-calls;graph_calls=model.geometry_calls-graphs
            if nfe!=35 or graph_calls!=35*len(target) or sample.shape!=target.shape or not bool(torch.isfinite(sample).all()):
                raise RuntimeError('Incomplete/nonfinite coordinate-only sampling or missing current geometry')
            if not torch.equal(observed.x,snapshot):raise RuntimeError('Observed coordinates modified')
        output[scheme]=dict(gradient_norms=gradients,loss_observation=metrics,equivariance_max_A=equivariance,
            packed_isolation_max_A=isolation,identical_input_repeat_max_A=repeat_error,
            packed_isolation_atol_A=2e-4,complete_sampling_NFE=nfe,current_geometry_rebuilds=graph_calls,
            generated_shape=list(sample.shape),parameters=sum(p.numel() for p in model.parameters()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),observed_unchanged=True)
        del model,condition,sample,prediction,rotated,rotated_condition,separated_condition
        torch.cuda.empty_cache()
    mixture.close()
    atomic_write_json(args.output,dict(status='completed',real_TRAIN_systems=[r['sample_id'] for r in selected],
        optimizer_updates=0,scientific_efficacy=False,arms=output,elapsed_seconds=time.monotonic()-began))


if __name__=='__main__':main()
