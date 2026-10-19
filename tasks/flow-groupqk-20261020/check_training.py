"""Small save/resume and same-data/noise verification for authorized Flow routing arms."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from molvid.runtime import atomic_write_json


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def finite(a):
    if isinstance(a,torch.Tensor):return bool(torch.isfinite(a).all())
    if isinstance(a,dict):return all(finite(v) for v in a.values())
    if isinstance(a,(tuple,list)):return all(finite(v) for v in a)
    if isinstance(a,float):return math.isfinite(a)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs/flow_groupqk_20261020'))
    p.add_argument('--arm',choices=['atom','group'],required=True)
    p.add_argument('--step',type=int,choices=[32,512],required=True)
    a=p.parse_args();torch.set_num_threads(2)
    name='coordinate_final.pt' if a.step==512 else 'checkpoint_latest.pt'
    saved=torch.load(a.run/a.arm/name,map_location='cpu',weights_only=False)
    rows=[json.loads(line) for line in (a.run/a.arm/'updates.jsonl').read_text().splitlines()]
    settings=saved['settings']
    checks=dict(step=saved['step']==a.step,seed=saved['seed']==71,unchanged_512=settings['updates']==512,
        scheme=settings['scheme']=='flow',mode=settings.get('query_qk_mode','atom')==a.arm,
        all_steps=[r['step'] for r in rows]==list(range(1,a.step+1)),
        finite=finite(saved['model']) and finite(saved['optimizer']) and finite(rows),
        optimizer_step=all(int(v['step'])==a.step for v in saved['optimizer']['state'].values()),
        optimizer_lr=math.isclose(saved['optimizer']['param_groups'][0]['lr'],2.5e-5 if a.step==32 else 1e-5,rel_tol=1e-12),
        exposure=saved['exposure']['updates']==a.step and saved['exposure']['static_examples']==0)
    states={}
    if a.arm!='atom':
        control=torch.load(a.run/'atom/coordinate_final.pt',map_location='cpu',weights_only=False)
        crows=[json.loads(line) for line in (a.run/'atom/updates.jsonl').read_text().splitlines()][:a.step]
        common=dict(settings);common.pop('query_qk_mode')
        ccommon=dict(control['settings']);ccommon.pop('query_qk_mode')
        keys=('step','task','sample','system','visit','domain','clock_sampling','time_bucket_id','history','queries','valid_atoms','query_frames','progress_mean')
        checks.update(same_original_config=equal(common,ccommon),same_parent=saved['initialization_report']['parent_checkpoint']==control['initialization_report']['parent_checkpoint'],
            same_data_noise=len(crows)==len(rows) and all(all(r[k]==c[k] for k in keys) for r,c in zip(rows,crows)))
        if a.step==512:
            states={key:equal(saved[key],control[key]) for key in ('generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng','mixture','exposure')}
            checks['same_final_random_access_states']=all(states.values())
    query_updates={}
    if a.step==512:
        parent=torch.load(saved['initialization_report']['parent_checkpoint'],map_location='cpu',weights_only=False)
        targets=('q.weight','k.weight','value_h.weight','value_h.bias','value_v.weight',
                 'out_h.weight','out_h.bias','out_v.weight')
        # Weight decay alone also changes nonzero parent tensors; record its
        # nominal-only reference so a parameter difference is not mislabeled
        # as proof of useful temporal learning. No model forward is executed.
        decay=1.
        for step in range(a.step):
            warmup=settings['warmup_updates']
            fraction=max(0.,(step-warmup)/max(1,settings['updates']-warmup-1))
            factor=(step+1)/warmup if step<warmup else .1+.45*(1+math.cos(math.pi*fraction))
            decay*=1-settings['learning_rate']*factor*settings['weight_decay']
        for suffix in targets:
            parameter_name='temporal.query_attention.'+suffix
            before=parent['model'][parameter_name].float();after=saved['model'][parameter_name].float()
            delta=after-before
            query_updates[suffix]=dict(parent_norm=float(before.norm()),final_norm=float(after.norm()),
                delta_norm=float(delta.norm()),nominal_weight_decay_only_residual_norm=float((after-before*decay).norm()),
                changed_elements=int(torch.count_nonzero(delta)),finite=bool(torch.isfinite(after).all()))
        checks['query_qk_value_out_parameters_updated']=all(r['finite'] and r['delta_norm']>0 for r in query_updates.values())
    result=dict(status='passed' if all(checks.values()) else 'failed',arm=a.arm,step=a.step,checks=checks,shared_state_checks=states,
        checkpoint=str(a.run/a.arm/name),query_parameter_updates=query_updates)
    atomic_write_json(a.run/'checks'/f'{a.arm}_{a.step}.json',result)
    print(json.dumps(result),flush=True)
    if not all(checks.values()):raise RuntimeError('Flow matching/save-resume check failed')

if __name__=='__main__':main()
