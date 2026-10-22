"""CPU saved exposure/RNG comparison; no model forward."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/static_ref_20261022'
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(32,4096,8192),default=8192);args=p.parse_args()
filename='coordinate_final.pt' if args.step==8192 else f'checkpoint_step{args.step}.pt'
saved={arm:torch.load(RUN/arm/filename,map_location='cpu',weights_only=False) for arm in ('control','concat')}
a,b=saved.values();checks={key:equal(a[key],b[key]) for key in ('step','seed','exposure','mixture','generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng')}
checks['step']=a['step']==b['step']==args.step
checks['schedule8192']=a['settings']['updates']==b['settings']['updates']==8192
checks['staticQ1']=a['exposure']['static_examples']==a['exposure']['target_frames']==args.step and a['exposure']['dynamic_examples']==0
keys=('step','sample','system','visit','domain','history','queries','source_sample_id','source_store_index','source_frame_index','source_time_ps','progress_mean')
rows={arm:[json.loads(line) for line in (RUN/arm/'updates.jsonl').read_text().splitlines() if json.loads(line)['step']<=args.step] for arm in saved}
checks['training_input_rows']=len(rows['control'])==len(rows['concat'])==args.step and all(all(x.get(k)==y.get(k) for k in keys) for x,y in zip(rows['control'],rows['concat']))
checks['finite_weights']=all(bool(torch.isfinite(v).all()) for item in saved.values() for v in item['model'].values() if v.is_floating_point())
fusion={k:float(v[:,256:].norm()) for k,v in b['model'].items() if k.startswith('reference_fusion.')}
checks['reference_columns_updated']=len(fusion)==2 and all(v>0 for v in fusion.values())
result=dict(passed=all(checks.values()),checks=checks,step=args.step,exposure=a['exposure'],reference_column_norms=fusion)
(RUN/f'pair_check_step{args.step}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['passed']:sys.exit(1)
