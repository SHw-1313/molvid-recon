"""CPU read of two actual checkpoints and exposure logs; no model calls."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flow_task_route_20261029'
PARENT=Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')


def equal(x,y):
    if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
    if isinstance(x,np.ndarray): return isinstance(y,np.ndarray) and np.array_equal(x,y)
    if isinstance(x,dict): return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)): return len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x==y


def finite(value):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite(v) for v in value)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,choices=(32,512),default=32)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Run in existing torch-ito with CUDA_VISIBLE_DEVICES empty')
    output=a.output or a.run/f'check_pair{a.step}.json'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    begin=time.monotonic(); result=dict(step=a.step,status='failed',errors=[],arms={},paired={},row_differences=[])
    def require(flag,message):
        if not flag: result['errors'].append(message)
    try:
        states={};rows={}
        filename='checkpoint_step32.pt' if a.step==32 else 'coordinate_final.pt'
        parent=torch.load(PARENT,map_location='cpu',weights_only=False)['model']
        keys=('step','task','sample','system','visit','domain','clock_sampling','time_bucket_id',
              'kernel_task','task_ordinal','history','queries','learning_rate','route_task',
              'source_epsilon_sum','source_epsilon_square_sum','progress_mean','valid_atoms','query_frames',
              'replica','plan_index','t0_index','absolute_t0_ps','absolute_time_ps',
              'target_lag_steps','target_lag_delta','target_lag_ps','history_policy','history_ages_delta',
              'band_source_sample_id','geometry_band_frames','kernel_counts','kernel_crossings',
              'ligand_geometry_triggered','ligand_geometry_counts','ligand_geometry_events')
        for arm in ('shared','routed'):
            path=a.run/arm/filename
            state=torch.load(path,map_location='cpu',weights_only=False);states[arm]=state
            rows[arm]=[json.loads(line) for line in (a.run/arm/'updates.jsonl').read_text().splitlines() if line.strip()][:a.step]
            require(state['step']==a.step and state['settings']['updates']==512,arm+' step/schedule mismatch')
            require(len(rows[arm])==a.step,arm+' incomplete log exposure')
            base={k:v for k,v in state['model'].items() if not k.startswith('task_adapters.')}
            unchanged=equal(base,parent)
            require(unchanged,arm+' frozen parent changed')
            model_finite=finite(state['model']);optimizer_finite=finite(state['optimizer'])
            require(model_finite and optimizer_finite,arm+' nonfinite model/AdamW')
            mode=state['settings']['task_adapter']['mode']
            require(mode==arm,arm+' routing recipe mismatch')
            sums={name:float(value.double().square().sum()) for name,value in state['model'].items()
                  if name.startswith('task_adapters.') and '.up_' in name}
            require(sum(sums.values())>0,arm+' all adapter outputs remain zero')
            counts={kind:sum(r['kernel_task']==kind for r in rows[arm]) for kind in ('joint','kernel')}
            require(counts==dict(joint=a.step//2,kernel=a.step//2),arm+' joint/Q1 update counts differ')
            require(all(r['route_task']==('q1' if r['kernel_task']=='kernel' else 'joint') for r in rows[arm]),arm+' mislabeled route')
            require(state['initialization_report']['newly_initialized_modules']==['task_adapters'],arm+' parent load report differs')
            result['arms'][arm]=dict(checkpoint=str(path),step=state['step'],rows=len(rows[arm]),task_updates=counts,
                unchanged_L4096_base=unchanged,model_finite=model_finite,optimizer_finite=optimizer_finite,
                learning_rates=[g['lr'] for g in state['optimizer']['param_groups']],exposure=state['exposure'],
                adapter_output_weight_norm=sum(sums.values())**.5,
                joint_aux_events=state['ligand_geometry']['events'],kernel_aux_events=state['kernel_ligand_geometry']['events'])
        for i,(x,y) in enumerate(zip(rows['shared'],rows['routed'])):
            for key in keys:
                if x.get(key)!=y.get(key): result['row_differences'].append(dict(step=i+1,key=key))
        require(not result['row_differences'],'actual data/source/progress/aux exposure differs')
        x,y=states['shared'],states['routed']
        for key in ('mixture','generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng','exposure',
                    'kernel_tasks','ligand_geometry','kernel_ligand_geometry'):
            flag=equal(x[key],y[key]);result['paired'][key]=flag;require(flag,key+' differs')
        result['paired']['optimizer_parameter_groups']=equal(x['optimizer']['param_groups'],y['optimizer']['param_groups'])
        require(result['paired']['optimizer_parameter_groups'],'optimizer parameter groups/LR differ')
        ax={k:v for k,v in x['settings'].items() if k!='task_adapter'}
        ay={k:v for k,v in y['settings'].items() if k!='task_adapter'}
        require(equal(ax,ay),'recipe differs beyond task adapter mode')
        require(x['settings']['task_adapter']['bottleneck']==y['settings']['task_adapter']['bottleneck']==64,'capacity mismatch')
        result['shared_initialization_evidence']=dict(parent=str(PARENT),same_base_frozen=True,
            tested_constructor_and_warmstart='tests/test_task_route.py; actual CUDA checks_execution.json',
            adapter_outputs_initially_zero=True,trained_adapter_weights_required_equal=False)
        result['status']='passed' if not result['errors'] else 'failed'
    except Exception as error:
        result['errors'].append(repr(error));raise
    finally:
        result.update(wall_seconds=time.monotonic()-begin,CUDA_initialized=torch.cuda.is_initialized())
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
    if result['errors']: raise SystemExit(1)


if __name__=='__main__': main()
