"""CPU-only comparison of the five saved HQ arms at slot32 or slot256.

Different H/Q implies different losses, gradients and backward counts. Match
the complete target set, draws, data/aux cursors and actual optimizer exposure.
Missing arms are errors in the receipt, not a reason to skip available arms.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flow_hq_ratio_20261102'
PARENT=Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')
ARMS=dict(h4q4=(4,4),h4q8=(4,8),h8q4=(8,4),h8q8=(8,8),h12q12=(12,12))
# Reuse the frozen lightweight read/comparison helpers, without calling its CLI.
spec=importlib.util.spec_from_file_location('_hq_saved_state_helpers',ROOT/'tasks/flow-q4-align-20261030/check_pair.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
equal,finite,read_rows,json_safe=helpers.equal,helpers.finite,helpers.read_rows,helpers.json_safe

ROW_KEYS=('step','slot','optimizer_updates','optimized','excluded','task','sample','kernel_task','task_ordinal',
    'plan_index','matched_joint_plan_index','joint_ordinal','domain','system','replica','split',
    'record_path','original_sample_id','time_bucket_id','clock_sampling','topology_fingerprint',
    'joint_reference_index','joint_absolute_t0_ps','joint_target_native_frame_indices','joint_target_absolute_time_ps',
    'queries','query_frames','valid_atoms','source_epsilon_sum','source_epsilon_square_sum','progress_mean',
    'progress_values','learning_rate','last_optimizer_lr','route_task','task_adapter_mixing_weights',
    'aux_group_history','ligand_geometry_triggered','ligand_geometry_counts','ligand_geometry_events',
    'geometry_band_frames','kernel_counts','kernel_crossings')


def semantic_settings(settings):
    answer=dict(settings);answer['hq_ratio']=dict(settings['hq_ratio']);answer['hq_ratio']['arm']='<declared-H/Q>'
    return answer


def matched_exposure(exposure):
    return {k:v for k,v in exposure.items() if k!='RF_backward_calls'}


def check(args):
    began=time.monotonic();result=dict(status='failed',slot=args.step,arms={},comparisons={},errors=[],
        sources=dict(run=str(args.run),parent=str(args.parent)),
        expected_differences=['HQ arm','joint observed histories','chunk lengths','RF backward count',
                              'trained adapter weights','losses and gradients'],
        source_commit_compared=False)
    def require(ok,scope,message):
        if not ok:result['errors'].append(dict(scope=scope,message=message))
        return bool(ok)
    manifest=None;parent=None;excluded=None
    try:
        raw=(args.run/'data/manifest.json').read_bytes();manifest=json.loads(raw)
        require(manifest['status']=='complete' and manifest['all_candidates_processed'] and not manifest['failures'],
                'data','Preparation incomplete or failed')
        excluded={int(r['plan_index']) for r in manifest['exclusions']}
        valid={int(r['plan_index']) for r in manifest['rows']}
        require(not valid&excluded and valid|excluded==set(range(128)),'data','Candidate partition differs')
        digest=hashlib.sha256(raw).hexdigest()
        result['data']=dict(manifest=str(args.run/'data/manifest.json'),sha256=digest,valid_joint_windows=len(valid),
                           excluded_plan_indices=sorted(excluded),effective_updates=2*len(valid),target_frames=25*len(valid))
    except Exception as error:result['errors'].append(dict(scope='data',message=repr(error)))
    try:parent=torch.load(args.parent,map_location='cpu',weights_only=False)['model']
    except Exception as error:result['errors'].append(dict(scope='parent',message=repr(error)))
    states={};logs={}
    for arm,(h,q) in ARMS.items():
        path=args.run/arm/('checkpoint_step32.pt' if args.step==32 else 'coordinate_final.pt')
        detail=result['arms'][arm]=dict(status='unavailable',checkpoint=str(path),joint_history=h,joint_query=q)
        before=len(result['errors'])
        try:
            state=torch.load(path,map_location='cpu',weights_only=False);states[arm]=state
            rows=read_rows(args.run/arm/'updates.jsonl',args.step);logs[arm]=rows
            require(state['stage_schema']=='molvid.hq_ratio.v1' and state['slot']==state['step']==args.step,
                    arm,'Stage/slot differs')
            require(state['seed']==71 and state['settings']['updates']==256 and state['settings']['warmup_updates']==128,
                    arm,'Full schedule/seed differs')
            require(state['settings']['hq_ratio']['arm']==arm and state['settings']['task_adapter']==dict(mode='routed',bottleneck=64),
                    arm,'Declared HQ or routed adapter configuration differs')
            base={k:v for k,v in state['model'].items() if not k.startswith('task_adapters.')}
            unchanged=parent is not None and equal(base,parent)
            require(unchanged,arm,'Base differs from L4096 or parent unavailable')
            model_ok=finite(state['model']);optimizer_ok=finite(state['optimizer'])
            require(model_ok and optimizer_ok,arm,'Nonfinite model/AdamW')
            up_norms={k:float(v.double().square().sum())**.5 for k,v in state['model'].items()
                      if k.startswith('task_adapters.') and '.up_' in k}
            require(bool(up_norms) and sum(up_norms.values())>0,arm,'Zero output maps did not update')
            active=[r for r in rows if r['optimized']]
            adam_steps=sorted({int(v['step']) for v in state['optimizer']['state'].values()})
            require(state['optimizer_updates']==state['exposure']['updates']==len(active),arm,'Actual update count differs')
            require(adam_steps==[len(active)],arm,'AdamW step count differs')
            running_updates=0;backwards=0
            for row in rows:
                slot=row['step'];index=(slot-1)//2;kind='joint' if slot%2 else 'kernel'
                require(row['slot']==slot and row['kernel_task']==kind,arm,f'Wrong task/slot{slot}')
                if excluded is not None:
                    require(row['optimized']==(index not in excluded),arm,f'Exclusion pair mismatch at slot{slot}')
                if not row['optimized']:
                    require(row['loss'] is None and row['RF_loss'] is None and row['RF_backward_calls']==0
                            and not row['ligand_geometry_triggered'],arm,f'Excluded slot{slot} did optimization')
                else:
                    running_updates+=1
                    require(all(row[k] is not None for k in ('RF_loss','loss','grad_norm')) and
                            finite([row['RF_loss'],row['loss'],row['grad_norm']]),arm,f'Nonfinite update at slot{slot}')
                    require(row['route_task']==('joint' if kind=='joint' else 'q1'),arm,'Wrong adapter route')
                    if kind=='joint':
                        require(row['history']==h and row['chunk_query_frames']==q and row['queries']==list(range(12,36))
                                and row['query_frames']==24 and row['geometry_band_frames']==24,arm,'Joint window changed')
                        require(row['aux_group_history']==(4,8)[index%2],arm,'Joint auxiliary visit H changed')
                        chunks=row['chunks']
                        require(len(chunks)==24//q and row['RF_backward_calls']==24//q,arm,'Joint chunk count differs')
                        flattened=[i for chunk in chunks for i in chunk['target_indices']]
                        require(flattened==list(range(12,36)),arm,'24 targets omitted/repeated/reordered')
                        for offset,chunk in enumerate(chunks):
                            begin=12+offset*q
                            require(chunk['history_indices']==list(range(begin-h,begin)) and
                                    chunk['target_indices']==list(range(begin,begin+q)) and chunk['weight']==q/24,
                                    arm,'Teacher-history/weight differs')
                        if row['ligand_geometry_triggered']:
                            aux_chunks=row['ligand_geometry_chunks']
                            require(len(aux_chunks)==len(chunks) and all(c['weight']==q/24 and
                                    c['metrics']['geometry_band_frames']==24 for c in aux_chunks),
                                    arm,'Auxiliary chunk dose/band changed')
                    else:
                        require(row['history'] in (1,4) and row['queries']==[row['history']] and row['query_frames']==1
                                and row['chunk_query_frames']==1 and row['geometry_band_frames']==16,
                                arm,'Original Q1 window/band changed')
                    backwards+=row['RF_backward_calls']
                require(row['optimizer_updates']==running_updates,arm,f'Log update count wrong at slot{slot}')
            frames=sum(r['query_frames'] for r in active)
            atom_queries=sum(r['query_frames']*r['valid_atoms'] for r in active)
            require(state['exposure']['target_frames']==frames and state['exposure']['atom_query_pairs']==atom_queries
                    and state['exposure']['RF_backward_calls']==backwards,arm,'Exposure differs from actual active logs')
            require(state['kernel_tasks']['counts']==dict(joint=args.step//2,kernel=args.step//2)
                    and state['joint_windows']['cursor']==args.step//2,arm,'Original slot/data cursors differ')
            if manifest is not None and excluded is not None:
                require(state['joint_windows']['manifest_sha256']==digest,arm,'Completed data manifest changed')
                require(state['effective_updates']==2*len(valid) and state['effective_joint_windows']==len(valid),
                        arm,'Effective budget differs from valid data')
                expected=sum(i not in excluded for i in range(args.step//2))
                require(len(active)==2*expected and frames==25*expected,arm,'Matched prefix target count differs')
            initialization=state['initialization_report']
            require(initialization['parent_step']==4096 and initialization['newly_initialized_modules']==['task_adapters'],
                    arm,'Parent load report differs')
            detail.update(status='passed' if len(result['errors'])==before else 'failed',slot=state['slot'],
                optimizer_updates=state['optimizer_updates'],effective_updates=state['effective_updates'],
                exposure=state['exposure'],unchanged_parent=unchanged,model_finite=model_ok,optimizer_finite=optimizer_ok,
                adam_steps=adam_steps,adapter_output_weight_norms=up_norms,
                joint_aux_events=state['ligand_geometry']['events'],Q1_aux_events=state['kernel_ligand_geometry']['events'],
                learning_rates=[g['lr'] for g in state['optimizer']['param_groups']],last_optimizer_lr=state['last_optimizer_lr'],
                parent_checkpoint=initialization['parent_checkpoint'],source_commit=state.get('source_commit'))
        except Exception as error:
            detail.update(status='failed',error=repr(error));result['errors'].append(dict(scope=arm,message=repr(error)))
    available=[arm for arm in ARMS if arm in states and arm in logs]
    if available:
        reference=available[0];result['comparison_reference']=reference
        for arm in available[1:]:
            scope=f'{reference}__{arm}';checks={};result['comparisons'][scope]=checks
            try:
                x,y=states[reference],states[arm]
                diff=helpers.row_diff(logs[reference],logs[arm],ROW_KEYS)
                checks['rows']=diff;require(diff['equal'],scope,'Common identity/draw/aux/LR rows differ')
                for key in ('generator_rng','global_rng','torch_rng','cuda_rng','python_rng','numpy_rng',
                            'joint_windows','kernel_tasks','ligand_geometry','kernel_ligand_geometry'):
                    checks[key]=equal(x[key],y[key]);require(checks[key],scope,key+' differs')
                checks['exposure_except_backward_count']=equal(matched_exposure(x['exposure']),matched_exposure(y['exposure']))
                require(checks['exposure_except_backward_count'],scope,'Target/atom-query exposure differs')
                checks['settings_except_arm']=equal(semantic_settings(x['settings']),semantic_settings(y['settings']))
                require(checks['settings_except_arm'],scope,'Recipe differs beyond declared HQ')
                checks['optimizer_parameter_groups']=equal(x['optimizer']['param_groups'],y['optimizer']['param_groups'])
                require(checks['optimizer_parameter_groups'],scope,'Optimizer groups/schedule differ')
            except Exception as error:result['errors'].append(dict(scope=scope,message=repr(error)))
    result.update(status='passed' if not result['errors'] else 'failed',wall_seconds=time.monotonic()-began,
                  CUDA_initialized=torch.cuda.is_initialized())
    return json_safe(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,choices=(32,256),required=True);p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--output',type=Path)
    p.add_argument('--fail-on-error',action='store_true');args=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Existing container torch-ito with CUDA hidden required')
    output=args.output or args.run/f'check_pair{args.step}.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);result=check(args);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],slot=args.step,errors=result['errors'],output=str(output),
                         wall_seconds=result['wall_seconds'],CUDA_initialized=result['CUDA_initialized'])),flush=True)
    if args.fail_on_error and result['errors']:raise SystemExit(1)


if __name__=='__main__':main()
