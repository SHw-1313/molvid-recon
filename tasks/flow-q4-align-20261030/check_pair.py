"""Read saved Q4 slots on CPU; no model calls, data resampling or GPU work.

Missing/failed arms remain explicit errors while other arms are still checked.
Default exit0 permits an independent evaluation chain to continue; consumers
must read status/errors. --fail-on-error is available for a deliberate stop.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/flow_q4_align_20261030'
OLD_RUN = Path('/workspace/MOLVID-FLOW-TASK-ROUTE-20261029/runs/flow_task_route_20261029')
PARENT = Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')
ARMS = ('shared_joint_only', 'shared_q4_q1', 'routed_q4_q1')
IDENTITY = ('step','task','sample','system','visit','domain','clock_sampling','time_bucket_id',
            'kernel_task','task_ordinal','history','queries','query_frames','replica','plan_index',
            't0_index','absolute_t0_ps','absolute_time_ps','target_lag_steps','target_lag_delta',
            'target_lag_ps','history_policy','history_ages_delta','band_source_sample_id')
DRAW = ('source_epsilon_sum','source_epsilon_square_sum','progress_mean','valid_atoms')
JOINT_OLD = IDENTITY + DRAW + ('learning_rate','route_task','geometry_band_frames',
            'ligand_geometry_triggered','ligand_geometry_counts','ligand_geometry_events')
PAIRED_ROWS = JOINT_OLD + ('slot','optimizer_updates','optimized','progress_values','last_optimizer_lr',
                         'kernel_counts','kernel_crossings','exposure','RF_backward_calls')


def equal(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and np.array_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x == y


def finite(x):
    if isinstance(x, torch.Tensor): return bool(torch.isfinite(x).all())
    if isinstance(x, np.ndarray): return bool(np.isfinite(x).all())
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)): return all(finite(v) for v in x)
    if isinstance(x, (float, np.floating)): return math.isfinite(x)
    return True


def json_safe(x):
    # A failed finite check must still produce readable structured evidence.
    if isinstance(x, dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return repr(x)
    return x


def local_cuda0(state):
    """Only the old8-visible/new1-visible comparison projects default CUDA0."""
    answer = dict(state)
    if 'cuda' in answer:
        if not answer['cuda']: raise ValueError('Missing local CUDA0 RNG state')
        answer['cuda'] = answer['cuda'][:1]
    return answer


def old_aux_view(state):
    answer = dict(state)
    answer['isolated_rng'] = local_cuda0(answer['isolated_rng'])
    return answer


def read_rows(path, step):
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if row['step'] <= step: rows.append(row)
            else: break
    if [r['step'] for r in rows] != list(range(1,step+1)):
        raise ValueError('Log prefix is missing, duplicated or reordered')
    return rows


def checkpoint_path(run, arm, step):
    # step32 is an immutable snapshot even while the arm continues training.
    return run/arm/('checkpoint_step32.pt' if step == 32 else 'coordinate_final.pt')


def row_diff(left, right, keys):
    differences = []
    for x,y in zip(left,right):
        for key in keys:
            if (key in x) != (key in y) or not equal(x.get(key),y.get(key)):
                differences.append(dict(slot=x.get('step'),key=key))
    if len(left) != len(right): differences.append(dict(key='number_of_rows'))
    return dict(equal=not differences,difference_count=len(differences),differences= differences[:100])


def semantic_settings(settings, *, remove_stage=False):
    value = copy.deepcopy(settings)
    if remove_stage:
        value.pop('q4_align', None)
    else:
        value['q4_align']['arm'] = '<declared-arm>'
        value['task_adapter']['mode'] = '<declared-mode>'
    return value


def check(args):
    began = time.monotonic()
    result = dict(status='failed',slot=args.step,errors=[],arms={},comparisons={},
                  sources=dict(run=str(args.run),old_run=str(args.old_run),parent=str(args.parent)),
                  interpretation=dict(step_is_slot=True,
                    A_deletes_Q1_supervision_and_updates=True,
                    trained_adapters_losses_gradients_need_not_match=True,
                    source_commit_not_a_pairing_constraint=True))
    def require(ok, scope, message):
        if not ok: result['errors'].append(dict(scope=scope,message=message))
        return bool(ok)
    def compare(name, x, y):
        ok = equal(x,y); result['comparisons'][name] = ok
        require(ok,name,'Mismatch');return ok
    parent = None; old = None; old_rows = None; plan = None
    try:
        parent = torch.load(args.parent,map_location='cpu',weights_only=False)['model']
    except Exception as error:
        result['errors'].append(dict(scope='parent',message=repr(error)))
    try:
        old = torch.load(checkpoint_path(args.old_run,'shared',args.step),map_location='cpu',weights_only=False)
        old_rows = read_rows(args.old_run/'shared/updates.jsonl', args.step)
        require(old['step'] == args.step,'old_shared','Checkpoint step mismatch')
    except Exception as error:
        result['errors'].append(dict(scope='old_shared',message=repr(error)))
    try:
        plan = json.loads((args.run/'exposure_plan.json').read_text())['slots'][:args.step]
        require([r['step'] for r in plan] == list(range(1,args.step+1)),'plan','Invalid frozen plan prefix')
    except Exception as error:
        result['errors'].append(dict(scope='plan',message=repr(error)))
    states = {}; logs = {}
    for arm in ARMS:
        path = checkpoint_path(args.run,arm,args.step)
        details = result['arms'][arm] = dict(status='unavailable',checkpoint=str(path))
        error_start = len(result['errors'])
        try:
            state = torch.load(path,map_location='cpu',weights_only=False); states[arm] = state
            rows = read_rows(args.run/arm/'updates.jsonl',args.step); logs[arm] = rows
            updates = args.step//2 if arm == ARMS[0] else args.step
            require(state['stage_schema']=='molvid.q4_align.v1' and state['step']==state['slot']==args.step,
                    arm,'Wrong stage/slot metadata')
            require(state['seed']==71 and state['settings']['updates']==512 and state['settings']['warmup_updates']==128,
                    arm,'Seed or full schedule changed')
            require(state['optimizer_updates']==state['exposure']['updates']==updates,arm,'Actual update count differs')
            expected_mode = 'routed' if arm == ARMS[2] else 'shared'
            require(state['settings']['task_adapter']==dict(mode=expected_mode,bottleneck=64)
                    and state['settings']['q4_align']['arm']==arm,arm,'Adapter/stage mode differs')
            initialization=state['initialization_report']
            require(initialization['newly_initialized_modules']==['task_adapters']
                    and initialization['parent_step']==4096,arm,'Unexpected parent/adapter initialization report')
            base = {k:v for k,v in state['model'].items() if not k.startswith('task_adapters.')}
            unchanged = parent is not None and equal(base,parent)
            require(unchanged,arm,'Frozen base differs or parent unavailable')
            model_finite = finite(state['model']); opt_finite = finite(state['optimizer'])
            require(model_finite and opt_finite,arm,'Nonfinite model/AdamW')
            norms = {k:float(v.double().square().sum())**.5 for k,v in state['model'].items()
                     if k.startswith('task_adapters.') and '.up_' in k}
            require(bool(norms) and sum(norms.values())>0,arm,'Zero-initialized adapter output weights never updated')
            adam_steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
            require(adam_steps==[updates],arm,'AdamW actual step counters differ')
            active = [r for r in rows if r['optimized']]
            require(len(active)==updates,arm,'Active log row count differs')
            require(all(r['slot']==r['step'] and r['optimized']==(arm!=ARMS[0] or r['kernel_task']=='joint')
                        for r in rows),arm,'Inactive/active slot mapping differs')
            for r in rows:
                count=(r['step']+1)//2 if arm==ARMS[0] else r['step']
                require(r['optimizer_updates']==count,arm,f'Update counter wrong at slot{r["step"]}')
                if r['optimized']:
                    require(all(r[k] is not None for k in ('RF_loss','loss','grad_norm')) and
                            finite([r['RF_loss'],r['loss'],r['grad_norm']]),arm,f'Nonfinite/missing log at slot{r["step"]}')
                    require(r['route_task']==('q1' if r['kernel_task']=='kernel' else 'joint'),arm,'Incorrect task route')
                else:
                    require(r['loss'] is None and r['RF_backward_calls']==0 and not r['ligand_geometry_triggered'],
                            arm,'Skipped Q1 did scientific work')
            frames = sum(r['query_frames'] for r in active)
            atom_queries = sum(r['query_frames']*r['valid_atoms'] for r in active)
            require(state['exposure']['target_frames']==frames and state['exposure']['atom_query_pairs']==atom_queries,
                    arm,'Saved target/atom-query exposure disagrees with active rows')
            require(state['kernel_tasks']['counts']==dict(joint=args.step//2,kernel=args.step//2),arm,'Data slot counts differ')
            if args.step==512:
                expected=(2560,6930776) if arm==ARMS[0] else (2816,7581179)
                require((frames,atom_queries)==expected,arm,'Final frozen exposure totals differ')
                require(state['ligand_geometry']['events']==15,arm,'Expected15 joint auxiliary events')
                require(state['kernel_ligand_geometry']['events']==(0 if arm==ARMS[0] else 15),arm,'Wrong Q1 auxiliary count')
            if arm==ARMS[0]:
                require(state['kernel_ligand_geometry']['counts']=={} and state['kernel_ligand_geometry']['events']==0,
                        arm,'Skipped Q1 consumed auxiliary visits')
            if plan is not None:
                # A's skipped slots expose identities, not a fictitious sampled epsilon.
                selected_plan=[r for r in plan if arm!=ARMS[0] or r['kernel_task']=='joint']
                selected_rows=active
                # The compact plan omits some optional identity fields present in logs.
                # Compare only its actual keys while requiring all declared draw fields.
                mismatches=[]
                for row,planned in zip(selected_rows,selected_plan):
                    for key,value in planned.items():
                        if key not in row or not equal(row[key],value):mismatches.append(dict(slot=row['step'],key=key))
                details['frozen_plan']=dict(equal=not mismatches,difference_count=len(mismatches),differences=mismatches[:100])
                require(not mismatches and len(selected_rows)==len(selected_plan),arm,'Frozen plan data/source/progress differs')
            details.update(status='passed' if len(result['errors'])==error_start else 'failed',
                slot=state['slot'],optimizer_updates=state['optimizer_updates'],rows=len(rows),
                unchanged_parent=unchanged,model_finite=model_finite,optimizer_finite=opt_finite,
                adam_step_counters=adam_steps,adapter_output_weight_norms=norms,exposure=state['exposure'],
                learning_rates=[g['lr'] for g in state['optimizer']['param_groups']],
                last_optimizer_lr=state['last_optimizer_lr'],
                joint_aux_events=state['ligand_geometry']['events'],kernel_aux_events=state['kernel_ligand_geometry']['events'],
                source_commit=state.get('source_commit'),declared_adapter_mode=expected_mode,
                parent_initialization=dict(checkpoint=initialization['parent_checkpoint'],
                    parent_step=initialization['parent_step'],inherited_parameters=initialization['inherited_parameters'],
                    newly_initialized_modules=initialization['newly_initialized_modules']))
        except Exception as error:
            details.update(status='failed',error=repr(error))
            result['errors'].append(dict(scope=arm,message=repr(error)))
    b,c=ARMS[1:]
    if b in states and c in states and b in logs and c in logs:
        try:
            x,y=states[b],states[c]
            diff=row_diff(logs[b],logs[c],PAIRED_ROWS);result['comparisons']['B_C_rows']=diff
            require(diff['equal'],'B_C_rows','Data/source/progress/exposure/aux row mismatch')
            for key in ('mixture','generator_rng','global_rng','torch_rng','cuda_rng','python_rng','numpy_rng',
                        'exposure','kernel_tasks','ligand_geometry','kernel_ligand_geometry'):
                compare('B_C_'+key,x[key],y[key])
            compare('B_C_optimizer_groups',x['optimizer']['param_groups'],y['optimizer']['param_groups'])
            compare('B_C_settings_except_declared_modes',semantic_settings(x['settings']),semantic_settings(y['settings']))
            result['comparisons']['B_C_weights']=dict(base_exact=equal(
                {k:v for k,v in x['model'].items() if not k.startswith('task_adapters.')},
                {k:v for k,v in y['model'].items() if not k.startswith('task_adapters.')}),
                trained_adapter_equality_not_required=True,
                modes=dict(B=x['settings']['task_adapter']['mode'],C=y['settings']['task_adapter']['mode']))
        except Exception as error: result['errors'].append(dict(scope='B_C',message=repr(error)))
    else: result['comparisons']['B_C']=dict(status='unavailable',reason='Missing checkpoint or log')
    a=ARMS[0]
    if a in states and a in logs and old is not None and old_rows is not None:
        try:
            x=states[a]
            left=[r for r in logs[a] if r['kernel_task']=='joint']
            right=[r for r in old_rows if r['kernel_task']=='joint']
            diff=row_diff(left,right,JOINT_OLD);result['comparisons']['A_oldT_joint_rows']=diff
            require(diff['equal'],'A_oldT_joint_rows','Joint same-slot data/source/progress/LR/aux mismatch')
            compare('A_oldT_mixture',x['mixture'],old['mixture'])
            compare('A_oldT_joint_generator',x['generator_rng'],old['generator_rng'])
            compare('A_oldT_joint_private_generator',x['kernel_tasks']['generators']['joint'],old['kernel_tasks']['generators']['joint'])
            compare('A_oldT_joint_default_rng_localcuda0',local_cuda0(x['kernel_tasks']['rng']['joint']),
                    local_cuda0(old['kernel_tasks']['rng']['joint']))
            compare('A_oldT_joint_aux_state_localcuda0',old_aux_view(x['ligand_geometry']),old_aux_view(old['ligand_geometry']))
            compare('A_oldT_settings_except_new_stage',semantic_settings(x['settings'],remove_stage=True),old['settings'])
            result['comparisons']['A_oldT_scope']=dict(joint_source_and_aux_checked=True,
                old_visible_cuda=len(old['kernel_tasks']['rng']['joint']['cuda']),
                new_visible_cuda=len(x['kernel_tasks']['rng']['joint']['cuda']),
                compared_default_cuda_indices=[0],skipped_Q1_rng_not_required_equal=True,
                optimizer_weights_and_moments_not_required_equal=True)
        except Exception as error: result['errors'].append(dict(scope='A_oldT',message=repr(error)))
    else: result['comparisons']['A_oldT']=dict(status='unavailable',reason='Missing checkpoint or log')
    result.update(status='passed' if not result['errors'] else 'failed',wall_seconds=time.monotonic()-began,
                  CUDA_initialized=torch.cuda.is_initialized())
    return json_safe(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=int,choices=(32,512),required=True)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--old-run',type=Path,default=OLD_RUN)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fail-on-error',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Existing container torch-ito and CUDA_VISIBLE_DEVICES empty required')
    output=args.output or args.run/f'check_pair{args.step}.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1)
    result=check(args)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],slot=args.step,errors=result['errors'],output=str(output),
                         wall_seconds=result['wall_seconds'],CUDA_initialized=result['CUDA_initialized'])),flush=True)
    if args.fail_on_error and result['errors']:raise SystemExit(1)


if __name__=='__main__':main()
