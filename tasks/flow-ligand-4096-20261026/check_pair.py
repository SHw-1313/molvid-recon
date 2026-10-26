"""CPU pairing of full-state C/L continuations; no new model calls or data reads."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from checks import PARENT,RUN,equal
from molvid.training.coordinate_state import continuation_learning_rate,validate_continuation

SHARED=('generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng','mixture','exposure')
ROW_KEYS=('step','task','sample','system','visit','domain','clock_sampling','time_bucket_id',
          'history','queries','progress_mean','valid_atoms','query_frames','learning_rate')


def finite(value):
    if isinstance(value,torch.Tensor): return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite(x) for x in value.values())
    if isinstance(value,(list,tuple)): return all(finite(x) for x in value)
    return True


def checkpoint_path(directory,step):
    preferred=directory/('coordinate_final.pt' if step==4096 else f'checkpoint_step{step}.pt')
    if preferred.exists(): return preferred
    return directory/'checkpoint_latest.pt'


def check(args,result):
    if torch.cuda.is_initialized(): raise RuntimeError('This check is CPU only; hide CUDA')
    torch.set_num_threads(1)
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    assert parent['step']==512
    checkpoints={};rows={};result['arms']={};differences=[]
    for arm in ('control','ligand'):
        directory=args.run/arm;path=checkpoint_path(directory,args.step)
        saved=torch.load(path,map_location='cpu',weights_only=False);checkpoints[arm]=saved
        if saved['step']!=args.step: raise ValueError(f'{arm}: saved step {saved["step"]}, expected {args.step}')
        validate_continuation(parent,saved['settings'],saved['seed'])
        if saved['settings']['updates']!=4096: raise ValueError('Full 4096 schedule was shortened')
        all_rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines() if line.strip()]
        selected=[row for row in all_rows if 512<row['step']<=args.step];rows[arm]=selected
        if [row['step'] for row in selected]!=list(range(513,args.step+1)):
            raise ValueError(f'{arm}: missing/duplicated new-stage update rows')
        missing={key for row in selected for key in ROW_KEYS if key not in row}
        if missing: raise ValueError(f'{arm}: missing pairing fields {sorted(missing)}')
        for row in selected:
            expected_lr=continuation_learning_rate(saved['settings'],row['step']-1)
            if not math.isclose(row['learning_rate'],expected_lr,rel_tol=1e-12,abs_tol=1e-15):
                differences.append(f'{arm}.step{row["step"]}.learning_rate')
            if row['history']!=(4,8)[(row['step']-1)%2] or row['queries']!=list(range(row['history'],16)):
                differences.append(f'{arm}.step{row["step"]}.history_queries')
            if not all(math.isfinite(row[key]) for key in ('loss','RF_loss','grad_norm','progress_mean')):
                differences.append(f'{arm}.step{row["step"]}.nonfinite_log')
        actual_lr=[group['lr'] for group in saved['optimizer']['param_groups']]
        expected_lr=continuation_learning_rate(saved['settings'],args.step-1)
        if any(not math.isclose(x,expected_lr,rel_tol=1e-12,abs_tol=1e-15) for x in actual_lr):
            differences.append(f'{arm}.optimizer_lr')
        optimizer_steps=sorted({int(value['step']) for value in saved['optimizer']['state'].values()})
        if optimizer_steps!=[args.step]: differences.append(f'{arm}.optimizer_steps')
        expected_exposure=dict(parent['exposure'])
        expected_exposure['updates']+=len(selected)
        expected_exposure['dynamic_examples']+=len(selected)
        expected_exposure['target_frames']+=sum(row['query_frames'] for row in selected)
        expected_exposure['atom_query_pairs']+=sum(row['valid_atoms']*row['query_frames'] for row in selected)
        if saved['exposure']!=expected_exposure: differences.append(f'{arm}.inherited_exposure')
        model_finite=finite(saved['model']);optimizer_finite=finite(saved['optimizer'])
        if not model_finite or not optimizer_finite: differences.append(f'{arm}.nonfinite_saved_state')
        initializer=saved['initialization_report']
        if initializer.get('kind')!='full_state_changed_objective_continuation' or initializer.get('parent_step')!=512:
            differences.append(f'{arm}.initialization_kind')
        if str(Path(initializer['parent_checkpoint']).resolve())!=str(args.parent.resolve()):
            differences.append(f'{arm}.parent_checkpoint')
        result['arms'][arm]=dict(checkpoint=str(path),step=args.step,new_updates=len(selected),
            optimizer_steps=optimizer_steps,optimizer_lr=actual_lr,expected_lr=expected_lr,
            model_finite=model_finite,optimizer_finite=optimizer_finite,exposure=saved['exposure'],
            history_counts=dict(Counter(str(row['history']) for row in selected)),
            domain_counts=dict(Counter(row['domain'] for row in selected)),
            latest_logged_step=all_rows[-1]['step'],task_elapsed_seconds=saved['elapsed_seconds'])
    c,l=checkpoints['control'],checkpoints['ligand']
    for key in SHARED:
        if key not in c or key not in l or not equal(c[key],l[key]): differences.append(f'shared.{key}')
    cs={k:v for k,v in c['settings'].items() if k!='ligand_geometry'}
    ls={k:v for k,v in l['settings'].items() if k!='ligand_geometry'}
    if cs!=ls or c['seed']!=l['seed']: differences.append('shared.recipe_or_seed')
    for cr,lr in zip(rows['control'],rows['ligand']):
        for key in ROW_KEYS:
            if not equal(cr[key],lr[key]): differences.append(f'update{cr["step"]}.{key}')
    if 'ligand_geometry' in c: differences.append('control_has_aux_state')
    if 'ligand_geometry' not in l: raise ValueError('L checkpoint lacks independent aux state')
    state=l['ligand_geometry'];counts={};triggers=0;applied=0;skipped=[]
    for row in rows['ligand']:
        expected=False
        if row['domain']=='complex':
            key=f'complex:H{row["history"]}';counts[key]=counts.get(key,0)+1
            expected=counts[key]%8==0
        if row.get('ligand_geometry_triggered') is not expected:
            differences.append(f'ligand.step{row["step"]}.trigger')
        if row.get('ligand_geometry_counts')!=counts:
            differences.append(f'ligand.step{row["step"]}.counts')
        triggers+=int(expected)
        if row.get('ligand_geometry_applied'):
            applied+=1
            if not expected or row.get('ligand_geometry_events')!=applied:
                differences.append(f'ligand.step{row["step"]}.applied_event')
        elif expected:
            skipped.append(dict(step=row['step'],reason=row.get('ligand_geometry_reason')))
    if state['counts']!=counts or state['events']!=applied:
        differences.append('ligand.saved_counts_events')
    if state['config']!=l['settings']['ligand_geometry']:
        differences.append('ligand.saved_config')
    if not all(key in state for key in ('generator_rng','isolated_rng')):
        differences.append('ligand.missing_private_rng')
    result['ligand_events']=dict(actual_visits=counts,scheduled=triggers,applied=applied,
        expected_from_actual_visits=sum(n//8 for n in counts.values()),skipped=skipped,
        nominal_final_estimate=224,
        note='Global H4/H8 alternation does not enforce equal domain×H counts. No extra events added to reach a nominal estimate.')
    result['initialization_evidence']=dict(parent=str(args.parent),step=512,
        source='control check restores exact saved model/AdamW/cursor/RNG; both saved continuation reports identify that parent',
        training_weights='Expected to diverge; not compared as equal')
    result['pairing_scope']=dict(row_fields=list(ROW_KEYS),checkpoint_states=list(SHARED),
        epsilon='Exact saved source generator plus equal draw shapes and RF code path; epsilon tensors themselves are not logged',
        progress='Each one-system update records the actual scalar progress_mean',
        statistics='No cross-dataset/clock/protocol scientific aggregates are produced here')
    result['differences']=differences;result['cuda_initialized']=torch.cuda.is_initialized()
    result['passed']=not differences and not result['cuda_initialized']
    if not result['passed']: raise AssertionError('Pair check failed: '+', '.join(differences[:30]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 512<args.step<=4096: raise ValueError('Use a saved global continuation step 513..4096')
    output=args.output or args.run/f'check_pair{args.step}.json'
    if output.exists(): raise FileExistsError(output)
    result=dict(step=args.step,passed=False,run=str(args.run));began=time.monotonic()
    try: check(args,result)
    except Exception as error:
        result['error']=repr(error);raise
    finally:
        result['cpu_wall_seconds']=time.monotonic()-began
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output),passed=result['passed'],cpu_wall_seconds=result['cpu_wall_seconds'])),flush=True)


if __name__=='__main__': main()
