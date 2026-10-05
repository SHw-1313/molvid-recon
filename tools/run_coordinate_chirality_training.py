"""Own-GPU paired continuation, real active-aux resume gate and fixed-final eval."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint
from tools.run_coordinate_campaign import assert_same


def qualify_pair(left, right):
    """The only within-seed recipe difference must be the auxiliary weight."""
    a,b=deepcopy(left),deepcopy(right)
    wa=a['coordinate_training'].pop('handedness_weight')
    wb=b['coordinate_training'].pop('handedness_weight')
    if wa != 0 or not 0 < wb <= 1 or a != b:
        raise ValueError('paired control/chirality recipes differ beyond registered weight')
    c=a['coordinate_training']
    if (c['handedness_every'],c['handedness_steps'],c['total_updates'],a['coordinate_schedule_offset']) != (8,16,2048,2048):
        raise ValueError('registered cadence/solver/budget/schedule required')


def qualified_profiles(status):
    if (status.get('status') != 'completed' or status.get('optimizer_updates_discarded') != 2
            or sorted(p['seed'] for p in status.get('profiles',[])) != [71,72]
            or any(p['profile_updates'] != 1 for p in status['profiles'])):
        raise ValueError('both real largest-query profiles required')


def audit_final(checkpoint, entry, runtime):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require_final_coordinate_checkpoint(saved)
    contract=saved['contract']
    if (contract['input_config_sha256'] != entry['configuration_sha256']
            or contract['seed'] != entry['seed'] or contract['schedule_offset'] != 2048
            or contract['coordinate_initialization']['sha256'] != entry['parent_sha256']
            or contract['prefix2048_sha256'] != entry['exposure_sha256']):
        raise ValueError('final checkpoint contract differs from frozen matched run')
    for name,digest in contract['runtime_source_sha256'].items():
        if sha256_file(runtime/name) != digest:
            raise ValueError(f'active training runtime changed: {name}')
    steps={int(state['step']) for state in saved['optimizer']['state'].values()}
    if steps != {2048} or not all(torch.isfinite(t).all() for t in saved['model'].values() if t.is_floating_point()):
        raise ValueError('incomplete optimizer cursor or nonfinite final model')
    return dict(status='completed',checkpoint_sha256=sha256_file(checkpoint),cursor=2048,
        optimizer_states=len(saved['optimizer']['state']),runtime_hashes=len(contract['runtime_source_sha256']),
        floating_model_all_finite=True,model_calls=0)


def run(args):
    # Do not pre-create the destination needed by the not-yet-run freezer.
    queue=args.root.parent/f'{args.root.name}_training_queue_v1'
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True)
    prerequisite=args.root/'profile_queue/status.json'
    atomic_write_json(queue/'status.json',dict(status='waiting_for_own_GPU_profiles',prerequisite=str(prerequisite)))
    started=time.monotonic()
    try:
        while True:
            if prerequisite.exists():
                status=json.loads(prerequisite.read_text())
                if status.get('status') == 'completed':
                    qualified_profiles(status); break
                if status.get('status') == 'failed':
                    raise RuntimeError('profile prerequisite failed; formal training not started')
            if time.monotonic()-started > 21600:
                raise TimeoutError('six-hour prerequisite budget')
            time.sleep(20)
        manifest=json.loads((args.root/'manifest.json').read_text())
        entries={r['name']:r for r in manifest['runs']}
        order=['control_71','chirality_71','chirality_72','control_72']
        if set(entries) != set(order): raise ValueError('exact two-seed paired campaign required')
        if sha256_file(args.root/'exposure.json') != manifest['exposure_sha256']:
            raise ValueError('frozen exposure metadata changed')
        for seed in (71,72):
            configs=[]
            for arm in ('control','chirality'):
                entry=entries[f'{arm}_{seed}']; path=Path(entry['configuration'])
                if sha256_file(path) != entry['configuration_sha256']:
                    raise ValueError('frozen config changed')
                configs.append(json.loads(path.read_text()))
            qualify_pair(*configs)
        freeze_run_identity(queue,inputs=[args.root/'manifest.json',args.root/'exposure.json',prerequisite],
            purpose='Paired coordinate handedness model training and fixed-final assessment',
            protocol=dict(order=order,updates_per_arm=2048,resume_gate='25 versus16→25 includes auxiliary step24',
                extra_resume_verification_updates_per_arm=9,test_opened=False,default_promotion=False,
                training_exposure='next2048 ordered batches',checkpoint_selection='fixed_final_only'))

        def execute(name,module,arguments,*,timeout=22000,cwd=None,cpu=False):
            import os
            command=[sys.executable,'-u','-m',module,*map(str,arguments)]
            atomic_write_json(queue/'status.json',dict(status='running',active=name,command=command))
            environment=dict(os.environ)
            if cpu: environment['CUDA_VISIBLE_DEVICES']=''
            with (queue/f'{name}.log').open('x') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                    timeout=timeout,cwd=cwd,env=environment)

        completed={}
        for name in order:
            entry=entries[name]; root=Path(entry['output'])
            if root.exists(): raise FileExistsError(root)
            if shutil.disk_usage(args.root).free < 9*2**30:
                raise RuntimeError('9GiB free-space reserve before new paired arm')
            root.mkdir()
            config=Path(entry['configuration'])
            if sha256_file(config) != entry['configuration_sha256']:
                raise ValueError('frozen config changed before arm start')
            base=['--config',config,'--seed',entry['seed']]
            execute(f'{name}_prefix25','tools.train_coordinate_flow',base+['--output',root/'pilot25','--max-steps',25])
            checkpoint16=root/'pilot25/coordinate_step_0016.pt'
            execute(f'{name}_resume16to25','tools.train_coordinate_flow',base+[
                '--output',root/'resume25','--max-steps',25,'--resume',checkpoint16,
                '--resume-sha256',sha256_file(checkpoint16)])
            left=root/'pilot25/coordinate_step_0025.pt'; right=root/'resume25/coordinate_step_0025.pt'
            assert_same(torch.load(left,map_location='cpu',weights_only=False),
                        torch.load(right,map_location='cpu',weights_only=False))
            atomic_write_json(root/'real_resume_parity.json',dict(exact=True,
                checkpoint_sha256=[sha256_file(left),sha256_file(right)],
                auxiliary_step24_in_resume=entry['arm']=='chirality',extra_verification_updates=9,
                scope='whole model,optimizer,CPU/CUDA/flow/Python/NumPy RNG,cursor,contract'))
            execute(f'{name}_formal','tools.train_coordinate_flow',base+['--output',root/'formal',
                '--resume',right,'--resume-sha256',sha256_file(right)])
            checkpoint=root/'formal/coordinate_step_2048.pt'
            audit=audit_final(checkpoint,entry,Path(__file__).resolve().parents[1]/'molvid')
            atomic_write_json(root/'final_audit.json',audit)
            # This marker also permits recoverable host-side relocation of ONLY
            # completed prefix/replay checkpoints; final remains in place.
            atomic_write_json(root/'training_complete.json',dict(status='completed',updates=2048,
                checkpoint=str(checkpoint),checkpoint_sha256=audit['checkpoint_sha256'],
                generation_quality_evaluated=False))
            generated=root/'expanded48'; variant=f'coordinate_{name}'
            execute(f'{name}_evaluate','tools.evaluate_expanded_native',[
                '--cache',args.main/'development48_v2','--output',generated,'--arm','coordinate',
                '--variant',variant,'--checkpoint',checkpoint])
            summary=json.loads((generated/'summary.json').read_text())
            if summary['status'] != 'completed' or summary['systems'] != 48 or summary['rows'] != 192:
                raise ValueError('full192 successful draws required; no partial-score selection')
            execute(f'{name}_chemistry','tools.audit_cached_stereochemistry',[
                '--cache',args.main/'development48_v2','--generated',generated,
                '--output',root/'stereochemistry'],timeout=1800,cpu=True)
            completed[name]=generated
            seed=entry['seed']
            if all(f'{arm}_{seed}' in completed for arm in ('control','chirality')):
                parent=args.coordinate/f'full192_v3/seed{seed}/expanded48'
                groups=[args.main/'expanded_a1',args.main/'expanded_pvb',parent,
                        completed[f'control_{seed}'],completed[f'chirality_{seed}']]
                execute(f'paired_{seed}','tools.compare_expanded_native',[
                    '--generated',*groups,'--exposure',args.main/'external_overlap_v3/summary.json',
                    '--pair',f'coordinate_chirality_{seed}:coordinate_control_{seed}',
                    '--pair',f'coordinate_control_{seed}:coordinate_{seed}',
                    '--pair',f'coordinate_chirality_{seed}:coordinate_{seed}',
                    '--output',args.root/f'paired_{seed}'],timeout=1800,cpu=True)
                execute(f'ordered_{seed}','tools.audit_ordered_motion',[
                    '--cache',args.main/'development48_v2','--generated',*groups,'--lag-ps',100,
                    '--output',(args.root/f'ordered_{seed}').resolve()],
                    timeout=1800,cwd=args.ordered_root,cpu=True)
        atomic_write_json(queue/'status.json',dict(status='completed',default_promotion=False,
            formal_updates=8192,extra_resume_verification_updates=36,generated_draws=768,
            qualification='development48 native100ps; not independent, complex or thermodynamic qualification'))
    except BaseException as error:
        atomic_write_json(queue/'status.json',dict(status='failed',error=repr(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--main',type=Path,default=Path('/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005'))
    p.add_argument('--coordinate',type=Path,default=Path('/workspace/molvid-coordinate-flow-20261005/runs/coordinate_flow_20261005'))
    p.add_argument('--ordered-root',type=Path,default=Path('/workspace/molvid-native-time-response-20261005'))
    args=p.parse_args()
    for name in ('root','main','coordinate','ordered_root'):
        setattr(args,name,getattr(args,name).resolve())
    run(args)
