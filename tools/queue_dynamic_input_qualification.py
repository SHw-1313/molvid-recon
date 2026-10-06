"""Own GPU7 after the full long8_72 lane, then four largest-input cases."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from tools.qualify_dynamic_input import input_record, RECIPE, RELEASE, read
from tools.qualify_query_geometry import process

GPU = 'GPU-76667725-c9c0-d162-6ee7-50625ecb785e'
LANE = Path('/workspace/molvid-coordinate-budget8192-20261006/runs/coordinate_budget8192_20261006/formal_v1/long8_72')
CASES = [(mode,history) for mode in ('frozen_query','trainable_query') for history in (4,8)]


def completed_lane(path):
    state = read(path/'status.json')
    if state.get('status') != 'completed' or state.get('updates') != 8192 or state.get('forecasts') != 192:
        raise ValueError('full long8_72 lane, not a GPU phase gap, required')
    generated = read(path/'expanded48/summary.json')
    if (generated.get('status') != 'completed' or generated.get('rows') != 192
            or generated.get('systems') != 48 or generated.get('failures') != []):
        raise ValueError('full long8_72 forecasts required')
    if read(path/'stereochemistry/summary.json').get('status') != 'completed':
        raise ValueError('chemistry phase required')
    audit = read(path/'final_audit.json')
    if audit.get('status') != 'completed' or audit.get('cursor') != 8192:
        raise ValueError('final model audit required')
    if sha256_file(path/'formal/coordinate_step_8192.pt') != audit['checkpoint_sha256']:
        raise ValueError('completed long checkpoint changed')
    evaluated = read(path/'expanded48/run_identity.json')['input_sha256']
    if evaluated.get(str(path/'formal/coordinate_step_8192.pt')) != audit['checkpoint_sha256']:
        raise ValueError('long forecast used a different checkpoint')
    trained = read(path/'formal/summary.json')
    if (trained.get('status') != 'completed' or trained.get('cursor_end') != 8192
            or trained.get('cursor_start') != 17 or trained.get('updates_this_execution') != 8175):
        raise ValueError('full long training cursor required')


def resumed_pair(left, right, mode, history, *, atoms=17002):
    for row,updates in ((left,2),(right,1)):
        if (row.get('status') != 'completed' or row.get('actual_updates') != updates
                or row.get('cursor_end') != 2 or row.get('optimizer_steps') != [2]
                or row.get('atoms') != atoms or row.get('history_frames') != history
                or row.get('query_frames') != 16-history or row.get('model_mode') != mode
                or row.get('finite_model_optimizer_and_gradients') is not True):
            raise ValueError('actual unshortened maximum-input updates required')
        if row['teacher_before_sha256'] != row['teacher_after_sha256']:
            raise ValueError('frozen teacher changed')
    if left.get('cursor_start') != 0 or right.get('cursor_start') != 1:
        raise ValueError('expected fresh two updates and cold resumed second update')
    for key in ('full_state_sha256','model_sha256','optimizer_sha256','rng_sha256','sample_id'):
        if left[key] != right[key]: raise ValueError('cold-resume state digest differs:'+key)


def occupied():
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    return any(row.split(',')[0].strip() == GPU for row in rows.splitlines())


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    owner = process(args.owner_pid) if args.owner_pid is not None else None
    if args.owner_pid is not None:
        if owner is None or 'tools.run_budget8192_campaign' not in owner[1]:
            raise ValueError('live long-budget controller must be identified at submission')
    else:
        completed_lane(LANE)
    record, inputs = input_record()
    del record
    recipe = read(RECIPE)
    inputs += [RECIPE,Path(recipe['coordinate_initialization']['checkpoint']),LANE.parent/'run_identity.json']
    release = read(RELEASE)
    inputs += [RELEASE,Path(release['configuration']['path']),Path(release['selected_checkpoint']['path'])]
    identity = freeze_run_identity(args.output,inputs=inputs,
        purpose='Deferred maximum TRAIN input CUDA qualification after registered full GPU7 lane',
        protocol=dict(owner_pid=args.owner_pid,owner_identity=owner,gpu_uuid=GPU,prerequisite=str(LANE),
            cases=CASES,expected_actual_discarded_updates=12,formal_training_submitted=False,
            protein_forecasts=0,test_opened=False,all_case_failures_retained=True,
            resume_proof='canonical full-state SHA256 including optimizer and all RNG fields'))
    root = Path(__file__).resolve().parents[1]
    def unchanged():
        for name,digest in identity['source_files_sha256'].items():
            if sha256_file(root/name) != digest: raise ValueError('queued source changed')
        for name,digest in identity['input_sha256'].items():
            if sha256_file(Path(name)) != digest: raise ValueError('queued input changed')
    try:
        started = time.monotonic()
        while True:
            current = process(args.owner_pid) if args.owner_pid is not None else None
            state = read(LANE/'status.json')
            if current is not None and current != owner: raise RuntimeError('owner PID reused')
            if state['status'] == 'completed':
                completed_lane(LANE)
                if not occupied(): break
            elif state['status'] == 'failed' or current is None:
                raise RuntimeError('long8_72 failed or owner missing before full completion')
            if time.monotonic()-started > 86400: raise TimeoutError('24h full-lane queue deadline')
            atomic_write_json(args.output/'status.json',dict(status='waiting',owner_pid=args.owner_pid,
                owner_live=current is not None,prerequisite=str(LANE),prerequisite_state=state,
                gpu_uuid=GPU,actual_updates=0))
            time.sleep(30)
        unchanged()
        atomic_write_json(args.output/'prerequisite_accepted.json',dict(status='completed',
            files={name:sha256_file(LANE/name) for name in ('status.json','final_audit.json',
                'expanded48/summary.json','stereochemistry/summary.json')}))
        results = []
        for mode,history in CASES:
            unchanged()
            if occupied(): raise RuntimeError('dedicated GPU7 occupied before case')
            if shutil.disk_usage(args.output).free < 10*2**30: raise RuntimeError('10GiB launch reserve required')
            name = f'{mode}_H{history}'
            destination = args.output/name; destination.mkdir()
            base = [sys.executable,'-u','-m','tools.qualify_dynamic_input','--mode',mode,'--history',str(history)]
            environment = dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            failure = None
            for phase in ('uninterrupted','resume'):
                command = base+['--output',str(destination/phase)]
                if phase == 'resume':
                    prefix = destination/'uninterrupted/prefix1.pt'
                    command += ['--resume',str(prefix),'--resume-sha256',sha256_file(prefix)]
                atomic_write_json(args.output/'status.json',dict(status='running',case=name,phase=phase,command=command))
                try:
                    with (destination/f'{phase}.log').open('x') as log:
                        subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,
                            check=True,timeout=7200)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    failure = repr(error); break
            if failure is None:
                left,right = [read(destination/phase/'summary.json') for phase in ('uninterrupted','resume')]
                try:
                    resumed_pair(left,right,mode,history)
                except ValueError as error:
                    failure = repr(error)
            evidence = dict(case=name,status='failed' if failure else 'completed',error=failure,
                actual_updates=3 if failure is None else None,
                recorded_update_lower_bound=sum(len(read(destination/p/'updates.json'))
                    for p in ('uninterrupted','resume') if (destination/p/'updates.json').exists()),
                resume_all_state_digest_exact=failure is None)
            atomic_write_json(destination/'comparison.json',evidence)
            results.append(evidence); atomic_write_json(args.output/'cases.json',results)
            # Child exit/OOM is terminal; no retries or data reduction. GPU must
            # really be released before trying the next registered diagnostic.
            deadline = time.monotonic()+120
            while occupied():
                if time.monotonic() > deadline: raise RuntimeError('GPU not released after qualification child')
                time.sleep(5)
        unchanged()
        success = all(row['status']=='completed' for row in results)
        atomic_write_json(args.output/'status.json',dict(status='completed' if success else 'needs_remediation',
            cases=results,all_four_cases_qualified=success,actual_updates=12 if success else None,
            formal_updates=0,protein_forecasts=0,efficacy_tested=False,
            scope='single largest ordinary16-frame input only; not512 loader/packed/distributed qualification'))
    except BaseException as error:
        atomic_write_json(args.output/'status.json',dict(status='failed',error=repr(error))); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner-pid',type=int,help='live controller PID; omit only after verified full lane completion')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve()
    run(args)
