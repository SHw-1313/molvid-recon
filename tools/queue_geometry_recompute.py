"""Qualification only, after the already submitted complete GPU7 input gate."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from tools.qualify_dynamic_input import input_record, read, RECIPE, RELEASE
from tools.qualify_query_geometry import process
from tools.queue_dynamic_input_qualification import GPU, occupied, resumed_pair

BASELINE = Path('/workspace/molvid-dynamic-input-qualification-20261006/runs/dynamic_input_qualification_20261006/queue_v1')
CASES = [('representative_control', 'representative', 4, False),
         ('representative_recompute', 'representative', 4, True),
         ('maximum_H4', 'maximum', 4, True), ('maximum_H8', 'maximum', 8, True)]


def completed_baseline(path):
    state = read(path/'status.json')
    if state.get('status') not in ('completed', 'needs_remediation'):
        raise ValueError('all original maximum-input cases must have terminated')
    expected = {f'{mode}_H{history}' for mode in ('frozen_query','trainable_query') for history in (4,8)}
    rows = state.get('cases', [])
    if len(rows) != 4 or {r['case'] for r in rows} != expected:
        raise ValueError('four terminal original cases required')
    for row in rows:
        if row['status'] not in ('completed','failed') or row != read(path/row['case']/'comparison.json'):
            raise ValueError('baseline terminal evidence mismatch')
    return state


def require_cuda_tests(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    if len(cases) != 5 or any(list(case) for case in cases):
        # No skipped CUDA tests, xfail, errors or failure allowed.
        raise ValueError('all five actual CUDA numerical gates must pass without skips')


def paired_execution(control, candidate):
    if control.get('recompute_geometry') is not False or candidate.get('recompute_geometry') is not True:
        raise ValueError('explicit off/on execution pair required')
    for key in ('model_sha256', 'optimizer_sha256', 'rng_sha256', 'sample_id', 'atoms',
                'history_frames', 'query_frames', 'cursor_end'):
        if control[key] != candidate[key]:
            raise ValueError('recomputation altered actual two-update result: '+key)
    return dict(exact_model_optimizer_rng_digest=True,
        control_peak_bytes=max(row['peak_allocated_bytes'] for row in control['updates']),
        recompute_peak_bytes=max(row['peak_allocated_bytes'] for row in candidate['updates']),
        control_seconds=control['seconds'],recompute_seconds=candidate['seconds'],
        scope='same real1578-atom input, weights, FP32, H4/Q12, seed and two updates; not efficacy')


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    owner = process(args.owner_pid)
    if owner is None or 'tools.queue_dynamic_input_qualification' not in owner[1]:
        raise ValueError('live original qualification controller required at submission')
    _, inputs = input_record()
    recipe = read(RECIPE); release = read(RELEASE)
    inputs += [RECIPE,RELEASE,Path(recipe['coordinate_initialization']['checkpoint']),
        Path(release['configuration']['path']),Path(release['selected_checkpoint']['path']),BASELINE/'run_identity.json']
    identity = freeze_run_identity(args.output, inputs=inputs,
        purpose='Explicit default-off geometry activation recomputation qualification; no efficacy claim',
        protocol=dict(owner_pid=args.owner_pid,owner_identity=owner,prerequisite=str(BASELINE),
            gpu_uuid=GPU,cases=CASES,expected_discarded_real_protein_updates=12,
            formal_updates=0,protein_forecasts=0,sealed_data_opened=False,
            cuda_tests=5,representative_pair_exact_model_optimizer_rng_required=True,
            no_crop=True,no_temporal_shortening=True,precision='float32'))
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def unchanged():
        for name,digest in identity['source_files_sha256'].items():
            if sha256_file(root/name) != digest: raise ValueError('queued source changed')
        for name,digest in identity['input_sha256'].items():
            if sha256_file(Path(name)) != digest: raise ValueError('queued input changed')
    def execute(command, log):
        unchanged()
        if occupied(): raise RuntimeError('registered GPU7 still occupied')
        if shutil.disk_usage(args.output).free < 9*2**30: raise RuntimeError('9GiB launch reserve required')
        atomic_write_json(args.output/'status.json',dict(status='running',command=command))
        with log.open('x') as handle:
            subprocess.run(command,cwd=root,env=environment,stdout=handle,stderr=subprocess.STDOUT,
                           check=True,timeout=7200)
        deadline = time.monotonic()+120
        while occupied():
            if time.monotonic() > deadline: raise RuntimeError('GPU not released after child')
            time.sleep(5)
    try:
        deadline = time.monotonic()+86400
        while True:
            state = read(BASELINE/'status.json'); current = process(args.owner_pid)
            if current is not None and current != owner: raise RuntimeError('owner PID reused')
            if state['status'] in ('completed','needs_remediation'):
                completed_baseline(BASELINE)
                if not occupied(): break
            elif state['status'] == 'failed' or current is None:
                raise RuntimeError('original maximum-input queue failed/disappeared before terminal cases')
            if time.monotonic() > deadline: raise TimeoutError('24h prerequisite deadline')
            atomic_write_json(args.output/'status.json',dict(status='waiting',prerequisite=str(BASELINE),
                owner_pid=args.owner_pid,owner_live=current is not None,prerequisite_state=state,
                actual_updates=0,formal_updates=0,protein_forecasts=0))
            time.sleep(30)
        unchanged()
        atomic_write_json(args.output/'prerequisite_accepted.json',dict(
            terminal_status_sha256=sha256_file(BASELINE/'status.json'),
            state=completed_baseline(BASELINE)))
        junit = args.output/'cuda_tests.xml'
        execute([sys.executable,'-m','pytest','-q','--color=no','-p','no:cacheprovider',
                 'tests/test_geometry_recompute.py','-k','cuda',f'--junitxml={junit}'],args.output/'cuda_tests.log')
        require_cuda_tests(junit)
        results = {}
        for name,size,history,recompute in CASES:
            destination = args.output/name; destination.mkdir()
            base = [sys.executable,'-u','-m','tools.qualify_dynamic_input','--mode','trainable_query',
                    '--history',str(history),'--input-size',size]
            if recompute: base += ['--recompute-geometry']
            for phase in ('uninterrupted','resume'):
                command = base+['--output',str(destination/phase)]
                if phase == 'resume':
                    prefix = destination/'uninterrupted/prefix1.pt'
                    command += ['--resume',str(prefix),'--resume-sha256',sha256_file(prefix)]
                execute(command,destination/f'{phase}.log')
            left,right = [read(destination/phase/'summary.json') for phase in ('uninterrupted','resume')]
            resumed_pair(left,right,'trainable_query',history,atoms=1578 if size=='representative' else 17002)
            if left['recompute_geometry'] != recompute or right['recompute_geometry'] != recompute:
                raise ValueError('execution flag differs from registered case')
            results[name] = left
            atomic_write_json(destination/'comparison.json',dict(status='completed',actual_updates=3,
                full_state_resume_digest_exact=True,recompute_geometry=recompute))
            if name == 'representative_recompute':
                pair = paired_execution(results['representative_control'],left)
                atomic_write_json(args.output/'representative_execution_pair.json',pair)
        unchanged()
        atomic_write_json(args.output/'status.json',dict(status='completed',actual_updates=12,
            formal_updates=0,protein_forecasts=0,efficacy_tested=False,all_four_cases_qualified=True,
            results={name:row['model_sha256'] for name,row in results.items()}))
    except BaseException as error:
        previous = read(args.output/'status.json') if (args.output/'status.json').exists() else None
        atomic_write_json(args.output/'status.json',dict(status='failed',error=repr(error),
            last_stage=previous,actual_updates=None,formal_updates=0,protein_forecasts=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-pid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); args.output = args.output.resolve()
    run(args)
