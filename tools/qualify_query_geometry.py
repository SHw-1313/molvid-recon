"""Wait for the owned solver lane, then run bounded CUDA qualification only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19], (root / 'cmdline').read_bytes().decode().replace('\0', ' ')
    except FileNotFoundError:
        return None


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    original = process(args.wait_pid)
    if original is None or 'tools.evaluate_expanded_native' not in original[1] or 'expanded_v1/parent64' not in original[1]:
        raise ValueError('expected live parent64 solver must be identified before submission')
    identity = freeze_run_identity(args.output, inputs=[args.solver_output / 'protocol.json'],
        purpose='Deferred GQ numerical qualification, not protein efficacy',
        protocol=dict(wait_pid=args.wait_pid, process_identity=original, gpu_uuid=args.gpu_uuid,
                      formal_updates=0, protein_forecasts=0, test_opened=False,
                      tests=['tests/test_query_geometry.py', 'tests/test_coordinate_flow.py']))
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    try:
        while True:
            if time.monotonic() - started > 21600:
                raise TimeoutError('six-hour owned-GPU wait budget')
            current = process(args.wait_pid)
            if current is not None and current != original:
                raise RuntimeError('solver PID reused')
            applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                    '--format=csv,noheader,nounits'], text=True)
            occupied = any(line.split(',')[0].strip() == args.gpu_uuid for line in applications.splitlines())
            if current is None and not occupied:
                break
            atomic_write_json(args.output / 'status.json', dict(status='waiting',
                solver_live=current is not None, gpu_occupied=occupied))
            time.sleep(30)
        summary = json.loads((args.solver_output / 'summary.json').read_text())
        if (summary['status'] != 'completed' or summary['rows'] != 192 or summary['systems'] != 48
                or summary['failures'] or summary['coordinate_field_calls'] != 12288):
            raise ValueError('full registered solver arm must complete before qualification')
        for name, digest in identity['source_files_sha256'].items():
            if sha256_file(root / name) != digest:
                raise ValueError(f'queued GQ source changed: {name}')
        for name, digest in identity['input_sha256'].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f'queued solver protocol changed: {name}')
        atomic_write_json(args.output / 'status.json', dict(status='cuda_tests_running'))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (args.output / 'pytest.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs',
                'tests/test_query_geometry.py', 'tests/test_coordinate_flow.py',
                '--junitxml', str(args.output / 'pytest.xml')], cwd=root, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        from xml.etree import ElementTree
        suites = list(ElementTree.parse(args.output / 'pytest.xml').getroot().iter('testsuite'))
        if not suites or any(int(s.get('skipped', '0')) or int(s.get('errors', '0'))
                             or int(s.get('failures', '0')) for s in suites):
            raise ValueError('all CUDA qualification cases must execute and pass; no skipped CUDA success')
        atomic_write_json(args.output / 'status.json', dict(status='completed',
            tests=sum(int(s.get('tests', '0')) for s in suites), formal_updates=0,
            protein_forecasts=0, qualification='synthetic numerical only; real profile and prefix gates pending'))
    except BaseException as error:
        atomic_write_json(args.output / 'status.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int, required=True)
    p.add_argument('--gpu-uuid', choices=['GPU-3b8ea60f-89b6-7112-c41c-97a516b30b5d'], required=True)
    p.add_argument('--solver-output', type=Path, default=Path('/workspace/molvid-coordinate-solver-audit-20261005/runs/coordinate_solver_audit_20261005/expanded_v1/parent64'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    run(args)
