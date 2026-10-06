"""One explicitly assigned GPU: actual qualification, then fixed-final train/eval.

Four independent invocations may run concurrently; no worker takes other GPUs.
Failed gates never submit formal training. Every subprocess has closed logs.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from molvid.training.coordinate_budget import QUERY_LONG_BUDGET
from tools.run_coordinate_campaign import assert_same


def read(path):
    return json.loads(path.read_text())


def compare_default(old, new):
    if old['cursor'] != 16 or new['cursor'] != 16:
        raise ValueError('actual step16 controls required')
    for name in ('model', 'optimizer', 'rng', 'cursor'):
        assert_same(old[name], new[name], name)
    a, b = deepcopy(old['contract']), deepcopy(new['contract'])
    for contract in (a, b):
        contract.pop('runtime_source_sha256')
    assert_same(a, b, 'scientific_contract')


def compare_resume(a, b, depth):
    for value in (a, b):
        if (value['cursor'] != 17 or value['contract']['coordinate_budget_experiment'] != QUERY_LONG_BUDGET
                or value['contract']['model'].get('dit_depth', 4) != depth):
            raise ValueError('registered step17 query/depth contract required')
    assert_same(a, b)


def require_cuda(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    if len(cases) != 6 or any(c.find(t) is not None for c in cases for t in ('skipped', 'failure', 'error')):
        raise ValueError('six actual depth/query/long-policy CUDA tests required, no skips')


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    manifest = args.recipes / 'manifest.json'
    row, = [r for r in read(manifest)['runs'] if r['name'] == args.arm]
    require_cuda(args.cuda_xml)
    inputs = [manifest, args.cuda_xml]
    for key in ('configuration', 'legacy_configuration', 'legacy_prefix'):
        path = Path(row[key])
        if sha256_file(path) != row[key + '_sha256']:
            raise ValueError('registered input changed')
        inputs.append(path)
    identity = freeze_run_identity(args.output, inputs=inputs,
        purpose='Independent query/depth8192 GPU lane with exact qualification',
        protocol=dict(arm=row, gpu_uuid=args.gpu_uuid, qualification_updates_planned=35,
            formal_total_updates=8192, retained_prefix_updates=17, new_formal_updates=8175,
            forecasts_planned=192, test_opened=False, independent_test=False, default_promotion=False))

    def unchanged():
        for name, digest in identity['source_files_sha256'].items():
            if sha256_file(root / name) != digest:
                raise ValueError('frozen lane source changed')
        for name, digest in identity['input_sha256'].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError('pinned lane input changed')

    def execute(label, arguments, gpu=True, timeout=48000):
        unchanged()
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if gpu and any(line.split(',')[0].strip() == args.gpu_uuid for line in apps.splitlines()):
            raise RuntimeError('assigned GPU is occupied; never co-schedule')
        if shutil.disk_usage(args.output).free < 9 * 2**30:
            raise RuntimeError('9GiB disk reserve unavailable')
        command = [sys.executable, '-u', *map(str, arguments)]
        atomic_write_json(args.output / 'status.json', dict(status='running', active=label, command=command,
            gpu_uuid=args.gpu_uuid, pid=os.getpid()))
        with (args.output / f'{label}.log').open('x') as log:
            subprocess.run(command, cwd=root, env=dict(os.environ,
                CUDA_VISIBLE_DEVICES=args.gpu_uuid if gpu else '',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)

    try:
        base = ['-m', 'tools.train_coordinate_flow', '--seed', row['seed']]
        default = args.output / 'default16'
        execute('default16', base + ['--config', row['legacy_configuration'], '--output', default, '--max-steps', 16])
        paths = [Path(row['legacy_prefix']), default / 'coordinate_step_0016.pt']
        compare_default(*[torch.load(p, map_location='cpu', weights_only=False) for p in paths])
        atomic_write_json(args.output / 'default_parity.json', dict(exact=True, cursor=16,
            checkpoint_paths=list(map(str, paths)), checkpoint_sha256=list(map(sha256_file, paths))))
        base += ['--config', row['configuration']]
        profile = args.output / 'profile'
        execute('profile', base + ['--output', profile, '--max-steps', 1, '--profile-largest'])
        contract = read(profile / 'contract.json')
        if (read(profile / 'summary.json')['updates_this_execution'] != 1
                or contract['largest_query_atom_frame_profile_only'] is not True
                or contract['selected_schedule_sha256'] != row['selected_schedule_sha256']):
            raise ValueError('actual largest-query8192 profile missing')
        pilot, resume = args.output / 'pilot17', args.output / 'resume17'
        execute('pilot17', base + ['--output', pilot, '--max-steps', 17])
        prefix = pilot / 'coordinate_step_0016.pt'
        execute('resume17', base + ['--output', resume, '--max-steps', 17,
            '--resume', prefix, '--resume-sha256', sha256_file(prefix)])
        paths = [p / 'coordinate_step_0017.pt' for p in (pilot, resume)]
        states = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
        compare_resume(*states, row['depth'])
        if states[0]['contract']['selected_schedule_sha256'] != row['selected_schedule_sha256']:
            raise ValueError('registered full8192 exposure mismatch')
        del states
        atomic_write_json(args.output / 'qualification.json', dict(status='completed', exact_resume=True,
            actual_real_optimizer_updates=35, discarded_default_updates=16, discarded_profile_updates=1,
            retained_prefix_updates=17, extra_resume_updates=1, forecasts=0,
            checkpoint_paths=list(map(str, paths)), checkpoint_sha256=list(map(sha256_file, paths))))
        execute('formal', base + ['--output', args.output / 'formal', '--resume', paths[1],
            '--resume-sha256', sha256_file(paths[1])])
        checkpoint = args.output / 'formal/coordinate_step_8192.pt'
        saved, before = [torch.load(p, map_location='cpu', weights_only=False) for p in (checkpoint, paths[1])]
        if saved['cursor'] != 8192 or saved['contract'] != before['contract']:
            raise ValueError('final budget/contract mismatch')
        if saved['model'].keys() != before['model'].keys():
            raise ValueError('model inventory changed')
        teacher = 0; query_changed = 0
        for name, value in saved['model'].items():
            if value.shape != before['model'][name].shape or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('nonfinite or changed model shape')
            if name.startswith('target_teacher.'):
                teacher += 1
                if not torch.equal(value, before['model'][name]):
                    raise ValueError('frozen teacher changed')
            if name.startswith('query_encoder.') and not torch.equal(value, before['model'][name]):
                query_changed += 1
        if not teacher or not query_changed or {float(s['step']) for s in saved['optimizer']['state'].values()} != {8192.}:
            raise ValueError('incomplete optimizer or no query adaptation')
        atomic_write_json(args.output / 'final_audit.json', dict(status='completed', cursor=8192,
            checkpoint_sha256=sha256_file(checkpoint), frozen_teacher_exact=True,
            query_tensors_changed=query_changed, finite_model=True))
        del saved, before
        cache = Path('/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2')
        generated = args.output / 'expanded48'
        execute('evaluation', ['-m', 'tools.evaluate_expanded_native', '--cache', cache,
            '--output', generated, '--arm', 'coordinate', '--variant', args.arm, '--checkpoint', checkpoint])
        result = read(generated / 'summary.json')
        if result['status'] != 'completed' or result['rows'] != 192 or result['systems'] != 48 or result['failures']:
            raise ValueError('full192 successful forecasts required')
        execute('chemistry', ['-m', 'tools.audit_cached_stereochemistry', '--cache', cache,
            '--generated', generated, '--output', args.output / 'stereochemistry'], gpu=False, timeout=1800)
        unchanged()
        atomic_write_json(args.output / 'status.json', dict(status='completed', updates=8192,
            new_updates=8175, retained_prefix_updates=17, forecasts=192, default_promotion=False,
            comparison='pending paired system-level audit; no efficacy claim from completion'))
    except BaseException as error:
        atomic_write_json(args.output / 'status.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('recipes', 'cuda-xml', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--arm', choices=['query4_71', 'query8_71', 'query4_72', 'query8_72'], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    for key in ('recipes', 'cuda_xml', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    run(args)
