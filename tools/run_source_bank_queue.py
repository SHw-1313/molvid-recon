"""Spawn fresh source processes after GPU release, avoiding stale imported code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from molvid.runtime import atomic_write_json


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.monotonic()
    end=args.root/'expanded_gcoord/execution_end.json'
    atomic_write_json(args.output/'status.json',dict(status='waiting_for_Gcoord_GPU_release',prerequisite=str(end)))
    while not end.exists():
        if time.monotonic()-started>86400: raise TimeoutError('24h prerequisite bound')
        time.sleep(20)
    if json.loads(end.read_text()).get('exception') is not None: raise RuntimeError('Gcoord prerequisite failed')
    jobs=[('source_contract_CUDA_gates',[sys.executable,'-m','pytest','-q',
        'tests/test_native_forecast_integration.py','tests/test_frame_source_cuda.py',
        'tests/test_static_backbone.py::test_d2_with_sampled_physical_full_and_streamed_backward',
        f'--junitxml={args.output}/cuda_gates.xml'],1200),
        ('source_bank_diagnostic',[sys.executable,'-m','tools.diagnose_source_scales',
         '--cache','/workspace/molvid-ca-detail-20261005/runs/ca_detail_20261005/cache',
         '--output',str(args.root/'source_bank_diagnostic')],7500)]
    for name,command,timeout in jobs:
        atomic_write_json(args.output/'status.json',dict(status='running',active=name,command=command))
        with (args.output/f'{name}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:
            atomic_write_json(args.output/'status.json',dict(status='failed',active=name,exit_code=result.returncode))
            raise RuntimeError(f'{name} failed; no downstream launch')
    atomic_write_json(args.output/'status.json',dict(status='completed',seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
