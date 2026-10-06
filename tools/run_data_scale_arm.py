"""One released GPU: real control/input qualification, fixed8192, then full48.

No architecture or objective change. A failed gate never starts formal training.
All prefixes and failures remain; only a verified candidate17 prefix is retained.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file
from molvid.training.data_scale_input import read,validate_reference,require_cuda_gate
from tools.run_query_depth_budget import compare_default,compare_resume
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint


GPUS={71:'GPU-3b8ea60f-89b6-7112-c41c-97a516b30b5d',72:'GPU-f0bea51c-913e-fe29-1cdd-4cde2adb3ff3'}


def check_input_audit(summary,seed):
    row=summary['seeds'][str(seed)]
    if (summary.get('status')!='completed' or summary.get('model_calls')!=0
            or summary.get('training_updates')!=0 or summary.get('protected_payloads_read')!=0
            or row.get('systems')!=512 or row.get('records')!=184320
            or row.get('selected_updates')!=8192 or row.get('real_batches_decoded',0)<9):
        raise ValueError('completed real full512 input audit required')
    return row


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];gpu=GPUS[args.seed]
    qualified=check_input_audit(read(args.input_audit/'summary.json'),args.seed)
    raw=deepcopy(read(args.input_audit/f'candidate_input_only_{args.seed}.json'))
    settings=raw['coordinate_data_scale']
    settings.update(cuda_gate=str(args.cuda_gate),cuda_gate_status_sha256=sha256_file(args.cuda_gate/'status.json'),
        cuda_gate_identity_sha256=sha256_file(args.cuda_gate/'run_identity.json'))
    validate_reference(raw);require_cuda_gate(settings)
    reference=Path(settings['reference_recipe'])
    old=Path('/workspace/molvid-query-depth-budget-20261006/runs/query_depth_budget_20261006/campaign_v1')/f'query4_{args.seed}'
    prefix=old/'pilot17/coordinate_step_0016.pt'
    if read(old/'qualification.json').get('exact_resume') is not True:
        raise ValueError('original query control was not CUDA/resume qualified')
    # GPU5 is released only after its entire native-depth inference AND the
    # strict CPU recovery. GPU4 additionally finished the max-system gate.
    native=Path('/workspace/molvid-coordinate-native1ns-20261006/runs/coordinate_native1ns_20261006')
    recovery=read(native/'cpu_analysis_recovery_v1/summary.json')
    if recovery.get('status')!='completed' or recovery.get('comparisons')!=6:
        raise ValueError('previous native GPU lanes not fully recovered/completed')
    owner=read(native/f'campaign_v1/seed{args.seed}/run_identity.json')['protocol']
    if owner['gpu_uuid']!=gpu:raise ValueError('GPU ownership mismatch')
    inputs=[args.input_audit/'summary.json',args.input_audit/'run_identity.json',
        args.input_audit/f'candidate_input_only_{args.seed}.json',args.cuda_gate/'status.json',
        args.cuda_gate/'run_identity.json',reference,prefix,old/'qualification.json',
        native/'cpu_analysis_recovery_v1/summary.json',native/f'campaign_v1/seed{args.seed}/run_identity.json']
    for depth in (4,8):
        folder=native/f'campaign_v1/seed{args.seed}/long{depth}_{args.seed}/native1ns'
        summary=read(folder/'summary.json')
        if (summary.get('status')!='completed' or summary.get('rows')!=192 or summary.get('systems')!=48
                or read(folder/'execution_end.json')['exception'] is not None):
            raise ValueError('previous native lane still incomplete')
        inputs += [folder/'summary.json',folder/'execution_end.json']
    config=args.output/'candidate.json';atomic_write_json(config,raw);inputs.append(config)
    identity=freeze_run_identity(args.output,inputs=inputs,purpose='Data-only TRAIN512 versus fixed TRAIN192 query4 control',
        protocol=dict(seed=args.seed,gpu_uuid=gpu,depth=4,formal_updates=8192,qualification_updates=35,
            retained_prefix_updates=17,forecasts=192,test_opened=False,independent_test=False,
            comparison='identical recipe except data; matched temporal/packing-count calendar; unequal atom exposure/compute',
            default_promotion=False))
    def execute(label,arguments,*,gpu_job=True,timeout=48000):
        for path,digest in identity['input_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('registered input changed')
        for path,digest in identity['source_files_sha256'].items():
            if sha256_file(root/path)!=digest:raise ValueError('live training source changed')
        if gpu_job:
            active=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip()
            if active:raise RuntimeError('assigned GPU occupied; never co-schedule')
        if shutil.disk_usage(args.output).free<9*2**30:raise RuntimeError('9GiB disk floor reached')
        command=[sys.executable,'-u',*map(str,arguments)]
        atomic_write_json(args.output/'status.json',dict(status='running',active=label,command=command))
        with (args.output/f'{label}.log').open('x') as log:
            subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu if gpu_job else '',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        base=['-m','tools.train_coordinate_flow','--seed',args.seed]
        execute('default16',base+['--config',reference,'--output',args.output/'default16','--max-steps',16])
        generated=args.output/'default16/coordinate_step_0016.pt'
        compare_default(*[torch.load(p,map_location='cpu',weights_only=False) for p in (prefix,generated)])
        atomic_write_json(args.output/'default_parity.json',dict(exact=True,cursor=16,
            scope='model,Adam,RNG,contract except runtime hashes',checkpoint_sha256=[sha256_file(p) for p in (prefix,generated)]))
        base+=['--config',config]
        execute('profile',base+['--output',args.output/'profile','--max-steps',1,'--profile-largest'])
        profile=read(args.output/'profile/contract.json')
        if (profile['selected_schedule_sha256']!=qualified['schedule_sha256']
                or profile['largest_query_atom_frame_profile_only'] is not True
                or read(args.output/'profile/summary.json')['updates_this_execution']!=1):
            raise ValueError('actual selected-largest full-data profile missing')
        execute('pilot17',base+['--output',args.output/'pilot17','--max-steps',17])
        prefix16=args.output/'pilot17/coordinate_step_0016.pt'
        execute('resume17',base+['--output',args.output/'resume17','--max-steps',17,
            '--resume',prefix16,'--resume-sha256',sha256_file(prefix16)])
        paths=[args.output/name/'coordinate_step_0017.pt' for name in ('pilot17','resume17')]
        states=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        compare_resume(*states,4)
        if states[0]['contract']['selected_schedule_sha256']!=qualified['schedule_sha256']:
            raise ValueError('qualified expanded calendar not used by trainer')
        del states
        atomic_write_json(args.output/'qualification.json',dict(status='completed',actual_updates=35,
            default_updates_discarded=16,profile_updates_discarded=1,retained_prefix_updates=17,
            extra_resume_updates=1,exact_default=True,exact_resume=True,
            checkpoint_sha256=[sha256_file(p) for p in paths]))
        execute('formal',base+['--output',args.output/'formal','--resume',paths[1],
            '--resume-sha256',sha256_file(paths[1])])
        checkpoint=args.output/'formal/coordinate_step_8192.pt'
        saved,before=[torch.load(p,map_location='cpu',weights_only=False) for p in (checkpoint,paths[1])]
        require_final_coordinate_checkpoint(saved)
        if saved['contract']!=before['contract'] or saved['model'].keys()!=before['model'].keys():
            raise ValueError('final model inventory or registered contract changed')
        for name,value in saved['model'].items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('nonfinite final model')
            if name.startswith('target_teacher.') and not torch.equal(value,before['model'][name]):
                raise ValueError('frozen teacher changed')
        if {float(v['step']) for v in saved['optimizer']['state'].values()}!={8192.}:
            raise ValueError('incomplete final optimizer state')
        atomic_write_json(args.output/'final_audit.json',dict(status='completed',cursor=8192,
            checkpoint_sha256=sha256_file(checkpoint),finite_model=True,frozen_teacher_exact=True))
        del saved,before
        cache=Path('/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2')
        execute('evaluation',['-m','tools.evaluate_expanded_native','--cache',cache,'--output',args.output/'expanded48',
            '--arm','coordinate','--variant',f'data512_{args.seed}','--checkpoint',checkpoint])
        evaluation=read(args.output/'expanded48/summary.json')
        if evaluation['status']!='completed' or evaluation['rows']!=192 or evaluation['systems']!=48 or evaluation['failures']:
            raise ValueError('complete full48 evaluation required')
        execute('chemistry',['-m','tools.audit_cached_stereochemistry','--cache',cache,
            '--generated',args.output/'expanded48','--output',args.output/'stereochemistry'],gpu_job=False,timeout=1800)
        atomic_write_json(args.output/'status.json',dict(status='completed',formal_updates=8192,
            retained_prefix_updates=17,new_formal_updates=8175,forecasts=192,default_promotion=False,
            paired_analysis='required separately; completion is not efficacy'))
    except BaseException as error:
        atomic_write_json(args.output/'status.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('input-audit','cuda-gate','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[71,72],required=True)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    run(args)
