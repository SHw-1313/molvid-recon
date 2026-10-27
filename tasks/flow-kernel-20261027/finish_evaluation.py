"""One finite CPU tail for the existing kernel generation receipt; never launches GPU."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from prepare_evaluation import ROOT,RUN,MAIN,ARMS,CASES,read,write_json,cpu_only

TASK=Path(__file__).resolve().parent
TERMINAL={'completed','completed_with_missing_or_failures','completed_with_failures','failed',
          'failed_or_budget_stopped','budget_stopped','resource_blocked','training_wait_timeout'}


def execute(args):
    out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    receipt=out/'cpu_execution.json'
    if receipt.exists():raise FileExistsError('Preserve completed/failed CPU attempt; choose another output root')
    state=dict(status='running',pid=os.getpid(),stages=[],GPU_seconds=0,model_calls=0,
               generation_execution=str(args.run_root/'generation_execution.json'),output_root=str(out),
               run_root=str(args.run_root),wait_seconds=args.wait_seconds)
    def save():write_json(receipt,state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(MAIN)+':/workspace/molvid_temporal_eval_v1')
    def step(label,argv):
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();log=out/(label+'.log')
        with log.open('x') as stream:
            process=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row=dict(stage=label,command=list(map(str,argv)),exit_code=process.returncode,wall_seconds=time.monotonic()-start,
                 CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,log=str(log))
        state['stages'].append(row);save();print(json.dumps(row),flush=True)
        return process.returncode
    save()
    # Existing checker uses available update/checkpoint evidence; a failed arm is
    # explicit coverage, never a reason to prevent scores for completed arms.
    for count in (32,512):
        step('check_pair'+str(count),[sys.executable,TASK/'check_pair.py','--step',count])
    cmd=[sys.executable,TASK/'prepare_evaluation.py','--import-generated','--run-root',args.run_root,
         '--output-root',out,'--allow-partial','--cases',*args.cases]
    if args.arms:cmd+=['--arms',*args.arms]
    if args.priorities:cmd+=['--priorities',*args.priorities]
    step('prepare',cmd)
    for label,needed in [('endpoints','imported/endpoint_panel.json'),('trajectories','imported/trajectory_panel.json'),
                         ('compatibility','trajectories/panel_index.json'),('summarize','imported/import_details.json')]:
        if (out/needed).exists():
            cmd=[sys.executable,TASK/'compare_results.py','--output-root',out,'--stage',label]
            if label=='summarize':cmd+=['--export-viewer']
            step(label,cmd)
    summary=out/'comparison/comparison_summary.json'
    if summary.exists():
        result=read(summary)
        state.update(status=result['status'],full_study_complete=result['full_study_complete'],
                     expected_new_draws=result['expected_new_draws'],available_new_draws=result['available_new_draws'],
                     reused_parent_draws=result['reused_parent_draws'],errors=result['errors'],missing_draws=len(result['missing']),
                     comparison_summary=str(summary))
    else:state.update(status='failed',full_study_complete=False,errors=[dict(reason='No comparison summary produced')])
    if any(s['exit_code']!=0 for s in state['stages']) and state['status']=='completed':state['status']='partial'
    state.update(CPU_wall_seconds=sum(r['wall_seconds'] for r in state['stages']),CPU_seconds=sum(r['CPU_seconds'] for r in state['stages']),
                 check_pair32=str(args.run_root/'check_pair32.json'),check_pair512=str(args.run_root/'check_pair512.json'),
                 viewer=dict(index=str(out/'viewer/index.html'),exists=(out/'viewer/index.html').is_file(),
                             HTML_files=len(list((out/'viewer').rglob('*.html'))),NPZ_files=len(list((out/'viewer').rglob('*.npz')))))
    save();print(json.dumps({k:state.get(k) for k in ['status','full_study_complete','available_new_draws','expected_new_draws','reused_parent_draws','CPU_wall_seconds','CPU_seconds']}),flush=True)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--output-root',type=Path);p.add_argument('--wait',action='store_true')
    p.add_argument('--allow-partial',action='store_true',help='Incomplete arms remain NA; tail always preserves actual available draws')
    p.add_argument('--max-wait-seconds',type=float,default=172800)
    p.add_argument('--arms',nargs='+',choices=[*ARMS,'control256'])
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--priorities',nargs='+',type=int,choices=range(1,8))
    args=p.parse_args();cpu_only();args.run_root=args.run_root.resolve();args.output_root=args.output_root or args.run_root/'evaluation'
    start=time.monotonic();generation={}
    if args.wait:
        wait_receipt=args.run_root/'cpu_wait.json'
        write_json(wait_receipt,dict(status='waiting_for_generation_terminal',pid=os.getpid(),generation_execution=str(args.run_root/'generation_execution.json'),
                                     output_root=str(args.output_root),max_wait_seconds=args.max_wait_seconds,GPU_seconds=0))
        while True:
            try:generation=read(args.run_root/'generation_execution.json')
            except (OSError,json.JSONDecodeError):generation={}
            if generation.get('status') in TERMINAL:break
            if time.monotonic()-start>=args.max_wait_seconds:break
            time.sleep(30)
        write_json(wait_receipt,dict(status='CPU_started',pid=os.getpid(),generation_status=generation.get('status','missing'),
                                     wait_seconds=time.monotonic()-start,output_root=str(args.output_root),GPU_seconds=0))
    args.wait_seconds=time.monotonic()-start
    result=execute(args)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
