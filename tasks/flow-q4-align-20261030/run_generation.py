"""One finite, ordered Q4 alignment generation process after all three training receipts."""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent
RUN=ROOT/'runs/flow_q4_align_20261030';ARMS=('shared_joint_only','shared_q4_q1','routed_q4_q1')
TERMINAL={'completed','failed','failed_or_budget_stopped','budget_stopped'}


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--wait',action='store_true');p.add_argument('--max-wait-seconds',type=float,default=86400)
    p.add_argument('--allocator-GiB',type=float,default=32)
    a=p.parse_args();run=a.run_root.resolve();receipt=run/'generation_execution.json'
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Parent must assign a real GPU in existing torch-ito')
    if receipt.exists():raise FileExistsError('Do not duplicate or restart finite generation')
    plan=read(run/'eval_inputs/genplan.json')
    if plan.get('expected_new_draws')!=108:raise ValueError('Frozen108-draw scope required')
    state=dict(status='waiting_for_training',pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],budget_seconds=86400,
               check_seconds=None,generation_gpu_seconds=0.,training_gpu_seconds=None,jobs=[],errors=[],missing=[],
               expected_jobs=len(plan['jobs']),expected_draws=plan['expected_new_draws'],completed_draws=0,
               genplan=str(run/'eval_inputs/genplan.json'))
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
    save();began=time.monotonic()
    while True:
        training={arm:read(run/f'{arm}_all_execution.json') for arm in ARMS}
        lane=read(run/'training_lane.json')
        if lane.get('status') in ('completed','partial','failed'):
            for arm in ARMS:
                if not training[arm]:
                    command=next((c for c in lane.get('commands',[]) if c.get('arm')==arm),None)
                    reserved=0. if command is None else float(command.get('wrapper_wall_seconds',10800.))
                    training[arm]={'status':'failed','seconds':reserved,'reason':'Finite training lane ended without arm receipt',
                                   'accounting':'not_started' if command is None else 'lane_wrapper_wall_seconds' if 'wrapper_wall_seconds' in command else 'conservative_arm_cap_reserved'}
        if all(s.get('status') in TERMINAL for s in training.values()):break
        if not a.wait or time.monotonic()-began>=a.max_wait_seconds:
            state.update(status='training_wait_timeout',training_receipts=training);save();return
        time.sleep(30)
    checks=read(run/'checks_execution.json')
    if not isinstance(checks.get('seconds'),(int,float)) or not math.isfinite(checks['seconds']) or checks['seconds']<0:
        state.update(status='resource_blocked',errors=[{'reason':'Missing actual check/failure GPU seconds'}]);save();return
    if any(not isinstance(s.get('seconds'),(int,float)) or not math.isfinite(s['seconds']) or s['seconds']<0 for s in training.values()):
        state.update(status='resource_blocked',errors=[{'reason':'Missing finite actual/reserved training seconds'}]);save();return
    state.update(check_seconds=float(checks['seconds']),checks_receipt=str(run/'checks_execution.json'))
    state.update(training_gpu_seconds=sum(float(s.get('seconds',0)) for s in training.values()),training_receipts=training,
                 wait_wall_seconds=time.monotonic()-began,status='running');save()
    logs=run/'generation_logs';logs.mkdir(exist_ok=True)
    unavailable={arm for arm in ARMS if training[arm].get('status')!='completed' or not (run/arm/'coordinate_final.pt').is_file()}
    budget_stopped=False
    for index,job in enumerate(plan['jobs']):
        if job['arm'] in unavailable:
            state['missing'].append(dict(job,reason='Training incomplete/failed; checkpoint unavailable'));save();continue
        left=86400-state['check_seconds']-state['training_gpu_seconds']-state['generation_gpu_seconds']
        if left<=31:
            state['missing'] += [dict(j,reason='Fixed total GPU budget exhausted before this planned condition') for j in plan['jobs'][index:]]
            budget_stopped=True;break
        argv=[sys.executable,'-u',str(TASK/'sample_entry.py'),'--entry',job['entry_id'],'--arm',job['arm'],
              '--checkpoint',job['checkpoint'],'--output',job['output'],'--manifest',str(run/'eval_inputs/manifest.json'),
              '--allocator-GiB',str(a.allocator_GiB),'--seeds',*map(str,job['seeds'])]
        log=logs/f'{index:03d}_{job["arm"]}_{job["entry_id"]}.log';item=dict(job,index=index,status='running',log=str(log),remaining_total_GPU_seconds=left)
        state['jobs'].append(item);save();start=time.monotonic()
        with log.open('x') as f:
            process=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
            item['pid']=process.pid;save()
            try:code=process.wait(timeout=left-30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=124
        elapsed=time.monotonic()-start;state['generation_gpu_seconds']+=elapsed
        protocol=read(Path(job['output'])/'protocol.json')
        present=[seed for seed in job['seeds'] if seed in protocol.get('seeds',[]) and str(seed) in protocol.get('draws',{}) and (Path(job['output'])/f'coordinates_seed{seed}.pt').is_file()]
        success=code==0 and protocol.get('status')=='completed' and len(present)==len(job['seeds'])
        item.update(status='completed' if success else 'failed_or_partial',exit_code=code,seconds=elapsed,completed_seeds=present)
        state['completed_draws']+=len(present)
        if not success:state['errors'].append(dict(arm=job['arm'],entry_id=job['entry_id'],exit_code=code,completed_seeds=present,protocol=str(Path(job['output'])/'protocol.json')))
        for seed in job['seeds']:
            if seed not in present:state['missing'].append(dict(arm=job['arm'],entry_id=job['entry_id'],seed=seed,reason='Real generation process failed/budget stopped; raw partial retained'))
        save();print(json.dumps(item),flush=True)
        if code==124:
            budget_stopped=True;state['missing'] += [dict(j,reason='Fixed total GPU budget exhausted') for j in plan['jobs'][index+1:]];break
    state['status']='completed' if state['completed_draws']==plan['expected_new_draws'] and not state['errors'] else 'budget_stopped' if budget_stopped else 'completed_with_missing_or_failures'
    state['total_GPU_seconds']=state['check_seconds']+state['training_gpu_seconds']+state['generation_gpu_seconds']
    save();print(json.dumps({k:state[k] for k in ('status','completed_draws','expected_draws','total_GPU_seconds')}),flush=True)

if __name__=='__main__':main()
