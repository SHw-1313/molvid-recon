"""Finite continuation after phase1: complete data, necessary FB, five trains, phase2."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_hq_ratio_20261102'
sys.path.insert(0,str(TASK))
from run_generation import read,used_seconds
ARMS=('h4q4','h4q8','h8q4','h8q8','h12q12')
p=RUN/'training_lane.json'
if p.exists():raise FileExistsError('Finite chain already exists')
assert os.environ['CONDA_DEFAULT_ENV']=='torch-ito' and os.environ['CUDA_VISIBLE_DEVICES']
s=dict(status='waiting_for_phase1',pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],budget_seconds=115200,
       planned_slots=256,phase1_wait_max_seconds=39600,data_wait_max_seconds=28800,commands=[],errors=[])
def save():p.write_text(json.dumps(s,indent=2)+'\n')
def wait_for(path,accept,seconds):
 start=time.monotonic()
 while time.monotonic()-start<seconds:
  d=read(path)
  if accept(d):return d
  if d.get('status') in ('failed','resource_blocked','completed_with_missing_or_failures','budget_stopped','failed_or_budget_stopped'):
   raise RuntimeError(f'Upstream finite stage stopped: {path}: {d.get("status")}')
  time.sleep(30)
 raise TimeoutError(str(path))
def child(label,argv,timeout,gpu_receipt=None):
 log=RUN/'process_logs'/f'{label}.log';item=dict(label=label,command=argv,status='running',log=str(log));s['commands'].append(item);save()
 r=dict(status='running',pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],seconds=0.,cap_seconds=timeout)
 if gpu_receipt:gpu_receipt.write_text(json.dumps(r,indent=2)+'\n')
 start=time.monotonic()
 with log.open('x') as f:
  proc=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=proc.pid;save()
  if gpu_receipt:r['child_pid']=proc.pid;gpu_receipt.write_text(json.dumps(r,indent=2)+'\n')
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   proc.terminate()
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   code=124
 elapsed=time.monotonic()-start;item.update(seconds=elapsed,exit_code=code,status='completed' if code==0 else 'failed');save()
 if gpu_receipt:
  r.update(seconds=elapsed,exit_code=code,status=item['status']);gpu_receipt.write_text(json.dumps(r,indent=2)+'\n')
 return code
save()
try:
 wait_for(RUN/'generation_phase1_execution.json',lambda d:d.get('status')=='completed',39600)
 s['status']='waiting_for_complete_data';save()
 data=wait_for(RUN/'data/manifest.json',lambda d:d.get('status')=='complete' and d.get('all_candidates_processed') and d.get('failures')==[],28800)
 s.update(effective_joint_windows=len(data['rows']),effective_optimizer_updates=2*len(data['rows']),target_frames_per_arm=25*len(data['rows']),status='training_checks');save()
 if child('training_checks',[sys.executable,'-u',str(TASK/'check_training.py')],min(1800,115200-used_seconds()-30),RUN/'training_checks_execution.json'):raise RuntimeError('Targeted training checks failed; no automatic retry')
 s['status']='training';save()
 for arm in ARMS:
  left=115200-used_seconds();cap=min(9000,left-30000)
  if cap<=60:raise RuntimeError('Training cap would consume fixed phase2 reserve; partial retained')
  code=child(f'{arm}_train_wrapper',[sys.executable,'-u',str(TASK/'run_train.py'),'--arm',arm,'--phase','all','--budget-seconds',str(cap),'--allocator-GiB','32'],cap+90)
  if code:s['errors'].append(dict(arm=arm,reason='Finite train failed; no auto retry'));save()
 s['status']='pair_checks';save()
 for step in (32,256):
  if child(f'check_pair{step}',[sys.executable,str(TASK/'check_pair.py'),'--step',str(step),'--fail-on-error'],600):raise RuntimeError('Real checkpoint pairing failed')
 s['status']='phase2_generation';save()
 if child('phase2_generation',[sys.executable,'-u',str(TASK/'run_generation.py'),'--phase','phase2','--allocator-GiB','32'],max(60,115200-used_seconds()+90)):raise RuntimeError('Phase2 incomplete; CPU partial retained')
 s.update(status='completed',total_GPU_seconds=used_seconds());save()
except Exception as error:
 s.update(status='failed',error=repr(error),total_GPU_seconds=used_seconds());save();raise
