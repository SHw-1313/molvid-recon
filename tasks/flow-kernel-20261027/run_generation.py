"""One finite predeclared generation list after the five independent training lanes.

This reuses ordinary subprocess receipts. Only one generation process may own
this receipt; no new scheduling/service layer or automatic failed-draw retry.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-kernel-20261027';RUN=ROOT/'runs/flow_kernel_20261027'
ARMS=('control','h1','history_mix','dense_history','lag1');TOTAL=172800.
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--allocator-GiB',type=float,default=28);a=p.parse_args()
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
receipt=RUN/'generation_execution.json'
if receipt.exists():raise FileExistsError(receipt)
s=dict(status='waiting_for_training' if a.wait else 'preparing',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),study_total_budget_seconds=TOTAL,seconds=0.,commands=[],allocation='total48h minus all actual auxiliary and training process seconds')
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
save();started_wait=time.monotonic()
while True:
 receipts=[RUN/(arm+'_execution.json') for arm in ARMS]
 ready=all(p.exists() and read(p)['status']!='running' for p in receipts)
 if ready:break
 if not a.wait:raise RuntimeError('All training costs must be settled before unreserved generation')
 if time.monotonic()-started_wait>36*3600:
  s.update(status='training_wait_timeout',error='No generation started; training receipts still active/missing');save();raise SystemExit(124)
 time.sleep(30)
training={arm:read(RUN/(arm+'_execution.json')) for arm in ARMS};aux=read(RUN/'auxiliary_execution.json')
if any(c['status']=='running' for c in aux['commands']):raise RuntimeError('Auxiliary costs still running')
used=sum(x['seconds'] for x in training.values())+aux['seconds']
s.update(status='running',budget_seconds=max(0.,TOTAL-used),prior_GPU_seconds=used,
 training_status={k:v['status'] for k,v in training.items()},training_GPU_seconds={k:v['seconds'] for k,v in training.items()},auxiliary_GPU_seconds=aux['seconds'])
plan=read(RUN/'eval_inputs/genplan.json');jobs=plan['jobs']
s.update(planned_jobs=len(jobs),planned_draws=sum(len(x['seeds']) for x in jobs));save()
try:
 for number,job in enumerate(jobs):
  arm=job['arm'];entry=job['entry_id'];checkpoint=Path(job['checkpoint']);out=RUN/'generation'/arm/entry
  item=dict(number=number,arm=arm,entry_id=entry,seeds=job['seeds'],priority=job['priority'],checkpoint=str(checkpoint),output=str(out),seconds=0.)
  s['commands'].append(item)
  if not checkpoint.is_file():item.update(status='missing_checkpoint',reason='Training partial/failed; never substitute parent');save();continue
  protocol=out/'protocol.json'
  if out.exists():
   if protocol.exists():
    old=read(protocol)
    if old['status']=='completed' and old['arm']==arm and old['checkpoint']==str(checkpoint.resolve()) and set(old['seeds'])==set(job['seeds']):
     item.update(status='reused_completed',reason='Existing identical task output retained');save();continue
   item.update(status='existing_partial_not_retried');save();continue
  left=s['budget_seconds']-s['seconds']-10
  if left<=0:s['status']='budget_stopped';save();break
  # A single authorized shared device; do not take memory from another job.
  gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
  available=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
  if float(available)<(a.allocator_GiB+2)*1024:
   item.update(status='resource_blocked',free_MiB=float(available));s['status']='resource_blocked';save();break
  argv=[sys.executable,'-u',str(TASK/'sample_entry.py'),'--entry',entry,'--arm',arm,'--checkpoint',str(checkpoint),'--output',str(out),'--allocator-GiB',str(a.allocator_GiB),'--seeds',*map(str,job['seeds'])]
  log=RUN/'process_logs'/f'gen_{number:04d}_{arm}.log';item.update(status='running',argv=argv,log=str(log),timeout_seconds=left);save();begin=time.monotonic()
  with log.open('x') as f:
   child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
   try:code=child.wait(timeout=left)
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=8)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    code=124
  elapsed=time.monotonic()-begin;s['seconds']+=elapsed;item.update(status='completed' if code==0 else 'failed',seconds=elapsed,exit_code=code);save()
  if code==124:s['status']='budget_stopped';save();break
 else:s['status']='completed' if all(x['status'] in ('completed','reused_completed') for x in s['commands']) else 'completed_with_missing_or_failures'
 save()
except Exception as error:s.update(status='failed',error=repr(error));save();raise
