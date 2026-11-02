"""Finite phase-specific generation; one process per HQ condition, all failures retained."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_hq_ratio_20261102'
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return {}
def used_seconds():
 paths=list(RUN.glob('*checks_execution*.json'))+list(RUN.glob('train_*_execution.json'))+list(RUN.glob('generation_phase*_execution.json'))
 return sum(float(read(p).get('seconds',0)) for p in paths)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['phase1','phase2'],required=True);p.add_argument('--allocator-GiB',type=float,default=32);a=p.parse_args()
 assert os.environ['CONDA_DEFAULT_ENV']=='torch-ito' and os.environ['CUDA_VISIBLE_DEVICES']
 receipt=RUN/f'generation_{a.phase}_execution.json'
 if receipt.exists():raise FileExistsError(receipt)
 plan=read(RUN/'eval_inputs/genplan.json');jobs=[x for x in plan['jobs'] if x['phase']==a.phase]
 assert len(jobs)==30 and sum(len(x['seeds']) for x in jobs)==60
 state=dict(status='running',phase=a.phase,pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],allocator_GiB=a.allocator_GiB,
  budget_seconds=115200,phase_cap_seconds=36000 if a.phase=='phase1' else 115200,seconds=0.,expected_draws=60,completed_draws=0,jobs=[],errors=[],missing=[])
 def save():
  tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
 save();logs=RUN/'generation_logs';logs.mkdir(exist_ok=True)
 for index,job in enumerate(jobs):
  left=min(115200-used_seconds(),state['phase_cap_seconds']-state['seconds'])
  if left<=30:
   state['missing'].extend(dict(j,reason='Finite GPU budget exhausted') for j in jobs[index:]);break
  if not Path(job['checkpoint']).is_file():state['missing'].append(dict(job,reason='Checkpoint unavailable'));save();continue
  argv=[sys.executable,'-u',str(TASK/'sample_entry.py'),'--arm',a.phase,'--entry',job['entry_id'],'--checkpoint',job['checkpoint'],'--output',job['output'],'--manifest',str(RUN/'eval_inputs/manifest.json'),'--allocator-GiB',str(a.allocator_GiB),'--seeds',*map(str,job['seeds'])]
  log=logs/f'{a.phase}_{index:03d}_{job["entry_id"]}.log';item=dict(job,status='running',log=str(log),command=argv,remaining_total_GPU_seconds=left);state['jobs'].append(item);save();start=time.monotonic()
  with log.open('x') as f:
   child=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'});item['pid']=child.pid;save()
   try:code=child.wait(timeout=max(1,left-25))
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    code=124
  elapsed=time.monotonic()-start;state['seconds']+=elapsed
  protocol=read(Path(job['output'])/'protocol.json')
  present=[s for s in job['seeds'] if s in protocol.get('seeds',[]) and str(s) in protocol.get('draws',{}) and (Path(job['output'])/f'coordinates_seed{s}.pt').is_file()]
  ok=code==0 and protocol.get('status')=='completed' and len(present)==len(job['seeds'])
  item.update(status='completed' if ok else 'failed',exit_code=code,seconds=elapsed,completed_seeds=present);state['completed_draws']+=len(present)
  if not ok:state['errors'].append(dict(entry=job['entry_id'],exit_code=code,protocol=str(Path(job['output'])/'protocol.json')))
  state['missing'].extend(dict(entry=job['entry_id'],seed=s,reason='Generation incomplete') for s in job['seeds'] if s not in present)
  save();print(json.dumps(item),flush=True)
  if code!=0:
   # Diagnose a genuine entry failure before more GPU work, no infinite auto retries.
   state['missing'].extend(dict(j,reason='Finite phase stopped after first genuine failure') for j in jobs[index+1:]);break
 state.update(status='completed' if state['completed_draws']==60 and not state['errors'] else 'completed_with_missing_or_failures',total_GPU_seconds=used_seconds());save()
 if state['status']!='completed':raise SystemExit(2)
if __name__=='__main__':main()
