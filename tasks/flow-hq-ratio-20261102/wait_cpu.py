"""Finite CPU tail for both phases; inspect actual coverage, never exit-code-only completion."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_hq_ratio_20261102'
sys.path.insert(0,str(TASK));from run_generation import read
p=RUN/'cpu_wait.json'
if p.exists():raise FileExistsError(p)
s=dict(status='waiting_for_phase1',pid=os.getpid(),GPU_seconds=0,model_calls=0,stages=[],max_wait_seconds_per_phase=144000)
def save():p.write_text(json.dumps(s,indent=2)+'\n')
save()
for phase,out in [('phase1',RUN/'evaluation_phase1'),('phase2',RUN/'evaluation')]:
 s['status']='waiting_for_'+phase;save();start=time.monotonic()
 while time.monotonic()-start<144000:
  gen=read(RUN/f'generation_{phase}_execution.json');lane=read(RUN/'training_lane.json')
  if gen.get('status') in ('completed','completed_with_missing_or_failures','budget_stopped','failed'):break
  if phase=='phase2' and lane.get('status')=='failed':break
  time.sleep(30)
 else:s.update(status='wait_timeout',phase=phase);save();raise SystemExit(2)
 args=['bash',str(TASK/'run_cpu.sh'),'--phases',phase,'--allow-partial','--output-root',str(out)]
 if phase=='phase2':args+=['--reuse-evaluation',str(RUN/'evaluation_phase1')]
 item=dict(phase=phase,status='running',command=args,output=str(out));s['stages'].append(item);s['status']='scoring_'+phase;save()
 with (RUN/'process_logs'/f'cpu_{phase}.log').open('x') as f:
  child=subprocess.Popen(args,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
  try:code=child.wait(timeout=43200)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   code=124
 summary=read(out/'comparison/comparison_summary.json')
 item.update(exit_code=code,status=summary.get('status','failed'),available_new_draws=summary.get('available_new_draws',0),expected_new_draws=summary.get('expected_new_draws'),errors=summary.get('errors',['No summary']),summary=str(out/'comparison/comparison_summary.json'));save()
s['status']='completed' if all(x['exit_code']==0 and x['available_new_draws']==60 and not x['errors'] for x in s['stages']) else 'partial_or_failed';save()
