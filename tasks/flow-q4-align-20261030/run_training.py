"""One finite ordered lane, no retries or extra arms."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=Path(__file__).resolve().parent;RUN=ROOT/'runs/flow_q4_align_20261030'
ARMS=('shared_joint_only','shared_q4_q1','routed_q4_q1')
def main():
 receipt=RUN/'training_lane.json'
 if receipt.exists():raise FileExistsError(receipt)
 checks=json.loads((RUN/'checks_execution.json').read_text())
 if checks['status']!='completed':raise RuntimeError('Targeted checks incomplete')
 s=dict(status='running',pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),total_GPU_budget_seconds=86400,training_cap_per_arm_seconds=10800,commands=[])
 def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
 save()
 try:
  for arm in ARMS:
   item=dict(arm=arm,status='running');s['commands'].append(item);save();start=time.monotonic()
   cmd=[sys.executable,'-u',str(TASK/'run_train.py'),'--arm',arm,'--phase','all','--budget-seconds','10800','--allocator-GiB','32']
   with (RUN/'process_logs'/f'{arm}_lane.log').open('x') as f:
    child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save();code=child.wait()
   item.update(status='completed' if code==0 else 'failed',exit_code=code,wrapper_wall_seconds=time.monotonic()-start);save()
  s['status']='completed' if all(c['exit_code']==0 for c in s['commands']) else 'partial';save()
 except Exception as e:s.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
