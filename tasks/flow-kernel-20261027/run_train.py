"""Existing finite subprocess pattern: first32 save, then256/512 without scheduler edits."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-kernel-20261027';RUN=ROOT/'runs/flow_kernel_20261027'
PARENT='/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt'
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('control','h1','history_mix','dense_history','lag1'),required=True);p.add_argument('--allocator-GiB',type=float,default=32);a=p.parse_args()
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
receipt=RUN/(a.arm+'_execution.json')
if receipt.exists():raise FileExistsError(receipt)
s=dict(status='running',pid=os.getpid(),arm=a.arm,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),budget_seconds=21600.,seconds=0.,commands=[],parent=PARENT,planned_updates=512,allocation='training only; total study48h includes separate auxiliary/generation accounts')
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
def command(label,argv):
 left=s['budget_seconds']-s['seconds']-15
 if left<=0:raise RuntimeError('Training allocation exhausted')
 log=RUN/'process_logs'/(a.arm+'_'+label+'.log');item=dict(label=label,argv=argv,status='running',log=str(log),timeout_seconds=left);s['commands'].append(item);save();start=time.monotonic()
 with log.open('x') as f:
  child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
  try:code=child.wait(timeout=left)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   code=124
 sec=time.monotonic()-start;s['seconds']+=sec;item.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=sec);save()
 if code:raise RuntimeError(f'{label} failed; retained partial, no automatic retry')
save()
try:
 out=RUN/a.arm;base=[sys.executable,'-u','tools/train_coordinate_state.py','--config',str(TASK/'configs'/f'{a.arm}.json'),'--output',str(out),'--seed','71','--allocator-GiB',str(a.allocator_GiB)]
 command('first32',base+['--warm-start',PARENT,'--stop-after','32']);shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step32.pt')
 command('to256',base+['--resume',str(out/'checkpoint_latest.pt'),'--stop-after','256']);shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step256.pt')
 command('to512',base+['--resume',str(out/'checkpoint_latest.pt')]);s['status']='completed';save()
except Exception as error:s.update(status='failed_or_budget_stopped',error=repr(error));save();raise
