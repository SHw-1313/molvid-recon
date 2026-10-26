"""Finite full-state512-to4096 continuation, sparse probes, then four protocols."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-ligand-4096-20261026';RUN=ROOT/'runs/flow_ligand_4096_20261026'
PARENT='/workspace/MOLVID-FLOW-STATIC-TO-MD-20261023/runs/flow_static_to_md_20261023/new_static/coordinate_final.pt'
CAP=115200.;CASES=('atlas_6e5y_A','4J8T','atlas_1whz_A','atlas_3vnb_A','2XSB','5TV1')
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('control','ligand'),required=True);a=p.parse_args()
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
receipt=RUN/(a.arm+'_execution.json')
if receipt.exists():raise FileExistsError(receipt)
s=dict(arm=a.arm,status='running',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),budget_seconds=CAP,seconds=0.,commands=[],parent=PARENT,new_updates=3584,total_updates=4096)
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
def command(label,argv,required=True):
 remain=CAP-s['seconds']-15
 if remain<=0:raise RuntimeError('Cumulative allocation exhausted')
 log=RUN/'process_logs'/(a.arm+'_'+label+'.log');item=dict(label=label,argv=argv,log=str(log),status='running',timeout_seconds=remain);s['commands'].append(item);s['current']=label;save();start=time.monotonic()
 with log.open('x') as f:
  child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
  try:code=child.wait(timeout=remain)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   code=124
 elapsed=time.monotonic()-start;s['seconds']+=elapsed;item.update(exit_code=code,seconds=elapsed,status='completed' if code==0 else 'failed');save()
 if code and required:raise RuntimeError(f'{label} exited{code}; outputs retained, no automatic retry')
 return code

def sampling(step,checkpoint,modes,cases,seeds,root):
 for mode in modes:
  for case in cases:
   command(f'gen{step}_{mode}_{case}',[sys.executable,'-u',str(TASK/'run_sampling.py'),'--arm',a.arm,'--mode',mode,'--record',f'/workspace/MOLVID-MAIN-20261016/runs/main_integration_20261016/short50_inputs/records/{case}.pt','--checkpoint',str(checkpoint),'--expected-step',str(step),'--output',str(root/mode/case),'--allocator-GiB','24','--seeds',*map(str,seeds)],required=False)

def probe(step,small):
 cmd=[sys.executable,'-u',str(TASK/'static_probe.py'),'generate','--stage',a.arm+str(step),'--allocator-GiB','24']
 if small:cmd+=['--snapshots','atlas_6e5y_A_f000','4J8T_f000']
 command(f'static_probe{step}',cmd,required=False)

save()
try:
 out=RUN/a.arm; cap='32' if a.arm=='control' else '64'
 base=[sys.executable,'-u','tools/train_coordinate_state.py','--config',str(TASK/'configs'/f'{a.arm}.json'),'--output',str(out),'--seed','71','--allocator-GiB',cap]
 command('first32',base+['--continue-from',PARENT,'--stop-after','544']);shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step544.pt')
 for step in (1024,2048,4096):
  command(f'to{step}',base+['--resume',str(out/'checkpoint_latest.pt'),'--stop-after',str(step)])
  checkpoint=out/('coordinate_final.pt' if step==4096 else f'checkpoint_step{step}.pt')
  if step!=4096:shutil.copy2(out/'checkpoint_latest.pt',checkpoint)
  if step!=4096:
   sampling(step,checkpoint,['joint'],CASES[:2],[17],RUN/'generation_probes'/a.arm/f'step{step}')
   probe(step,True)
 # Prioritize full joint comparison. All later modes use the same saved weight.
 sampling(4096,checkpoint,['joint','q12','q4','coarse_fine'],CASES,[17,29],RUN/'generation'/a.arm)
 probe(4096,False)
 s['status']='completed' if all(x['exit_code']==0 for x in s['commands']) else 'completed_with_failures';save()
except Exception as error:s.update(status='failed_or_budget_stopped',error=repr(error));save();raise
