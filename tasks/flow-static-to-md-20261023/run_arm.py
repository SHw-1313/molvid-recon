"""Finite matched MD continuation, two restoration checkpoints and three protocols."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-static-to-md-20261023';RUN=ROOT/'runs/flow_static_to_md_20261023'
PARENTS={'old_static':'/workspace/MOLVID-FLOW-CAPACITY-20261020/runs/flow_capacity_20261020/query_post/static/coordinate_final.pt','new_static':'/workspace/MOLVID-STATIC-REF-20261022/runs/static_ref_20261022/control/coordinate_final.pt','new_static_replay':'/workspace/MOLVID-STATIC-REF-20261022/runs/static_ref_20261022/control/coordinate_final.pt'}
CAP=18000.;CASES=('atlas_6e5y_A','4J8T','atlas_1whz_A','atlas_3vnb_A','2XSB','5TV1')
p=argparse.ArgumentParser();p.add_argument('--arm',choices=PARENTS,required=True);a=p.parse_args()
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito':raise RuntimeError('Existing torch-ito required')
receipt=RUN/(a.arm+'_execution.json')
if receipt.exists():raise FileExistsError(receipt)
s=dict(arm=a.arm,status='running',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),budget_seconds=CAP,seconds=0.,commands=[])
def save():receipt.write_text(json.dumps(s,indent=2)+'\n')
def command(label,argv):
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
 if code:raise RuntimeError(f'{label} exited{code}; outputs retained, no automatic retry')
save()
try:
 out=RUN/a.arm;base=[sys.executable,'-u','tools/train_coordinate_state.py','--config',str(TASK/'configs'/f'{a.arm}.json'),'--output',str(out),'--seed','71','--allocator-GiB','32']
 command('first32',base+['--warm-start',PARENTS[a.arm],'--stop-after','32']);shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step32.pt')
 command('to128',base+['--resume',str(out/'checkpoint_step32.pt'),'--stop-after','128']);shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_step128.pt')
 command('to512',base+['--resume',str(out/'checkpoint_step128.pt')])
 for step in (128,512):command(f'probe{step}',[sys.executable,'-u',str(TASK/'static_probe.py'),'generate','--stage',a.arm+str(step),'--allocator-GiB','24'])
 for mode in ('joint','q12','coarse_fine'):
  for case in CASES:command(f'{mode}_{case}',[sys.executable,'-u',str(TASK/'run_sampling.py'),'--arm',a.arm,'--mode',mode,'--record',f'/workspace/MOLVID-MAIN-20261016/runs/main_integration_20261016/short50_inputs/records/{case}.pt','--checkpoint',str(out/'coordinate_final.pt'),'--output',str(RUN/'generation'/a.arm/mode/case),'--allocator-GiB','24'])
 s['status']='completed';save()
except Exception as error:s.update(status='failed_or_budget_stopped',error=repr(error));save();raise
