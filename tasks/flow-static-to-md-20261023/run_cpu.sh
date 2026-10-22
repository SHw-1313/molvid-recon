#!/usr/bin/env bash
set -euo pipefail
[[ ${CONDA_DEFAULT_ENV:-} == torch-ito ]] || { echo 'Use existing torch-ito container'; exit 2; }
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python - "$task_dir" "$@" <<'PY'
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
task=Path(sys.argv[1]);sys.path.insert(0,str(task));from prepare_scores import RUN,STUDY,ARMS,BASE
root=task.parents[1];main=Path('/workspace/MOLVID-MAIN-20261016');causal=Path('/workspace/MOLVID-FLOW-CAUSAL-20261022')
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path);p.add_argument('--allow-partial',action='store_true');p.add_argument('--cases',nargs='+');p.add_argument('--seeds',type=int,nargs='+')
a=p.parse_args(sys.argv[2:]);out=(a.output_root or a.run_root).resolve();out.mkdir(exist_ok=True,parents=True);receipt=out/'cpu_execution.json'
if receipt.exists():raise FileExistsError('Preserve old CPU attempt; use independent output root')
extra=['--allow-partial'] if a.allow_partial else [];selection=[]
for key in ('cases','seeds'):
 if getattr(a,key):selection+=['--'+key,*map(str,getattr(a,key))]
commands=[]
if STUDY=='P1':
 commands.append(('final_pair',[sys.executable,str(task/'check_pair.py'),'--step','512']))
 commands.append(('static_probe',[sys.executable,str(task/'static_probe.py'),'score','--stages','old_static128','old_static512','new_static128','new_static512','new_static_replay128','new_static_replay512','--output',str(out/'static_probe/scores'),'--allow-partial']))
else:commands.append(('final_pair',[sys.executable,str(task/'check_pair.py'),'--step','512']))
commands += [
 ('prepare',[sys.executable,str(task/'prepare_scores.py'),'--run-root',str(a.run_root),'--output-root',str(out),*selection,*extra]),
 ('common_score',[sys.executable,str(main/'tools/evaluate_common_panel.py'),'--manifest',str(out/'common_input/manifest.json'),'--output',str(out/'common_scores'),'--engine-root','/workspace/complexmd_eval_logicmd55_misato89_v0.3.1_20261014/engine','--no-local-geometry']),
 ('merge_new_geometry',[sys.executable,str(task/'compare_results.py'),'--output-root',str(out),'--merge-panel']),
 ('temporal_order',[sys.executable,str(main/'tasks/main-integration-20261016/evaluate_temporal_order.py'),'--panel-index',str(out/'comparison_inputs/panel_index.json'),'--manifest',str(out/'common_input/manifest.json'),'--models',*ARMS,'--output',str(out/'temporal_order_20261023/sidecar'),'--reuse-sidecar',str(BASE/'temporal_order_20261022/sidecar')]),
 ('compare_viewer',[sys.executable,str(task/'compare_results.py'),'--output-root',str(out),'--export-viewer',*extra]),
 ('distance',[sys.executable,str(task/'compare_results.py'),'--output-root',str(out),'--distance']),
]
state=dict(status='running',study=STUDY,run_root=str(a.run_root),output_root=str(out),GPU_seconds=0,stages=[])
def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
save();started=time.monotonic();env=dict(os.environ,PYTHONPATH=str(main))
required={'prepare':out/'common_input/manifest.json','common_score':out/'common_scores/panel_index.json','merge_new_geometry':out/'comparison_inputs/panel_index.json','temporal_order':out/'temporal_order_20261023/sidecar/draw_metrics.json'}
for name,cmd in commands:
 before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
 with (out/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN);row=dict(stage=name,command=cmd,exit_code=r.returncode,wall_seconds=time.monotonic()-t,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
 state['stages'].append(row);save();print(json.dumps(row),flush=True)
 if name in required and not required[name].exists():break
 if r.returncode and not a.allow_partial and name not in ('final_pair','static_probe'):break
state.update(status='completed' if len(state['stages'])==len(commands) and all(r['exit_code']==0 for r in state['stages']) else 'incomplete',wall_seconds=time.monotonic()-started,CPU_seconds=sum(r['CPU_seconds'] for r in state['stages']))
summary_path=out/'comparison/comparison_summary.json'
if summary_path.is_file():
 summary=json.loads(summary_path.read_text())
 state.update(available_draws=summary['available_draws'],expected_draws=summary['expected_draws'],comparison_errors=summary['errors'],comparison_completion=summary['completion_status'])
 if summary['completion_status']!='complete' or summary['errors']:state['status']='incomplete'
else:state['status']='incomplete'
save()
if state['status']!='completed':sys.exit(1)
PY
