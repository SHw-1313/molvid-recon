#!/bin/bash
# One finite six-lane run; invoke once from enter-container / torch-ito.
set -u
cd /workspace/MOLVID-FLOW-CAPACITY-20261020 || exit 2
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run=runs/flow_capacity_20261020
CUDA_VISIBLE_DEVICES=1 python -u tasks/flow-capacity-20261020/run_arm.py --arm observed --then-control > "$run/process_logs/lane_observed.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=2 python -u tasks/flow-capacity-20261020/run_arm.py --arm query_pre > "$run/process_logs/lane_query_pre.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=3 python -u tasks/flow-capacity-20261020/run_arm.py --arm temporal > "$run/process_logs/lane_temporal.log" 2>&1 & p3=$!
CUDA_VISIBLE_DEVICES=5 python -u tasks/flow-capacity-20261020/run_arm.py --arm collective > "$run/process_logs/lane_collective.log" 2>&1 & p5=$!
CUDA_VISIBLE_DEVICES=6 python -u tasks/flow-capacity-20261020/run_arm.py --arm query_post > "$run/process_logs/lane_query_post.log" 2>&1 & p6=$!
CUDA_VISIBLE_DEVICES=7 python -u tasks/flow-capacity-20261020/run_arm.py --arm wide > "$run/process_logs/lane_wide.log" 2>&1 & p7=$!
printf 'observed GPU1 pid%s; query_pre GPU2 pid%s; temporal GPU3 pid%s; collective GPU5 pid%s; query_post GPU6 pid%s; wide GPU7 pid%s\n' "$p1" "$p2" "$p3" "$p5" "$p6" "$p7"
failed=0
for pid in "$p1" "$p2" "$p3" "$p5" "$p6" "$p7"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
  printf 'At least one finite lane failed/stopped; results preserved, no automatic retries.\n'
  exit 1
fi
CUDA_VISIBLE_DEVICES='' python -u - <<'PY'
import json, os, subprocess, time
from pathlib import Path
root=Path.cwd(); run=root/'runs/flow_capacity_20261020'; start=time.monotonic(); rows=[]
commands=[
 ['python','tasks/flow-capacity-20261020/compare_capacity.py','prepare'],
 ['python','/workspace/MOLVID-MAIN-20261016/tools/evaluate_common_panel.py','--manifest',str(run/'common_input/manifest.json'),'--output',str(run/'common_scores'),'--engine-root','/workspace/complexmd_eval_logicmd55_misato89_v0.3.1_20261014/engine'],
 ['python','tasks/flow-capacity-20261020/compare_capacity.py','compare','--export-viewer']]
for i,cmd in enumerate(commands):
 began=time.monotonic(); env=dict(os.environ)
 if i==1: env['PYTHONPATH']='/workspace/MOLVID-MAIN-20261016'
 with (run/f'process_logs/final_cpu_{i}.log').open('w') as log:
  process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows.append({'argv':cmd,'seconds':time.monotonic()-began,'exit_code':process.returncode})
 (run/'cpu_execution.json').write_text(json.dumps({'GPU_seconds':0,'wall_seconds':time.monotonic()-start,'commands':rows,'status':'completed' if i==2 and not process.returncode else 'running' if not process.returncode else 'failed'},indent=2)+'\n')
 if process.returncode: raise SystemExit(process.returncode)
PY
