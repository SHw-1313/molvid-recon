#!/bin/bash
# Complete the already authorized finite two-arm recovery comparison.
set -u
while kill -0 1235254 2>/dev/null || kill -0 1235255 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=''
python -u tasks/static-ref-20261022/check_pair.py --step 8192 > runs/static_ref_20261022/process_logs/final_pair.log 2>&1
python -u tasks/static-ref-20261022/static_probe.py score --stages control4096 control8192 concat4096 concat8192 --output runs/static_ref_20261022/static_probe/scores_new4 --allow-partial > runs/static_ref_20261022/process_logs/score_new4.log 2>&1
python -u tasks/static-ref-20261022/static_probe.py merge --sources runs/static_ref_20261022/static_probe/scores_old02 runs/static_ref_20261022/static_probe/scores_new4 --output runs/static_ref_20261022/static_probe/scores --allow-partial > runs/static_ref_20261022/process_logs/merge_scores.log 2>&1
python - <<'PY'
import json
from pathlib import Path
r=Path('runs/static_ref_20261022');receipts={}
for name in ('auxiliary','control','concat'):
 p=r/(name+'_execution.json')
 if p.exists():receipts[name]=json.loads(p.read_text())
seconds=sum(v.get('seconds',0.) for v in receipts.values())
result=dict(GPU_seconds=seconds,A100_GPU_hours=seconds/3600,budget_seconds=43200,within_budget=seconds<=43200,
 receipts={k:dict(seconds=v.get('seconds'),status=v.get('status','serial_auxiliary_finished')) for k,v in receipts.items()},
 failed_commands=[dict(lane=k,**c) for k,v in receipts.items() for c in v.get('commands',[]) if c.get('exit_code',0)!=0],
 scope='All CUDA checks/maxQ1, old0.2 baseline12, two8192 static training and48 new4096/8192 Q1 probes; independent of INPAINT',
 CPU_execution_sources=['static_probe/scores_old02/summary.json','static_probe/scores_new4/summary.json','static_probe/scores/summary.json'])
(r/'resource_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY
