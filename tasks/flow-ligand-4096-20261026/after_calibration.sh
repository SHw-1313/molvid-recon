#!/usr/bin/env bash
# One already-authorized finite lane follows its one running calibration PID.
set -euo pipefail
cd /workspace/MOLVID-FLOW-LIGAND-4096-20261026
calibration_pid=$1
while kill -0 "$calibration_pid" 2>/dev/null; do sleep 5; done
python - <<'PY'
import json
from pathlib import Path
r=Path('runs/flow_ligand_4096_20261026')
p=r/'calibration_gpu2/calibration.json'
if not p.exists() or not json.loads(p.read_text()).get('passed'):
 q=r/'ligand_execution.json'
 if not q.exists():q.write_text(json.dumps(dict(arm='ligand',status='failed_or_budget_stopped',seconds=0,budget_seconds=115200,commands=[],error='Calibration/max check did not complete; no automatic training'),indent=2)+'\n')
 raise SystemExit(1)
PY
export CUDA_VISIBLE_DEVICES=2
exec python -u tasks/flow-ligand-4096-20261026/run_arm.py --arm ligand
