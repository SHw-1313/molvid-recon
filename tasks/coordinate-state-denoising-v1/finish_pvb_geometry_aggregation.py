"""Reaggregate the already-running first PVB scorer without repeating its work.

That process loaded the initial script before the plain-system-ID fix. Retain
its completed original summary, then aggregate the same full128 scored rows.
"""
import json
from pathlib import Path
import time

from score_complex_geometry import summarize
from molvid.runtime import atomic_write_json

root=Path('/workspace/molvid-coordinate-state-denoising-v1/runs/coordinate_state_denoising_v1/complex_geometry/pvb')
while not (root/'summary.json').exists():
    time.sleep(30)
old=json.loads((root/'summary.json').read_text())
if old['status']!='completed':
    raise RuntimeError('Do not reaggregate a failed/incomplete score')
if not old['estimates']:
    rows=json.loads((root/'rows.json').read_text())
    inventory=json.loads(Path('/workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data/summary.json').read_text())
    new=summarize(rows,inventory,old['variant'],old['seconds'])
    new['aggregation_fix']='Use declared plain-ID old16+new48 rather than metadata dictionaries; no score/forecast recomputation'
    atomic_write_json(root/'summary_before_plain_ID_fix.json',old)
    atomic_write_json(root/'summary.json',new)
    print('Reaggregated all64 using original128 condition rows; no model calls',flush=True)
