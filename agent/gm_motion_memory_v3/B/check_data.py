"""Metadata/IO-only raw-pair audit; contains no model execution."""
import json
from pathlib import Path
import numpy as np
from molvid.data.raw_history import RawHistoryWindow, load_pilot_inputs

card=json.loads(Path('runs/learned_memory_v1/raw_pilot_r01/data_card.json').read_text())
checks=[]
for row in card['rows']:
    path=Path(row['bank_path'])
    for dt in (100,200,400):
        for lag in (100,200,400):
            short,anchors,target=load_pilot_inputs(path,RawHistoryWindow(1280,800,dt,lag,'train'))
            for span in (3200,12800):
                long,other_anchors,other_target=load_pilot_inputs(path,RawHistoryWindow(1280,span,dt,lag,'train'))
                assert np.array_equal(anchors.x.numpy(),other_anchors.x.numpy())
                assert np.array_equal(target.x.numpy(),other_target.x.numpy())
                assert np.array_equal(short.coordinates.numpy(),long.coordinates[-len(short.coordinates):].numpy())
                assert len(long.coordinates)==span//dt+1
                assert long.physical_time_ps[0,0]==-span and long.physical_time_ps[0,-1]==0
            checks.append({'system':row['system'],'replica':row['replica'],'observed_dt_ps':dt,'query_lag_ps':lag,'paired':True})
try:
    RawHistoryWindow(1280,800,100,300,'train')
except ValueError:
    pass
else:
    raise AssertionError('300ps accepted in training')
result={'status':'pass','scope':'data IO and indexing only','test_opened':False,'model_execution':False,
        'paired_span_checks':len(checks)*2,'history_dt_query_lag_combinations':9,'checks':checks}
Path('agent/gm_motion_memory_v3/B/data_checks.json').write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='checks'})
