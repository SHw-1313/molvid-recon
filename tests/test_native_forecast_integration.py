"""Real frozen-A1 CUDA checks for the new observed-only evaluation boundary."""
from pathlib import Path
import copy
import numpy as np
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.native_forecast import molvid_forecast
from molvid.training.predictive_data import load_parent


def test_cuda_real_A1_unit_source_parity_and_future_boundary():
    assert torch.cuda.is_available()
    torch.use_deterministic_algorithms(True)
    path=Path('runs/geometry_scaling_20261005/development48_v2/0000.pt')
    record=torch.load(path,map_location='cpu',weights_only=False)['record']
    changed=copy.deepcopy(record)
    for name in ('x','bpos'): changed[name][4:]+=100000
    first=collate_clip_records([record]);second=collate_clip_records([changed])
    observed=slice_clip_frames(first,0,4);mutated=slice_clip_frames(second,0,4)
    for name in ('x','bpos','time_ps','frame_mask'):
        torch.testing.assert_close(getattr(observed,name),getattr(mutated,name),atol=0,rtol=0)
    model,_,_=load_parent(torch.device('cuda'))
    original=molvid_forecast(model,observed,first.time_ps[:,4:],device=torch.device('cuda'),seed=17)
    explicit=molvid_forecast(model,mutated,second.time_ps[:,4:],device=torch.device('cuda'),seed=17,
        diagnostic_source_scales=(1.,1.))
    torch.testing.assert_close(original,explicit,atol=0,rtol=0)
    assert torch.isfinite(original).all()
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
