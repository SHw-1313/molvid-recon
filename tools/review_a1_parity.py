"""Fixed same-H A1 inference sample for old/fixed code identity comparison."""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.repo))
import torch
from molvid.runtime import configure_device
from molvid.training.predictive_data import load_parent,encode_observation,fixed_projection,VIEWS
from molvid.data.store import ClipMMapDataset
from molvid.data.batch import collate_clip_records
from molvid.flow.sampling import euler_sample_frames
from molvid.latent.types import QuerySpec
device=configure_device('cuda',deterministic=True)
model,raw,release=load_parent(device);dataset=ClipMMapDataset(VIEWS/'clip_store/valid')
try:
    row=next(i for i,r in enumerate(dataset._index) if str(r[0]).endswith('_R1_a000000_dt_100ps_H4'))
    cpu=collate_clip_records([dataset[row]])
    observation=encode_observation(model,cpu,fixed_projection(128).to(device),device=device)
    query=QuerySpec(cpu.time_ps[:,4:].to(device),cpu.frame_mask[:,4:].to(device))
    with torch.no_grad():
        latent,_=euler_sample_frames(model,context=observation['context'],query=query,steps=8,seed=0)
        coordinates=model.decoder(observation['context'],latent,query).coordinates
    if a.output.exists(): raise FileExistsError(a.output)
    torch.save(dict(coordinates=coordinates.cpu(),features=observation['x'].cpu(),sample_id=cpu.sample_id),a.output)
finally:
    dataset.close()
