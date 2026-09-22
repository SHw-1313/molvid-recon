"""Recompute per-atom RMSF with future-subsequence alignment on CUDA."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np, torch
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.runtime import configure_device
from tools.evaluate_frame_joint_multitime import rmsf_arrays

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--paired-store',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--checkpoint-sha256',required=True); p.add_argument('--codec',type=Path,required=True); p.add_argument('--codec-sha256',required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
 d=configure_device(a.device,deterministic=True); ds=ClipMMapDataset(a.paired_store); idx={str(x[0]):i for i,x in enumerate(ds._index)}; rows=[json.loads(x) for x in (a.run_root/'rows.jsonl').read_text().splitlines() if x.strip()]; cache={}; updated=0
 try:
  for n,row in enumerate(rows):
   m=row['meta']; sid=m['sample_id']; h=int(m['history_frames']); key=(sid,h)
   if key not in cache:
    rec=ds[idx[sid]]; b=collate_clip_records([rec]); cb=b.to(d); cache[key]=(cb.x.float(),cb)
   target,cb=cache[key]; zpath=a.run_root/'coordinates'/row['coordinate_file']
   with np.load(zpath) as z: pred_np=np.array(z['prediction'],copy=True)
   pred=torch.as_tensor(pred_np,device=d,dtype=torch.float32); pa,ta,rs=rmsf_arrays(pred,target,cb,h); np.savez_compressed(zpath,prediction=pred_np,rmsf_prediction=pa.detach().cpu().numpy(),rmsf_target=ta.detach().cpu().numpy())
   row['rmsf']=rs; sm=row['summary']; sm['rmsf_prediction']=rs['prediction_mean']; sm['rmsf_target']=rs['target_mean']; sm['rmsf_absolute_error']=rs['mae']; sm['rmsf_correlation']=rs['pearson']; updated+=1
   if (n+1)%200==0: print(json.dumps({'rows':n+1,'updated':updated,'cache':len(cache)}),flush=True)
  tmp=a.run_root/'rows.jsonl.rmsf.tmp'; tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)); os.replace(tmp,a.run_root/'rows.jsonl'); print(json.dumps({'complete':True,'rows':len(rows),'updated':updated,'cache':len(cache)}))
 finally: ds.close()
if __name__=='__main__': main()
