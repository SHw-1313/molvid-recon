"""Build paired, raw-timestamped multitime valid views for evaluation only."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any
import numpy as np

from molvid.data.io import read_topology
from molvid.data.preprocess import make_clip_record, topology_metadata
from molvid.data.store import ClipMMapDataset, ClipMMapWriter
from molvid.runtime import sha256_file


def _args():
    p=argparse.ArgumentParser()
    p.add_argument('--source-valid-store',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--anchor-windows',type=int,nargs='+',default=[0,2])
    p.add_argument('--systems-limit',type=int)
    return p.parse_args()


def _selected(source: Path, anchors: list[int], limit: int|None):
    ds=ClipMMapDataset(source)
    records=[]
    try:
        parsed=[]
        for index,(sid,_,_) in enumerate(ds._index):
            sid=str(sid)
            if '_dt_300ps_' not in sid: continue
            m=re.match(r'^(atlas_.+)_R([123])_dt_300ps_w(\d{6})$',sid)
            if not m: raise RuntimeError(f'cannot parse valid id {sid}')
            w=int(m.group(3))
            if w in anchors: parsed.append((m.group(1),f'R{m.group(2)}',w,index,sid))
        systems=sorted({a for a,_,_,_,_ in parsed})
        if limit is not None: systems=systems[:int(limit)]
        chosen=[x for x in parsed if x[0] in systems]
        expected={(system,rep,w) for system in systems for rep in ('R1','R2','R3') for w in anchors}
        actual={(a,b,c) for a,b,c,_,_ in chosen}
        missing=sorted(expected-actual)
        if missing: raise RuntimeError(f'missing selected 300ps windows: {missing[:5]}')
        for system,rep,w,index,sid in chosen:
            rec=ds[index]
            if str(rec['system_id']) != system.removeprefix('atlas_') or str(rec['replica']) != rep:
                raise RuntimeError(f'explicit metadata mismatch for {sid}')
            records.append({'system':system,'system_id':system.removeprefix('atlas_'),'replica':rep,'anchor_window_300':w,'source_index_300':index,'source_sample_id_300':sid})
    finally: ds.close()
    return systems,records


def _read_needed(xtc: Path,pdb: Path,needed: set[int]):
    import mdtraj as md
    coords={}; times={}; raw_index=0; observed=None; prev=None
    for chunk in md.iterload(str(xtc),top=str(pdb),chunk=256):
        xyz=np.asarray(chunk.xyz,dtype=np.float32)*10.0
        ts=np.asarray(chunk.time,dtype=np.float64)
        if xyz.shape[0]!=ts.size: raise RuntimeError(f'bad XTC chunk {xtc}')
        for frame,t in zip(xyz,ts):
            if prev is not None:
                delta=float(t-prev)
                if observed is None: observed=delta
                elif not np.isclose(delta,observed,rtol=1e-5,atol=1e-3): raise RuntimeError(f'irregular native time {xtc}')
            if raw_index in needed:
                coords[raw_index]=frame; times[raw_index]=float(t)
            prev=float(t); raw_index+=1
    if observed is None or not np.isclose(observed,10.0,rtol=1e-4,atol=1e-3): raise RuntimeError(f'native dt mismatch {xtc}: {observed}')
    missing=sorted(needed-set(coords))
    if missing: raise RuntimeError(f'{xtc} missing raw frames {missing[:5]} (length {raw_index})')
    return coords,times,raw_index


def main():
    a=_args()
    if a.output_root.exists(): raise FileExistsError(f'refusing existing output {a.output_root}')
    anchors=sorted(set(int(x) for x in a.anchor_windows))
    if not anchors or any(x<0 for x in anchors): raise ValueError('anchor windows must be nonnegative')
    systems,selected=_selected(a.source_valid_store,anchors,a.systems_limit)
    output=a.output_root; store_root=output/'clip_store'/'valid'; store_root.mkdir(parents=True)
    records=[]; source_inventory=[]
    for n,system in enumerate(systems,1):
        system_id=system.removeprefix('atlas_'); pdb=a.raw_root/system_id/f'{system_id}.pdb'
        if not pdb.exists(): raise FileNotFoundError(pdb)
        metadata=topology_metadata(read_topology(pdb),complex_topology=False)
        for rep in ('R1','R2','R3'):
            xtc=a.raw_root/system_id/f'{system_id}_prod_{rep}_fit.xtc'
            if not xtc.exists(): raise FileNotFoundError(xtc)
            selected_rows=[r for r in selected if r['system']==system and r['replica']==rep]
            needed=set()
            for r in selected_rows:
                # Common t0 is the final frame (frame 15) of the selected 300 ps anchor.
                # This leaves enough raw history for H8 at dt400.
                anchor=int((r['anchor_window_300']*16+15)*30)
                r['anchor_raw_frame_index']=anchor
                for stride in (10,20,30,40):
                    for h in (4,8):
                        start=anchor-(h-1)*stride
                        needed.update(start+i*stride for i in range(16))
            coords,times,raw_length=_read_needed(xtc,pdb,needed)
            source_inventory.append({'path':str(xtc.resolve()),'size':xtc.stat().st_size,'mtime_ns':xtc.stat().st_mtime_ns,'sha256':sha256_file(xtc),'raw_frame_count':raw_length})
            for r in selected_rows:
                anchor=r['anchor_raw_frame_index']; anchor_time=times[anchor]
                for dt,stride in ((100,10),(200,20),(300,30),(400,40)):
                    for h in (4,8):
                        indices=[anchor-(h-1)*stride+i*stride for i in range(16)]
                        frame=np.stack([coords[i] for i in indices],axis=0)
                        abs_time=np.asarray([times[i] for i in indices],dtype=np.float64)
                        sid=f"{system}_{rep}_a{r['anchor_window_300']:06d}_dt_{dt}ps_H{h}"
                        rec=make_clip_record(frame,abs_time,metadata,sample_id=sid,source='atlas',system_id=system_id,replica=rep,split='valid',source_stride=stride,native_dt_ps=10.0,timestamp_provenance='ATLAS_XTC_time_ps')
                        rec.update({'sampled_delta_time_ps':float(dt),'time_bucket_id':f'dt_{dt}ps','lag_ps':dt,'history_frames':h,'anchor_id':f'a{r["anchor_window_300"]:06d}','anchor_window_300':r['anchor_window_300'],'anchor_raw_frame_index':anchor,'anchor_time_ps':float(anchor_time),'source_raw_frame_indices':indices,'source_absolute_time_ps':abs_time.tolist(),'source_sample_id_300':r['source_sample_id_300'],'pair_rule':'common t0 at dt300 anchor frame 15; anchor windows 0 and 2','raw_trajectory_path':str(xtc.resolve()),'raw_frame_count':raw_length})
                        records.append(rec)
    with ClipMMapWriter(store_root) as writer:
        for rec in records: writer.append(rec)
    protocol={'schema_version':'molvid.frame_joint.multitime.eval_views.v1','source_valid_store':str(a.source_valid_store.resolve()),'source_valid_index_sha256':sha256_file(a.source_valid_store/'index.txt'),'raw_root':str(a.raw_root.resolve()),'raw_native_dt_ps':10.0,'strides_ps':{'100':10,'200':20,'300':30,'400':40},'systems':systems,'replicas':['R1','R2','R3'],'anchor_windows_300':anchors,'anchor_rule':'anchor raw index = (300ps_window*16 + 15)*30; same t0 shared across H4/H8 and all lag views','history_frames':[4,8],'views_per_trajectory':len(anchors)*4*2,'record_count':len(records),'coordinate_unit':'angstrom','source_inventory':source_inventory,'selected_records':selected,'test_opened':False}
    (output/'paired_view_manifest.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(output),'systems':len(systems),'selected_anchor_rows':len(selected),'records':len(records),'index_sha256':sha256_file(store_root/'index.txt')},sort_keys=True))

if __name__=='__main__': main()
