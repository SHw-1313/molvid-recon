"""Native-frame development materialization; reserved confirmation stays closed."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .io import read_topology
from .preprocess import topology_metadata,make_clip_record
from ..provenance import freeze_run_identity
from ..runtime import atomic_write_json,sha256_file
from ..training.predictive_data import atomic_torch_save


def materialize(manifest_path,output):
    import mdtraj as md
    if output.exists():raise FileExistsError(output)
    manifest=json.loads(manifest_path.read_text())
    selected=manifest['development']
    if len(selected)!=48 or set(selected)&set(manifest['confirmation_reserved']+manifest['excluded_sealed']):
        raise ValueError('development boundary violation')
    anchors=(450,2450);lags=(100,400,1000)
    protocol=dict(systems=selected,replica='R1',anchors=list(anchors),query_intervals_ps=list(lags),
        observed_offsets_ps=[-300,-200,-100,0],query_frames=12,split='development',test_opened=False,
        confirmation_trajectories_opened=False,selection_manifest_sha256=sha256_file(manifest_path),
        source_hash_scope='selected native frame coordinate+timestamp arrays; not a full XTC file hash')
    freeze_run_identity(output,inputs=[manifest_path],purpose='Expanded native development clips',protocol=protocol)
    atomic_write_json(output/'protocol.json',protocol)
    files=[];sources=[];started=time.monotonic()
    for system in selected:
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('8GiB disk reserve reached')
        entry=manifest['entries'][system];pdb=Path(entry['pdb']);xtc=Path(entry['replicas'][0])
        if sha256_file(pdb)!=entry['pdb_sha256']:raise ValueError('PDB identity changed')
        metadata=topology_metadata(read_topology(pdb),complex_topology=False)
        specs={(anchor,lag):[anchor-30,anchor-20,anchor-10,anchor]+[anchor+i*(lag//10) for i in range(1,13)]
               for anchor in anchors for lag in lags}
        wanted=set(i for ids in specs.values() for i in ids);found={};offset=0;previous=None
        for chunk in md.iterload(str(xtc),top=str(pdb),chunk=256):
            for j,timestamp in enumerate(chunk.time):
                if previous is not None and abs(float(timestamp)-previous-10)>1e-3:
                    raise ValueError('expected native10ps timestamps')
                previous=float(timestamp);index=offset+j
                if index in wanted:found[index]=(chunk.xyz[j].copy()*10.,float(timestamp))
            offset+=len(chunk)
            if offset>max(wanted):break
        if set(found)!=wanted:raise ValueError(f'trajectory too short: {system}')
        digest=hashlib.sha256()
        for index in sorted(found):
            digest.update(np.asarray(index,dtype=np.int64).tobytes())
            digest.update(found[index][0].tobytes());digest.update(np.asarray(found[index][1],dtype=np.float64).tobytes())
        source=dict(system=system,pdb=str(pdb),pdb_sha256=entry['pdb_sha256'],xtc=str(xtc),
            xtc_size=xtc.stat().st_size,xtc_mtime_ns=xtc.stat().st_mtime_ns,
            selected_frames_sha256=digest.hexdigest(),native_dt_ps=10.,frames_read=offset)
        sources.append(source)
        for (anchor,lag),ids in specs.items():
            xyz=np.stack([found[i][0] for i in ids]);times=np.array([found[i][1] for i in ids])
            record=make_clip_record(xyz,times,metadata,sample_id=f'{system}_R1_native{anchor}_q{lag}_H4',
                source='atlas',system_id=system,replica='R1',split='valid',source_stride=1,
                native_dt_ps=10.,timestamp_provenance='native_XTC_exact_frame_selection')
            record.update(history_frames=4,lag_ps=lag,anchor=anchor,native_frame_indices=ids,
                          time_bucket_id=f'fixed_H4_100ps_query_{lag}ps')
            path=output/f'{len(files):04d}.pt'
            atomic_torch_save(path,dict(record=record,source=source))
            files.append(dict(path=path.name,sha256=sha256_file(path),system=system,anchor=anchor,
                              lag_ps=lag,sample_id=record['sample_id'],atoms=len(record['atype'])))
        atomic_write_json(output/'progress.json',dict(systems=len(sources),records=len(files),total_systems=48,
            seconds=time.monotonic()-started,test_opened=False))
        print(json.dumps(dict(systems=len(sources),records=len(files),system=system,seconds=time.monotonic()-started)),flush=True)
    atomic_write_json(output/'complete.json',dict(status='completed',files=files,sources=sources,
        systems=48,records=len(files),seconds=time.monotonic()-started,test_opened=False,
        confirmation_trajectories_opened=False,protocol_sha256=sha256_file(output/'protocol.json')))
