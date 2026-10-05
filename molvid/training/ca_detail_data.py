"""Frozen observed-only A1 generation for a CA-conditioned recovery pilot."""
import hashlib
import json
from pathlib import Path
import time
import torch
from ..data.batch import collate_clip_records
from ..data.store import ClipMMapDataset
from ..geometry.ca_detail import condition_from_record
from ..flow.sampling import euler_sample_frames
from ..runtime import configure_device,atomic_write_json,sha256_file
from ..provenance import freeze_run_identity
from ..cli.train_frame_joint import _open_data
from .batches import prepare_frame_joint_batch
from .predictive_data import load_parent,select_records,VIEWS,VALID_PROTOCOL,atomic_torch_save
from .execution_evidence import ExecutionEvidence


@torch.no_grad()
def build_ca_cache(output,train_systems=32):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True)
    model,raw,release=load_parent(device)
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    valid=ClipMMapDataset(VIEWS/'clip_store/valid')
    try:
        tr=[r for r in select_records(train,windows=2) if r['trajectory'].endswith('_R1')]
        systems=sorted({r['system'] for r in tr},key=lambda s:hashlib.sha256(('ca-detail-v1:'+s).encode()).hexdigest())[:train_systems]
        tr=[r for r in tr if r['system'] in systems]
        va=[r for r in select_records(valid,paired_views=True,windows=2) if r['trajectory'].endswith('_R1')]
        if set(systems)&{r['system'] for r in va}:raise ValueError('train/valid overlap')
        expected=json.loads(VALID_PROTOCOL.read_text())
        if sha256_file(VIEWS/'clip_store/valid/index.txt')!=expected['store_index_sha256']:raise ValueError('valid index changed')
        if sha256_file(VIEWS/'paired_view_manifest.json')!=expected['manifest_sha256']:raise ValueError('valid manifest changed')
        protocol=dict(schema='molvid.ca_detail.cache.v1',data_hash=data_hash,parent=release['selected_checkpoint'],
            codec_sha256=raw['codec']['sha256'],plan={'train':tr,'valid':va},history=4,query_frames=12,
            observed_dt_ps=100,query_dt_ps=100,solver_steps=16,train_source_seeds=[11],valid_source_seeds=[17,29],
            coarse='CA extracted from frozen A1 generated full coordinates; not a new CA generator',test_opened=False)
        freeze_run_identity(output,inputs=[release['selected_checkpoint']['path'],release['configuration']['path'],
            VIEWS/'paired_view_manifest.json'],purpose='Frozen A1 CA pilot cache',protocol=protocol)
        atomic_write_json(output/'protocol.json',protocol)
        with ExecutionEvidence(output):
            files=[];draws=0;started=time.monotonic()
            for split,plan,dataset,seeds in [('train',tr,train,[11]),('valid',va,valid,[17,29])]:
                for i,row in enumerate(plan):
                    if time.monotonic()-started>7200:raise TimeoutError('CA cache two-hour budget exceeded')
                    record=dataset[row['index']]
                    if record['sample_id']!=row['sample_id']:raise ValueError('record identity changed')
                    c=condition_from_record(record)
                    if not torch.equal(c.query_time_ps-c.observed_time_ps[-1],torch.arange(1,13)*100.):
                        raise ValueError('non-native 100ps pilot record')
                    cpu=collate_clip_records([record])
                    batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
                    if not torch.equal(batch.observed_context.coordinates.cpu(),c.observed):raise ValueError('observed gauge changed')
                    generated=[];event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
                    event_start.record()
                    for seed in seeds:
                        latent,_=euler_sample_frames(model,context=batch.observed_context,query=batch.query,steps=16,seed=seed)
                        prediction=model.decoder(batch.observed_context,latent,batch.query).coordinates.float()
                        if not torch.isfinite(prediction).all():raise FloatingPointError('nonfinite parent generation')
                        generated.append(dict(seed=seed,coordinates=prediction.cpu()));draws+=1
                    event_end.record();torch.cuda.synchronize()
                    file=output/f'{split}_{i:03d}.pt'
                    atomic_torch_save(file,dict(record=record,generated=generated,metadata=row))
                    files.append(dict(path=file.name,sha256=sha256_file(file),split=split,**row))
                    progress=dict(split=split,clips=i+1,total=len(plan),draws=draws,wall_seconds=time.monotonic()-started,
                        inference_cuda_ms=event_start.elapsed_time(event_end),atoms=len(c.atom_type),device=str(device),test_opened=False)
                    atomic_write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('parent not frozen')
            atomic_write_json(output/'complete.json',dict(status='completed',files=files,draws=draws,
                seconds=time.monotonic()-started,protocol_sha256=sha256_file(output/'protocol.json'),test_opened=False))
    finally:valid.close();close()


def load_ca_cache(cache,split):
    cache=Path(cache);manifest=json.loads((cache/'complete.json').read_text())
    if manifest['status']!='completed' or manifest['test_opened'] is not False:raise ValueError('invalid cache')
    if manifest['protocol_sha256']!=sha256_file(cache/'protocol.json'):raise ValueError('protocol changed')
    values=[]
    for row in manifest['files']:
        if row['split']!=split:continue
        path=cache/row['path']
        if sha256_file(path)!=row['sha256']:raise ValueError('cache payload changed')
        values.append(torch.load(path,map_location='cpu',weights_only=False))
    if not values:raise ValueError('empty split')
    return values
