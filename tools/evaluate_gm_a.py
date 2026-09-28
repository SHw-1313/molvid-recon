"""Observed-only Euler16 A evaluation with explicit families and saved profiles."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.families import metric_family
from molvid.evaluation.geometry import coordinate_errors,bond_errors,contact_scores,clash_statistics,angle_statistics
from molvid.evaluation.physical import sampled_clash_metrics
from molvid.evaluation.temporal import TEMPORAL_METRIC_SCHEMA,temporal_metrics_v3
from molvid.flow.sampling import euler_sample_frames
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json,configure_device,sha256_file
from molvid.training.batches import prepare_frame_joint_batch
from tools.recompute_frame_gm_metrics_v3 import _jsonable,_aggregate,_write_csv,ROW_SCHEMA,SCALARS
from tools.evaluate_frame_joint_multitime import meta_for


SCALARS.update({
    'clashes_per_1000_atoms':('physical','clashes_per_1000_atoms'),
    'affected_atom_fraction':('physical','affected_atom_fraction'),
    'vdw_severity_A2_per_atom':('physical','vdw_severity_A2_per_atom'),
})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','recipe','store','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ('coordinates','profiles'): (args.output/name).mkdir(exist_ok=True)
    start=time.time();device=configure_device('cuda',deterministic=True)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    loaded=load_frame_joint_inference(args.checkpoint,expected_sha256=args.checkpoint_sha256,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
    model=loaded['model'].eval()
    recipe=json.loads(args.recipe.read_text());config=PhysicalAuxiliaryConfig.resolve(recipe['physical'])
    ds=ClipMMapDataset(args.store)
    protocol={'schema':'molvid.A.evaluation.v1','checkpoint_sha256':args.checkpoint_sha256,'recipe_sha256':sha256_file(args.recipe),
              'store_index_sha256':sha256_file(args.store/'index.txt'),'manifest_sha256':sha256_file(args.manifest),
              'steps':16,'seeds':[0,1,2],'limit':args.limit,'test_opened':False,'precision':'fp32',
              'conditioning':'observed prefix/topology/query times only','aggregation':'draw -> anchor -> replica -> system equal'}
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text())!=protocol: raise ValueError('evaluation identity differs or output already exists')
    atomic_write_json(protocol_path,protocol)
    rows_path=args.output/'rows.jsonl'
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    completed={r['row_key'] for r in rows}
    manifest=json.loads(args.manifest.read_text());by_id={str(row[0]):i for i,row in enumerate(ds._index)}
    ids=list(manifest.get('sample_ids',by_id))
    if args.limit:ids=ids[:args.limit]
    try:
        for position,sid in enumerate(ids):
            record=ds[by_id[sid]]
            history=int(record['history_frames'])
            meta=meta_for(record)
            cpu=collate_clip_records([record]);batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=history)
            target=batch.coordinate_batch.x.float()
            frames=[history+i for i in torch.where(batch.query.frame_mask[0])[0].tolist()]
            family=metric_family(record,query_frames=len(frames))
            for path,seed in (('persistence',None),('oracle',None),('generated',0),('generated',1),('generated',2)):
                key=f'{sid}|H{history}|{path}|seed={seed}'
                if key in completed:continue
                with torch.no_grad():
                    if path=='generated':
                        latent,_=euler_sample_frames(model,context=batch.observed_context,query=batch.query,steps=16,seed=seed)
                        future=model.decoder(batch.observed_context,latent,batch.query).coordinates.float()
                    elif path=='oracle':future=model.decoder(batch.observed_context,batch.target_future,batch.query).coordinates.float()
                    else:future=batch.observed_context.coordinates[-1:].expand_as(batch.target_coordinates).float()
                    prediction=torch.cat((batch.observed_context.coordinates.float(),future))
                    temporal=temporal_metrics_v3(prediction,target,batch.coordinate_batch,history_frames=history)
                    geometry={**coordinate_errors(prediction,target,batch.coordinate_batch,frames),
                              **bond_errors(prediction,target,batch.coordinate_batch,frames),
                              **contact_scores(prediction,target,batch.coordinate_batch,frames),
                              **clash_statistics(prediction,batch.coordinate_batch,frames),
                              'angles':angle_statistics(prediction,target,batch.coordinate_batch,frames)}
                    physical=sampled_clash_metrics(future,batch,config)
                name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
                np.savez_compressed(args.output/'coordinates'/name,prediction=prediction.cpu().numpy())
                np.savez_compressed(args.output/'profiles'/name,**{n:x.cpu().numpy() for n,x in temporal.profiles.items()})
                row={'schema':ROW_SCHEMA,'metric_schema':TEMPORAL_METRIC_SCHEMA,'metric_family':family,'row_key':key,
                     'path':path,'clock':'true','seed':seed,'meta':meta,'coordinate_file':name,'profile_file':name,
                     'metrics':_jsonable({**temporal.metrics,'geometry':geometry,'physical':physical})}
                with rows_path.open('a') as handle:handle.write(json.dumps(row,sort_keys=True)+'\n')
                rows.append(row);completed.add(key)
            if (position+1)%8==0:print(json.dumps({'records':position+1,'total':len(ids),'rows':len(rows)}),flush=True)
        if len(rows)!=len(ids)*5:raise RuntimeError('evaluation row coverage mismatch')
        per_system,buckets=_aggregate(rows)
        for row in per_system:row['family']=rows[0]['metric_family']['family']
        _write_csv(args.output/'per_system.csv',per_system)
        atomic_write_json(args.output/'summary.json',{'family':rows[0]['metric_family']['family'],'bucket_summary':buckets,'records':len(ids),'rows':len(rows),
            'main_rmsf_available':rows[0]['metric_family']['family']=='ordinary','gpu_hours_this_invocation':(time.time()-start)/3600,
            'checkpoint_sha256':args.checkpoint_sha256,'protocol_sha256':sha256_file(protocol_path),'test_opened':False})
    finally:ds.close()


if __name__=='__main__':main()
