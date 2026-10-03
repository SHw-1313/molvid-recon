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


def _evaluation_paths(seeds: list[int]) -> tuple[tuple[str, int | None], ...]:
    if not seeds:
        raise ValueError("at least one generated seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("generated seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("generated seeds must be unique")
    return (
        ("persistence", None),
        ("oracle", None),
        *(("generated", seed) for seed in seeds),
    )


BUDGET_STOP_EXIT_CODE = 75


def evaluation_protocol(
    *,
    checkpoint_sha256: str,
    recipe: Path,
    store: Path,
    manifest: Path,
    limit: int | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Return the complete scientific identity of one evaluation family."""

    seeds = [0, 1, 2] if seeds is None else list(seeds)
    _evaluation_paths(seeds)
    return {
        'schema': 'molvid.A.evaluation.v1',
        'checkpoint_sha256': checkpoint_sha256,
        'recipe_sha256': sha256_file(recipe),
        'store_index_sha256': sha256_file(store / 'index.txt'),
        'manifest_sha256': sha256_file(manifest),
        'steps': 16,
        'seeds': seeds,
        'limit': limit,
        'test_opened': False,
        'precision': 'fp32',
        'conditioning': 'observed prefix/topology/query times only',
        'aggregation': 'draw -> anchor -> replica -> system equal',
    }


def _evaluation_sample_ids(store: Path, *, limit: int | None = None) -> list[str]:
    ids = [
        line.split('\t', 1)[0]
        for line in (store / 'index.txt').read_text(encoding='utf-8').splitlines()
        if line
    ]
    return ids[:limit] if limit is not None else ids


def validate_completed_evaluation(
    output: Path,
    *,
    expected_protocol: dict,
    expected_family: str,
    store: Path,
    limit: int | None = None,
) -> dict:
    """Reject completed outputs not bound to the full frozen protocol/coverage."""

    protocol_path = output / 'protocol.json'
    summary_path = output / 'summary.json'
    rows_path = output / 'rows.jsonl'
    if not protocol_path.is_file() or not summary_path.is_file() or not rows_path.is_file():
        raise ValueError('completed evaluation is missing protocol, summary, or rows')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol != expected_protocol:
        raise ValueError('completed evaluation protocol differs from frozen inputs')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    protocol_sha256 = sha256_file(protocol_path)
    if summary.get('protocol_sha256') != protocol_sha256:
        raise ValueError('completed evaluation summary protocol SHA-256 differs')
    if summary.get('checkpoint_sha256') != expected_protocol['checkpoint_sha256']:
        raise ValueError('completed evaluation checkpoint differs')
    if summary.get('test_opened') is not False:
        raise ValueError('completed evaluation does not preserve the sealed-test boundary')
    if summary.get('family') != expected_family:
        raise ValueError('completed evaluation family differs')

    sample_ids = _evaluation_sample_ids(store, limit=limit)
    expected_paths = set(_evaluation_paths(expected_protocol['seeds']))
    expected_rows = len(sample_ids) * len(expected_paths)
    if int(summary.get('records', -1)) != len(sample_ids):
        raise ValueError('completed evaluation record coverage differs')
    if int(summary.get('rows', -1)) != expected_rows:
        raise ValueError('completed evaluation row coverage differs')
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding='utf-8').splitlines()
        if line
    ]
    if len(rows) != expected_rows:
        raise ValueError('completed evaluation rows file coverage differs')
    row_keys = [str(row.get('row_key', '')) for row in rows]
    if len(set(row_keys)) != expected_rows:
        raise ValueError('completed evaluation contains duplicate row keys')
    by_sample: dict[str, set[tuple[str, int | None]]] = {
        sample_id: set() for sample_id in sample_ids
    }
    for row in rows:
        key = str(row.get('row_key', ''))
        sample_id = key.split('|H', 1)[0]
        if sample_id not in by_sample:
            raise ValueError('completed evaluation contains an unexpected sample')
        family = row.get('metric_family', {}).get('family')
        if family != expected_family:
            raise ValueError('completed evaluation row family differs')
        by_sample[sample_id].add((str(row.get('path')), row.get('seed')))
        for directory, field in (
            ('coordinates', 'coordinate_file'),
            ('profiles', 'profile_file'),
        ):
            name = row.get(field)
            if not isinstance(name, str) or not (output / directory / name).is_file():
                raise ValueError(f'completed evaluation is missing {field}')
    if any(paths != expected_paths for paths in by_sample.values()):
        raise ValueError('completed evaluation path/seed coverage differs')
    if not (output / 'per_system.csv').is_file():
        raise ValueError('completed evaluation is missing per-system results')
    return {
        'protocol_sha256': protocol_sha256,
        'records': len(sample_ids),
        'rows': expected_rows,
        'family': expected_family,
        'test_opened': False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','recipe','store','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2],help='generated-path RNG seeds; persistence and oracle are always evaluated')
    p.add_argument('--stop-before-unix',type=float)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=_evaluation_paths(args.seeds)
    for name in ('coordinates','profiles'): (args.output/name).mkdir(exist_ok=True)
    start=time.time();device=configure_device('cuda',deterministic=True)
    raw=load_config('configs/frame_gm_p2_formal_GM_260923.yaml',schema='molvid.frame_joint.train.v1')
    loaded=load_frame_joint_inference(args.checkpoint,expected_sha256=args.checkpoint_sha256,codec_path=raw['codec']['checkpoint'],codec_sha256=raw['codec']['sha256'],device=device)
    model=loaded['model'].eval()
    recipe=json.loads(args.recipe.read_text());config=PhysicalAuxiliaryConfig.resolve(recipe['physical'])
    ds=ClipMMapDataset(args.store)
    protocol=evaluation_protocol(checkpoint_sha256=args.checkpoint_sha256,recipe=args.recipe,store=args.store,manifest=args.manifest,limit=args.limit,seeds=args.seeds)
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
    progress_path=args.output/'progress.json'
    def stop_at_budget_boundary(position, path, seed):
        if args.stop_before_unix is None or time.time()<args.stop_before_unix:return False
        atomic_write_json(progress_path,{'schema':'molvid.A.evaluation.progress.v1','protocol_sha256':sha256_file(protocol_path),
            'records_total':len(ids),'rows_completed':len(rows),'next_record_position':position,'next_path':path,'next_seed':seed,
            'elapsed_seconds_this_invocation':time.time()-start,'status':'budget_stopped','test_opened':False})
        return True
    try:
        for position,sid in enumerate(ids):
            if stop_at_budget_boundary(position,None,None):raise SystemExit(BUDGET_STOP_EXIT_CODE)
            record=ds[by_id[sid]]
            history=int(record['history_frames'])
            meta=meta_for(record)
            cpu=collate_clip_records([record]);batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=history)
            target=batch.coordinate_batch.x.float()
            frames=[history+i for i in torch.where(batch.query.frame_mask[0])[0].tolist()]
            family=metric_family(record,query_frames=len(frames))
            for path,seed in paths:
                key=f'{sid}|H{history}|{path}|seed={seed}'
                if key in completed:continue
                if stop_at_budget_boundary(position,path,seed):raise SystemExit(BUDGET_STOP_EXIT_CODE)
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
                atomic_write_json(progress_path,{'schema':'molvid.A.evaluation.progress.v1','protocol_sha256':sha256_file(protocol_path),
                    'records_total':len(ids),'rows_completed':len(rows),'last_row_key':key,
                    'elapsed_seconds_this_invocation':time.time()-start,'status':'running','test_opened':False})
            if (position+1)%8==0:print(json.dumps({'records':position+1,'total':len(ids),'rows':len(rows)}),flush=True)
        expected_rows=len(ids)*len(paths)
        if len(rows)!=expected_rows:raise RuntimeError(f'evaluation row coverage mismatch: expected {expected_rows}, got {len(rows)}')
        per_system,buckets=_aggregate(rows)
        for row in per_system:row['family']=rows[0]['metric_family']['family']
        _write_csv(args.output/'per_system.csv',per_system)
        atomic_write_json(args.output/'summary.json',{'family':rows[0]['metric_family']['family'],'bucket_summary':buckets,'records':len(ids),'rows':len(rows),
            'main_rmsf_available':rows[0]['metric_family']['family']=='ordinary','gpu_hours_this_invocation':(time.time()-start)/3600,
            'checkpoint_sha256':args.checkpoint_sha256,'protocol_sha256':sha256_file(protocol_path),'test_opened':False})
        atomic_write_json(progress_path,{'schema':'molvid.A.evaluation.progress.v1','protocol_sha256':sha256_file(protocol_path),
            'records_total':len(ids),'rows_completed':len(rows),'elapsed_seconds_this_invocation':time.time()-start,
            'status':'completed','test_opened':False})
    finally:ds.close()


if __name__=='__main__':main()
