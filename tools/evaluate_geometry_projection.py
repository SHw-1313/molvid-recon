"""Registered training-set control: geometry repair without suppressing motion?"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from molvid.data.batch import collate_clip_records
from molvid.geometry.ca_detail import condition_from_record
from molvid.geometry.projection import ProjectionConfig,repair_coordinates
from molvid.geometry.types import StaticTopologyMetadata
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.evaluation.ca_detail import score,system_estimate
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,configure_device,sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import RELEASE,atomic_torch_save


def run(cache,output):
    if output.exists():raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True)
    grouped=defaultdict(list)
    for item in load_ca_cache(cache,'train'):grouped[item['metadata']['system']].append(item)
    items=[sorted(grouped[s],key=lambda x:x['record']['sample_id'])[0] for s in sorted(grouped)[:8]]
    release=json.loads(RELEASE.read_text());recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']:raise ValueError('recipe SHA changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    config=ProjectionConfig()
    freeze_run_identity(output,inputs=[cache/'complete.json',recipe],purpose='Observed-only geometry repair train diagnostic',
        protocol=dict(configuration=asdict(config),samples=[i['record']['sample_id'] for i in items],
            split='train',neural_training_updates=0,coordinates_optimized=True,test_opened=False,
            no_future_targets_in_repair=True,scope='parameter-free postprocessing baseline, not a learned architecture'))
    rows=[];started=time.monotonic()
    with ExecutionEvidence(output):
        for index,item in enumerate(items):
            if time.monotonic()-started>7200:raise TimeoutError('two-hour diagnostic budget')
            record=item['record'];c=condition_from_record(record).to(device)
            # Topology is fixed metadata; only observed x enters the repair interface.
            batch=collate_clip_records([record]).to(device);topology=StaticTopologyMetadata.from_batch(batch)
            original=item['generated'][0]['coordinates'].to(device)
            repaired,trace=repair_coordinates(original,observed=c.observed,topology=topology,
                loss_mask=batch.loss_mask,frame_mask=batch.frame_mask[:,4:],physical=physical,config=config)
            with torch.no_grad():
                for variant,coordinates in [('a1',original),('observed_geometry_projection',repaired)]:
                    rows.append(dict(system=item['metadata']['system'],sample_id=record['sample_id'],variant=variant,
                        metrics=score(coordinates,record,c,physical)))
            atomic_torch_save(output/f'projection_{index:02d}.pt',dict(coordinates=repaired.cpu(),trace=trace,sample_id=record['sample_id']))
            atomic_write_json(output/'rows.json',rows)
            print(json.dumps(dict(completed=index+1,total=len(items),seconds=time.monotonic()-started)),flush=True)
        aggregates=defaultdict(list)
        for row in rows:
            for metric,value in row['metrics'].items():aggregates[(row['variant'],metric)].append((row['system'],value))
        atomic_write_json(output/'summary.json',dict(status='completed',seconds=time.monotonic()-started,
            arms={'|'.join(k):system_estimate(v) for k,v in aggregates.items()},test_opened=False,
            systems=len(items),neural_training_updates=0,split='train',not_independent_validation=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.cache,args.output)
