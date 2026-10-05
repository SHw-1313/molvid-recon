"""Post-pilot, preregistered local-frame control: inference only, no training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from molvid.geometry.ca_detail import condition_from_record,ca_frame_transport
from molvid.evaluation.ca_detail import score,system_estimate
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import RELEASE
from molvid.runtime import configure_device,atomic_write_json,sha256_file
from molvid.provenance import freeze_run_identity
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig


@torch.no_grad()
def evaluate(cache,output):
    if output.exists():raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True);items=load_ca_cache(cache,'valid')
    release=json.loads(RELEASE.read_text());recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']:raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    freeze_run_identity(output,inputs=[cache/'complete.json',recipe],purpose='CA local-frame follow-up control',protocol=dict(
        rationale='Initial translation-based recovery worsens inter-residue bonds; test missing residue orientation explicitly',
        extra_training_updates=0,extra_A1_draws=0,validation_interventions=48,max_wall_seconds=1800,test_opened=False))
    with ExecutionEvidence(output):
        rows=[];start=time.monotonic()
        for item in items:
            if time.monotonic()-start>1800:raise TimeoutError('frame-control budget')
            record=item['record'];c=condition_from_record(record).to(device)
            candidates=[('frame_oracle',-1,torch.as_tensor(record['x'][4:]).to(device)[:,c.ca_index])]
            candidates += [('frame_generated',draw['seed'],draw['coordinates'].to(device)[:,c.ca_index]) for draw in item['generated']]
            for variant,seed,ca in candidates:
                prediction,info=ca_frame_transport(c,ca)
                rows.append(dict(system=item['metadata']['system'],sample_id=record['sample_id'],variant=variant,source_seed=seed,
                    frame_info=info,metrics=score(prediction,record,c,physical)))
            atomic_write_json(output/'rows.json',rows)
        arms={}
        for variant in ('frame_generated','frame_oracle'):
            selected=[r for r in rows if r['variant']==variant]
            arms[variant]={k:system_estimate([(r['system'],r['metrics'][k]) for r in selected]) for k in selected[0]['metrics']}
        atomic_write_json(output/'summary.json',dict(status='completed',arms=arms,rows=len(rows),seconds=time.monotonic()-start,
            test_opened=False,extra_training_updates=0,scope='post-pilot diagnosis, not an independently preregistered confirmation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.cache,a.output)
