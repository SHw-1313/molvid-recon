"""Train-only causal source-bank intervention, explicitly inference-only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.ca_detail import score,system_estimate
from molvid.evaluation.native_forecast import molvid_forecast
from molvid.evaluation.stereochemistry import ca_handedness
from molvid.geometry.ca_detail import condition_from_record
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.provenance import freeze_run_identity
from molvid.runtime import configure_device,atomic_write_json,sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent,atomic_torch_save


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if args.wait_for:
        start=time.monotonic()
        while not args.wait_for.exists():
            if time.monotonic()-start>86400: raise TimeoutError('24h prerequisite budget')
            time.sleep(20)
        if json.loads(args.wait_for.read_text()).get('exception') is not None: raise RuntimeError('previous GPU job failed')
    device=configure_device('cuda',deterministic=True);torch.cuda.set_per_process_memory_fraction(.9)
    model,raw,release=load_parent(device)
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: raise ValueError('recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    groups=defaultdict(list)
    for item in load_ca_cache(args.cache,'train'): groups[item['metadata']['system']].append(item)
    items=[sorted(groups[s],key=lambda i:i['record']['sample_id'])[0] for s in sorted(groups)[:8]]
    variants={'original':None,'h_only':(.2,1.),'v_only':(1.,.2),'both':(.2,.2)}
    freeze_run_identity(args.output,inputs=[args.cache/'complete.json',recipe,Path(release['selected_checkpoint']['path'])],
        purpose='Train-only h/v source amplitude causal diagnostic',protocol=dict(variants=variants,seeds=[83,84],
            systems=[i['metadata']['system'] for i in items],draws=64,training_updates=0,test_opened=False,
            default_parity_gate='explicit(1,1) must match originalNone bit-for-bit',
            scope='changed inference distribution only; not a trained improvement or benchmark',
            trained_source_scales=[1.,1.],causal_hypothesis='isolate scalar versus vector source burden'))
    rows=[];started=time.monotonic()
    with ExecutionEvidence(args.output),torch.no_grad():
        for index,item in enumerate(items):
            if time.monotonic()-started>7200: raise TimeoutError('two-hour diagnostic budget')
            record=item['record'];cpu=collate_clip_records([record]);observed=slice_clip_frames(cpu,0,4)
            c=condition_from_record(record).to(device);saved=[]
            for seed in (83,84):
                for variant,scales in variants.items():
                    predicted=molvid_forecast(model,observed,cpu.time_ps[:,4:],device=device,seed=seed,
                        diagnostic_source_scales=scales)
                    if index==0 and seed==83 and variant=='original':
                        parity=molvid_forecast(model,observed,cpu.time_ps[:,4:],device=device,seed=seed,
                            diagnostic_source_scales=(1.,1.))
                        if not torch.equal(parity,predicted): raise RuntimeError('unit scales changed original forecast')
                        atomic_write_json(args.output/'default_parity.json',dict(exact_equal=True,extra_draws=1))
                    metrics=score(predicted,record,c,physical)
                    metrics.update(ca_handedness(predicted,observed=c.observed,
                        atom_identity=record['atom_identity'],block_id=record['block_id']))
                    rows.append(dict(system=item['metadata']['system'],sample_id=record['sample_id'],variant=variant,
                        seed=seed,metrics=metrics))
                    saved.append(dict(variant=variant,seed=seed,coordinates=predicted.cpu()))
            atomic_torch_save(args.output/f'coordinates_{index:02d}.pt',dict(sample_id=record['sample_id'],predictions=saved))
            atomic_write_json(args.output/'rows.json',rows)
            print(json.dumps(dict(systems=index+1,draws=len(rows),seconds=time.monotonic()-started)),flush=True)
        aggregates=defaultdict(list)
        for row in rows:
            for key,value in row['metrics'].items(): aggregates[(row['variant'],key)].append((row['system'],value))
        atomic_write_json(args.output/'summary.json',dict(status='completed',draws=64,parity_draws=1,
            seconds=time.monotonic()-started,training_updates=0,split='train',test_opened=False,
            arms={'|'.join(k):system_estimate(v) for k,v in aggregates.items()},not_a_trained_candidate=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-for',type=Path)
    run(p.parse_args())
