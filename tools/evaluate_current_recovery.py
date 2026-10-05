"""Same frozen A1 draws for every Gcoord arm; no extra generator calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import torch

from molvid.data.batch import collate_clip_records
from molvid.evaluation.ca_detail import score,system_estimate
from molvid.geometry.ca_detail import condition_from_record
from molvid.geometry.current_recovery import CurrentGeometryRecovery,RecoveryConfig
from molvid.geometry.types import StaticTopologyMetadata
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.provenance import freeze_run_identity
from molvid.runtime import configure_device,atomic_write_json,sha256_file
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import RELEASE,atomic_torch_save


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    source_summary=json.loads((args.generated/'summary.json').read_text())
    if source_summary['status']!='completed' or source_summary['systems']!=48 or source_summary['pilot']:
        raise ValueError('complete48-system A1 generation required')
    protocol=json.loads((args.generated/'protocol.json').read_text())
    if protocol['arm']!='a1': raise ValueError('wrong generator')
    device=configure_device('cuda',deterministic=True);torch.cuda.set_per_process_memory_fraction(.85)
    inputs=[args.generated/'summary.json',args.generated/'protocol.json',args.cache/'complete.json'];models={}
    for arm in ('static','mixed'):
        for seed in (71,72):
            directory=args.training/f'{arm}_{seed}';checkpoint=directory/'final.pt'
            summary=json.loads((directory/'summary.json').read_text())
            if summary['status']!='completed' or summary['updates']!=2048: raise ValueError('formal2048 only')
            if sha256_file(checkpoint)!=summary['final_sha256']: raise ValueError('model identity changed')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload['protocol_sha256']!=sha256_file(directory/'protocol.json'): raise ValueError('training protocol changed')
            model=CurrentGeometryRecovery(RecoveryConfig(**payload['configuration'])).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
            models[f'{arm}_{seed}']=model;inputs.extend((checkpoint,directory/'protocol.json'))
    coordinates=[args.generated/f'coordinates_{index:04d}.pt' for index in range(len(protocol['records']))]
    inputs+=coordinates
    release=json.loads(RELEASE.read_text());recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical']);inputs.append(recipe)
    identity=freeze_run_identity(args.output,inputs=inputs,purpose='Matched Gcoord48 development evaluation',
        protocol=dict(models=list(models),generator='frozen A1 Euler16 cached actual draws',
            generator_calls=0,training_updates=0,test_opened=False,confirmation_opened=False,
            checkpoint_selection='all final2048, no best-valid choice',development_only=True))
    rows=[];started=time.monotonic()
    with ExecutionEvidence(args.output),torch.no_grad():
        for index,entry in enumerate(protocol['records']):
            if time.monotonic()-started>21600: raise TimeoutError('six-hour evaluation budget')
            if shutil.disk_usage(args.output).free<8*1024**3: raise RuntimeError('8GiB disk reserve')
            path=args.cache/entry['path']
            if sha256_file(path)!=entry['sha256']: raise ValueError('source clip changed')
            record=torch.load(path,map_location='cpu',weights_only=False)['record']
            c=condition_from_record(record).to(device);batch=collate_clip_records([record]).to(device)
            topology=StaticTopologyMetadata.from_batch(batch)
            draw_path=coordinates[index]
            if sha256_file(draw_path)!=identity['input_sha256'][str(draw_path)]: raise ValueError('draw changed')
            draws=torch.load(draw_path,map_location='cpu',weights_only=False)
            if draws['sample_id']!=entry['sample_id']: raise ValueError('sample mismatch')
            saved=[]
            for draw in draws['predictions']:
                if draw['variant']!='a1': continue
                original=draw['coordinates'].to(device)
                for name,model in models.items():
                    predicted=model(original,topology=topology,frame_mask=batch.frame_mask[:,4:])
                    if not torch.isfinite(predicted).all(): raise FloatingPointError('nonfinite geometry output')
                    rows.append(dict(system=entry['system'],sample_id=entry['sample_id'],seed=draw['seed'],
                        variant=name,metrics=score(predicted,record,c,physical)))
                    saved.append(dict(variant=name,seed=draw['seed'],coordinates=predicted.cpu()))
            atomic_torch_save(args.output/f'coordinates_{index:04d}.pt',dict(sample_id=entry['sample_id'],predictions=saved))
            atomic_write_json(args.output/'rows.json',rows)
            print(json.dumps(dict(records=index+1,total=len(coordinates),rows=len(rows),seconds=time.monotonic()-started)),flush=True)
        aggregates=defaultdict(list)
        for row in rows:
            for metric,value in row['metrics'].items(): aggregates[(row['variant'],metric)].append((row['system'],value))
        atomic_write_json(args.output/'summary.json',dict(status='completed',systems=len({r['system'] for r in rows}),
            rows=len(rows),seconds=time.monotonic()-started,generator_calls=0,training_updates=0,
            arms={'|'.join(k):system_estimate(v) for k,v in aggregates.items()},test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
