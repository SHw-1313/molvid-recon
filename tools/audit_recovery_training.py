"""CPU-only completed checkpoint/exposure audit; never counted as training."""
import argparse
import json
from pathlib import Path
import torch
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file,canonical_hash


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    inputs=[];runs={};protocols={};metrics={}
    for seed in (71,72):
        for arm in ('static','mixed'):
            name=f'{arm}_{seed}';directory=args.training/name
            protocol=json.loads((directory/'protocol.json').read_text())
            summary=json.loads((directory/'summary.json').read_text())
            checkpoint=directory/'final.pt'
            if summary['status']!='completed' or summary['updates']!=2048: raise ValueError('unfinished formal arm')
            if sha256_file(checkpoint)!=summary['final_sha256']: raise ValueError('checkpoint SHA changed')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload['protocol_sha256']!=sha256_file(directory/'protocol.json'): raise ValueError('protocol changed')
            states=payload['optimizer']['state'];steps=sorted({int(value['step']) for value in states.values()})
            if steps!=[2048]: raise ValueError(f'optimizer steps disagree: {steps}')
            rows=[json.loads(line) for line in (directory/'train_metrics.jsonl').read_text().splitlines()]
            if [r['step'] for r in rows]!=list(range(1,2049)): raise ValueError('update ledger gap')
            if len({r['system'] for r in rows})!=192: raise ValueError('system coverage changed')
            inputs.extend([checkpoint,directory/'protocol.json',directory/'train_metrics.jsonl'])
            protocols[name]=protocol;metrics[name]=rows
            runs[name]=dict(parameters=protocol['parameters'],updates=2048,optimizer_parameter_states=len(states),
                optimizer_steps=steps,unique_static_systems=192,seconds=summary['seconds'],
                generated_aux_updates=summary['generated_updates'],parent_generator_updates=0,
                positive_gradient_steps=sum(r['grad_norm']>0 for r in rows),
                schedule_sha256=canonical_hash(protocol['static_schedule']),
                output_finite=all(torch.isfinite(v).all().item() for v in payload['model'].values()))
    pair_checks={}
    for seed in (71,72):
        left,right=f'static_{seed}',f'mixed_{seed}'
        same=protocols[left]['static_schedule']==protocols[right]['static_schedule']
        fields=('sample_id','frame','noise_sigma_A','static_coordinate','static_bond','static_loss','grad_norm')
        first_three=all(a[key]==b[key] for a,b in zip(metrics[left][:3],metrics[right][:3]) for key in fields)
        if not same or not first_three: raise ValueError('pre-intervention pairing failed')
        pair_checks[str(seed)]=dict(identical_static_schedule=same,first_three_updates_exact=first_three,
            first_intended_intervention_update=4)
    freeze_run_identity(args.output,inputs=inputs,purpose='Real Gcoord training and matched exposure audit',
        protocol=dict(compute='CPU',new_training_updates=0,new_model_inference_calls=0,test_opened=False))
    atomic_write_json(args.output/'summary.json',dict(status='completed',runs=runs,pair_checks=pair_checks,
        total_verified_formal_updates=8192,new_training_updates=0,new_model_calls=0,
        scope='171744-parameter geometry module, not full-generator training'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
