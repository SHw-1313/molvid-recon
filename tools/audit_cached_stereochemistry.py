"""CPU-only chemistry rescoring of completed saved draws; no model calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.stereochemistry import ca_handedness
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    summary=json.loads((args.generated/'summary.json').read_text())
    if summary['status']!='completed': raise ValueError('completed prediction run required')
    protocol=json.loads((args.generated/'protocol.json').read_text())
    records=protocol['records']
    paths=[args.generated/f'coordinates_{index:04d}.pt' for index in range(len(records))]
    identity=freeze_run_identity(args.output,inputs=[args.generated/'summary.json',args.generated/'protocol.json']+paths,
        purpose='Observed-reference C-alpha chirality rescoring',protocol=dict(compute='CPU',
            model_calls=0,training_updates=0,test_opened=False,posthoc_diagnostic=True,
            no_geometry_threshold_selection_from_predictions=True))
    rows=[]
    for index,entry in enumerate(records):
        path=args.cache/entry['path']
        if sha256_file(path)!=entry['sha256']: raise ValueError('clip identity mismatch')
        record=torch.load(path,map_location='cpu',weights_only=False)['record']
        if sha256_file(paths[index])!=identity['input_sha256'][str(paths[index])]: raise ValueError('prediction changed')
        draws=torch.load(paths[index],map_location='cpu',weights_only=False)
        if draws['sample_id']!=entry['sample_id']: raise ValueError('prediction order changed')
        observation=torch.as_tensor(record['x'][:4]).float()
        values=[dict(variant='MD',seed=-1,coordinates=torch.as_tensor(record['x'][4:]).float())]+draws['predictions']
        for draw in values:
            rows.append(dict(system=entry['system'],sample_id=entry['sample_id'],variant=draw['variant'],seed=draw['seed'],
                metrics=ca_handedness(draw['coordinates'],observed=observation,
                    atom_identity=record['atom_identity'],block_id=record['block_id'])))
        atomic_write_json(args.output/'rows.json',rows)
    groups=defaultdict(list)
    for row in rows:
        for key,value in row['metrics'].items(): groups[(row['variant'],key)].append((row['system'],value))
    atomic_write_json(args.output/'summary.json',dict(status='completed',compute='CPU',model_calls=0,
        training_updates=0,arms={'|'.join(k):system_estimate(v) for k,v in groups.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
