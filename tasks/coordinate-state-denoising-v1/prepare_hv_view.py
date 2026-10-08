"""Expose all original A1 draws without mixing the historical projection arm."""
import argparse
import json
from pathlib import Path
from molvid.runtime import atomic_write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    summary=json.loads((args.source/'summary.json').read_text())
    protocol=json.loads((args.source/'protocol.json').read_text())
    rows=[r for r in json.loads((args.source/'rows.json').read_text()) if r['variant']=='a1']
    if summary['status']!='completed' or len(rows)!=192 or len({r['system'] for r in rows})!=48:
        raise ValueError('Complete original48-system A1 required')
    if {r['seed'] for r in rows}!={17,29} or len({r['sample_id'] for r in rows})!=96:
        raise ValueError('No seed/condition selection allowed')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol.update(variant='a1',source_view=str(args.source),
        view_rule='All a1 rows/draws; exclude only explicitly separate a1_projection arm; original tensors unchanged',
        inference_refinement=False)
    for index in range(96):
        source=(args.source/f'coordinates_{index:04d}.pt').resolve(strict=True)
        (args.output/f'coordinates_{index:04d}.pt').symlink_to(source)
    atomic_write_json(args.output/'protocol.json',protocol)
    atomic_write_json(args.output/'rows.json',rows)
    atomic_write_json(args.output/'summary.json',dict(status='completed',systems=48,rows=192,
        source_summary=str(args.source/'summary.json'),model_calls=0,new_forecasts=0,
        source_forecasts_reused=192,variant='a1',postprocessing=False))
