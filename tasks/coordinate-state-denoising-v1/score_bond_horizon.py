"""Per-query covalent errors from complete cached forecasts; no model calls."""
import argparse
import json
from pathlib import Path

import torch

from molvid.data.batch import collate_clip_records
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.geometry import bond_errors
from molvid.runtime import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    panels = {}
    for root in args.generated:
        summary = json.loads((root/'summary.json').read_text())
        protocol = json.loads((root/'protocol.json').read_text())
        if summary['status'] != 'completed' or summary['systems'] != 48:
            raise ValueError('Complete48-system protein panel required')
        key = f"{protocol['variant']}|{protocol['query_dt_ps']}ps"
        if key in panels:
            raise ValueError('Duplicate variant/time panel')
        panel_rows = []
        for index, entry in enumerate(protocol['records']):
            record = torch.load(args.cache/entry['path'], map_location='cpu', weights_only=False)['record']
            batch = collate_clip_records([record])
            saved = torch.load(root/f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
            if saved['sample_id'] != entry['sample_id']:
                raise ValueError('Wrong condition')
            predictions = [p for p in saved['predictions'] if p['variant'] == protocol['variant']]
            if [p['seed'] for p in predictions] != protocol['seeds']:
                raise ValueError('Missing declared draws')
            for prediction in predictions:
                future = torch.as_tensor(prediction['coordinates']).float()
                if future.shape != batch.x[4:].shape or not torch.isfinite(future).all():
                    raise ValueError('Incomplete forecast')
                xyz = torch.cat((batch.x[:4].float(), future))
                for frame in range(4, len(xyz)):
                    panel_rows.append(dict(panel=key, system=entry['system'], sample_id=entry['sample_id'],
                        seed=prediction['seed'], query_index=frame-3,
                        lag_ps=float(batch.time_ps[0, frame]-batch.time_ps[0, 3]),
                        bond_RMSE_A=bond_errors(xyz, batch.x.float(), batch, [frame])['bond_rmse']))
        panels[key] = dict(source=str(root), conditions=len(protocol['records']),
            estimates={str(q): system_estimate([(r['system'], r['bond_RMSE_A'])
                for r in panel_rows if r['query_index'] == q]) for q in range(1, 13)})
        rows.extend(panel_rows)
        print(json.dumps(dict(panel=key, query_rows=len(panel_rows), model_calls=0)), flush=True)
    atomic_write_json(args.output, dict(status='completed', model_calls=0, panels=panels, rows=rows,
        metric='Same bond_errors function and loss/frame masks as primary evaluation, separately per physical query',
        limitations=['Mean per-frame RMSE is not the pooled-window RMSE',
            'Different query grids have different MD futures, not a controlled time-feature intervention',
            'Marginal system bootstrap; no claim about causality, full geometry or dynamics']))


if __name__ == '__main__':
    main()
