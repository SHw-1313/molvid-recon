"""Reuse the established MD response metric, not the old RF-specific launcher.

Run from molvid-native-clock-training-20261005 using its existing analysis
package. No new generation, model imports, weight hashing, or protocol replay.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch

from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.ordered_motion import observed_contact_pairs, distance_series
from molvid.evaluation.time_response import clock_response
from molvid.geometry.ca_detail import condition_from_record
from molvid.runtime import atomic_write_json
from tools.audit_trained_clock_response import draws, matched_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('root', 'cache80', 'cache100', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directories = [args.root/'native80', args.root/'expanded48']
    inventories = []
    for directory in directories:
        summary = json.loads((directory/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['rows'] != 192:
            raise ValueError('Need all192 cached forecasts at each clock')
        protocol = json.loads((directory/'protocol.json').read_text())
        inventories.append({(e['system'], e['anchor']): (i,e) for i,e in enumerate(protocol['records'])})
    if inventories[0].keys() != inventories[1].keys() or len(inventories[0]) != 96:
        raise ValueError('Need96 matched system/anchor conditions')
    rows, grouped = [], defaultdict(list)
    for key in inventories[0]:
        records, predictions = [], []
        for directory, cache, inventory in zip(directories, [args.cache80,args.cache100], inventories):
            i, entry = inventory[key]
            records.append(torch.load(cache/entry['path'], map_location='cpu', weights_only=False)['record'])
            predictions.append(draws(directory/f'coordinates_{i:04d}.pt', entry, args.variant))
        matched_records(*records)
        condition = condition_from_record(records[0])
        ca = condition.ca_index.numpy()
        observed = condition.observed[-1,ca].numpy()
        pairs = observed_contact_pairs(observed)
        d0 = distance_series(observed[None], pairs)[0]
        pred = [np.stack([distance_series(draw.numpy()[:,ca], pairs) for draw in p]) for p in predictions]
        truth = [distance_series(np.asarray(r['x'])[4:,ca], pairs) for r in records]
        metrics = clock_response(*pred, *truth, d0)
        for target, response in [('displacement_error_A2','internal_displacement_A2'),
                                  ('fluctuation_error_A','internal_fluctuation_rms_A')]:
            zero = abs(metrics['MD_delta_'+response])
            metrics['zero_response_'+target] = zero
        metrics['candidate_minus_zero_displacement_error_A2'] = metrics['absolute_displacement_response_error_A2']-metrics['zero_response_displacement_error_A2']
        metrics['candidate_minus_zero_fluctuation_error_A'] = metrics['absolute_fluctuation_response_error_A']-metrics['zero_response_fluctuation_error_A']
        rows.append(dict(system=key[0], anchor=key[1], metrics=metrics))
        for name, value in metrics.items():
            if value is not None:
                grouped[name].append((key[0], value))
    atomic_write_json(args.output/'rows.json', rows)
    atomic_write_json(args.output/'summary.json', dict(status='completed', systems=48, conditions=96,
        variant=args.variant, model_calls=0, promotion=False,
        estimates={name:system_estimate(values) for name,values in grouped.items()},
        limitations=['Reused development48; marginal bootstrap CIs', 'Two model draws, one MD future per condition',
            'Matched query-index noise is not common-physical-time path coupling', 'No thermodynamic or kinetic claim']))
