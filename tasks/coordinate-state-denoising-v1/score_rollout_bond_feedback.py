"""Cached whole-complex bond errors across generated-history rollout blocks.

Training targets are used only for evaluation. This observational diagnostic
does not identify the causal effect of generated versus clean history.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import system_estimate
from molvid.runtime import atomic_write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generated', type=Path, nargs='+', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    store = ClipMMapDataset(args.data / 'valid')
    rows = []
    try:
        for root in args.generated:
            summary = json.loads((root / 'summary.json').read_text())
            protocol = json.loads((root / 'protocol.json').read_text())
            if summary['status'] != 'completed' or summary['systems'] != 16 or summary['forecasts'] != 64:
                raise ValueError('Only complete16-system64-draw rollout panels')
            if protocol['future_frames'] != 92 or protocol['native_dt_ps'] != 80 or len(store) != 16:
                raise ValueError('Expected original92-frame80ps rollout inventory')
            for index in range(len(store)):
                record = store[index]
                saved = torch.load(root / f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
                if saved['sample_id'] != record['sample_id'] or saved['seeds'] != protocol['seeds']:
                    raise ValueError('Condition or draw identity mismatch')
                if saved['variant'] != protocol['variant']:
                    raise ValueError('Forecast variant mismatch')
                xyz = np.asarray(record['x'], dtype=np.float64)
                pred = np.asarray(saved['coordinates'], dtype=np.float64)
                if pred.shape != (4, 92, xyz.shape[1], 3) or not np.isfinite(pred).all():
                    raise ValueError('Complete finite raw trajectories required')
                component = np.asarray(record['component_id'])
                bonds = np.asarray(record['bond_index'])
                bonds = bonds[:, bonds[0] < bonds[1]]
                metrics = {}
                counts = {}
                for group in ('all', 'protein', 'ligand'):
                    select = np.ones(bonds.shape[1], dtype=bool)
                    if group != 'all':
                        cid = 0 if group == 'protein' else 1
                        select = (component[bonds[0]] == cid) & (component[bonds[1]] == cid)
                    i, j = bonds[:, select]
                    if not len(i):
                        raise ValueError('Missing required covalent group')
                    counts[group] = len(i)
                    true_length = np.linalg.norm(xyz[4:, i] - xyz[4:, j], axis=-1)
                    pred_length = np.linalg.norm(pred[:, :, i] - pred[:, :, j], axis=-1)
                    error2 = (pred_length - true_length[None]) ** 2
                    metrics[f'{group}.whole_RMSE_A'] = float(np.sqrt(error2.mean()))
                    for block, start in enumerate(range(0, 92, 12)):
                        stop = min(start + 12, 92)
                        # Initial history is observed. LaterH4 consists only of generated frames.
                        history = float(np.sqrt(error2[:, start - 4:start].mean())) if start else 0.
                        output = float(np.sqrt(error2[:, start:stop].mean()))
                        prefix = f'{group}.block{block}'
                        metrics[prefix + '.history_RMSE_A'] = history
                        metrics[prefix + '.output_RMSE_A'] = output
                        metrics[prefix + '.output_minus_history_A'] = output - history
                        metrics[prefix + '.first_RMSE_A'] = float(np.sqrt(error2[:, start].mean()))
                        metrics[prefix + '.last_RMSE_A'] = float(np.sqrt(error2[:, stop - 1].mean()))
                rows.append(dict(variant=protocol['variant'], system=record['system_id'],
                                 sample_id=record['sample_id'], bonds=counts, metrics=metrics))
    finally:
        store.close()
    estimates = {}
    for variant in sorted({r['variant'] for r in rows}):
        chosen = [r for r in rows if r['variant'] == variant]
        if len({r['system'] for r in chosen}) != 16:
            raise ValueError('Keep every registered system')
        estimates[variant] = {k: system_estimate([(r['system'], r['metrics'][k]) for r in chosen])
                              for k in chosen[0]['metrics']}
    atomic_write_json(args.output, dict(status='completed', model_calls=0, optimizer_updates=0,
        generated=[str(p) for p in args.generated], data=str(args.data), rows=rows, estimates=estimates,
        metric='Per-system RMSE over all four draws,selected frames and covalent edges versus same-time MD bond lengths;then equal-system estimates',
        convention='Rigid-transform-invariant raw bond distances,no alignment/projection or future-derived model inputs',
        limitations=['Descriptive development16 diagnostic,not an intervention on history quality',
                     'PVB uses nativeH1;its precedingH4 statistics are descriptive only,not its actual conditioning tensor',
                     'RMSE output-minus-history is not an additive decomposition or chemical force',
                     'No clash/chirality/angle coverage and no equilibrium or kinetic conclusion'], promotion=False))


if __name__ == '__main__':
    main()
