"""Frozen-rule all-frame clashes on complete cached generated-history rollouts.

CPU analysis only. No generation, coordinate modification, frame subsampling,
or model selection from partial panels.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.cached_clashes import CachedClashScorer
from molvid.evaluation.ca_detail import system_estimate
from molvid.geometry.types import StaticTopologyMetadata
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json
from molvid.training.predictive_data import RELEASE


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ('generated', 'data', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.generated / 'summary.json').read_text())
    protocol = json.loads((args.generated / 'protocol.json').read_text())
    if summary['status'] != 'completed' or summary['systems'] != 16 or summary['forecasts'] != 64:
        raise ValueError('Complete registered16-system64-draw rollout required')
    if protocol['future_frames'] != 92 or protocol['native_dt_ps'] != 80:
        raise ValueError('Expected original92-frame80ps protocol')
    release = json.loads(RELEASE.read_text())
    recipe = Path('/workspace/molvid-recon-gm-motion-v3') / release['auxiliary_recipe']['path']
    config = PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output / 'protocol.json', dict(
        generated=str(args.generated), data=str(args.data), physical=config.contract(),
        units='Angstrom', model_calls=0, optimizer_updates=0, raw_output=True,
        frames='All92 future frames and all4 draws; block12 with final block8',
        clash_rule='Same CachedClashScorer as complex64: graph-distance<=2 excluded; separate1-4 allowance; uncapped undirected pairs',
        normalization='Every category uses total valid complex atoms; categories are not normalized separately'))
    store = ClipMMapDataset(args.data / 'valid')
    rows = []
    started = time.monotonic()
    try:
        if len(store) != 16:
            raise ValueError('Expected original16-system inventory')
        for index in range(len(store)):
            record = store[index]
            saved = torch.load(args.generated / f'coordinates_{index:04d}.pt',
                               map_location='cpu', weights_only=False)
            if (saved['sample_id'] != record['sample_id'] or saved['seeds'] != protocol['seeds']
                    or saved['variant'] != protocol['variant']):
                raise ValueError('Condition/draw/variant mismatch')
            pred = np.asarray(saved['coordinates'])
            truth = np.asarray(record['x'])[4:]
            if pred.shape != (4, 92, truth.shape[1], 3) or truth.shape[0] != 92:
                raise ValueError('Complete raw future atom/frame set required')
            batch = collate_clip_records([record])
            scorer = CachedClashScorer(StaticTopologyMetadata.from_batch(batch), batch.loss_mask, config)
            metrics = {}
            for prefix, draws in (('', pred), ('MD.', truth[None])):
                # Score each frame once; aggregate only after all required frames.
                scored = [[scorer.frame(frame) for frame in draw] for draw in draws]
                for key in scored[0][0]:
                    values = np.asarray([[frame[key] for frame in draw] for draw in scored])
                    metrics[prefix + 'whole.' + key] = float(values.mean())
                    for block, start in enumerate(range(0, 92, 12)):
                        metrics[prefix + f'block{block}.' + key] = float(values[:, start:start + 12].mean())
            rows.append(dict(system=record['system_id'], sample_id=record['sample_id'], metrics=metrics))
            atomic_write_json(args.output / 'rows.json', rows)
            print(json.dumps(dict(completed_systems=len(rows), seconds=time.monotonic()-started,
                                  model_calls=0)), flush=True)
    finally:
        store.close()
    if len({row['system'] for row in rows}) != 16:
        raise ValueError('Keep all registered systems')
    estimates = {key: system_estimate([(row['system'], row['metrics'][key]) for row in rows])
                 for key in rows[0]['metrics']}
    atomic_write_json(args.output / 'summary.json', dict(
        status='completed', variant=protocol['variant'], systems=16, conditions=16,
        forecasts=64, future_frames=92, estimates=estimates, model_calls=0,
        optimizer_updates=0, seconds=time.monotonic()-started,
        limitations=['Reused development16; exploratory system bootstrap, not sealed confirmation',
                     'Frozen legacy TRAIN radius table, not a force-field energy calculation',
                     'PVB nativeH1 versus candidateH4; training/input resources differ',
                     'Clashes must be interpreted jointly with bond geometry and motion',
                     'No angle/chirality/equilibrium/long-time kinetic conclusion'], promotion=False))


if __name__ == '__main__':
    main()
