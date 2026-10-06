"""Actual joint-coordinate forecasts on unchanged development inventories.

No projection/refinement or target conditioning. This experimental bridge
inherits released static PVB geometry; it is not the untouched PVB baseline.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import score, system_estimate
from molvid.evaluation.complex_motion import score_complex
from molvid.flow.pretrained_bridge import PretrainedGeometryBridge
from molvid.geometry.ca_detail import condition_from_record
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json
from molvid.training.bridge_runner import load_geometry
from molvid.training.predictive_data import RELEASE, atomic_torch_save


@torch.no_grad()
def forecast(model, observed_cpu, query_time_ps, device, seed):
    observed = observed_cpu.to(device)
    condition = model.prepare_condition(observed)
    generator = torch.Generator(device=device).manual_seed(seed)
    return model.sample(condition, query_time_ps.to(device), steps=10, generator=generator).float()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(8)
    torch.manual_seed(71)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved['step'] != saved['settings']['updates']:
            raise ValueError('Fixed-final model required for this comparison')
        settings = saved['settings']
    else:
        settings = json.loads(args.initial_config.read_text())
        saved = None
    model = PretrainedGeometryBridge(load_geometry(settings, device), recompute_frames=True).to(device)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
        del saved
    model.eval().requires_grad_(False)
    seeds = [17, 29, 43, 59, 71, 83, 97, 101] if args.domain == 'complex' else [17, 29]
    protocol = dict(variant=args.variant, domain=args.domain, history_frames=4, query_frames=12,
        training_query_frames=settings['data']['query_frames'], solver_steps=10, seeds=seeds,
        checkpoint=str(args.checkpoint), initial_config=str(args.initial_config),
        model=model.contract(), original_pvb_modified=False, inference_refinement=False,
        development=True, independent_test=False, test_payload_opened=False,
        limitations=['Inherited released static pretraining; not training from scratch',
            'Four training queries versus twelve joint evaluation queries',
            'No independent family/scaffold generalization claim', 'Short horizons do not establish thermodynamics'])
    atomic_write_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    rows = []
    if args.domain == 'complex':
        dataset = ClipMMapDataset(args.data/'valid')
        selected, counts = [], defaultdict(int)
        for index, fields in enumerate(dataset._clip_index_fields):
            sample = fields[0]
            if '_dt_80ps_' not in sample:
                continue
            system = sample.split('_')[1]
            if counts[system] < 2:
                selected.append(index); counts[system] += 1
        if len(counts) != 16 or len(selected) != 32:
            raise ValueError('Expected existing16-system32-condition development inventory')
        try:
            for position, index in enumerate(selected):
                record = dataset[index]
                batch = collate_clip_records([record])
                observed = slice_clip_frames(batch, 0, 4)
                values = torch.stack([forecast(model, observed, batch.time_ps[:, 4:], device, seed).cpu()
                                      for seed in seeds])
                # Persist expensive generation before CPU metric work.
                atomic_torch_save(args.output/f'coordinates_{position:04d}.pt',
                    dict(sample_id=record['sample_id'], seeds=seeds, coordinates=values, variant=args.variant))
                metrics = score_complex(values.numpy(), record)
                persistence = np.repeat(np.asarray(record['x'])[3:4], 12, axis=0)
                baseline = score_complex(np.stack([persistence, persistence]), record)
                rows.append(dict(system=record['system_id'], sample_id=record['sample_id'],
                                 metrics=metrics, persistence=baseline))
                atomic_write_json(args.output/'rows.json', rows)
                print(json.dumps(dict(conditions=len(rows), forecasts=len(rows)*len(seeds),
                    seconds=time.monotonic()-started, metrics=metrics)), flush=True)
            means = {key:float(np.mean([r['metrics'][key] for r in rows if r['metrics'][key] is not None]))
                     for key in rows[0]['metrics']}
            atomic_write_json(args.output/'summary.json', dict(status='completed', systems=16,
                conditions=32, forecasts=256, variant=args.variant, mean_metrics=means,
                seconds=time.monotonic()-started, training_updates=0))
        finally:
            dataset.close()
    else:
        manifest = json.loads((args.data/'complete.json').read_text())
        entries = [e for e in manifest['files'] if e['lag_ps'] == 100]
        if len(entries) != 96 or len({e['system'] for e in entries}) != 48:
            raise ValueError('Expected existing48-system96-condition development inventory')
        protocol['records'] = entries
        atomic_write_json(args.output/'protocol.json', protocol)
        release = json.loads(RELEASE.read_text())
        recipe = Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
        physical = PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
        for position, entry in enumerate(entries):
            record = torch.load(args.data/entry['path'], map_location='cpu', weights_only=False)['record']
            batch = collate_clip_records([record])
            observed = slice_clip_frames(batch, 0, 4)
            condition = condition_from_record(record).to(device)
            predictions = [dict(variant=args.variant, seed=seed,
                coordinates=forecast(model, observed, batch.time_ps[:, 4:], device, seed).cpu()) for seed in seeds]
            atomic_torch_save(args.output/f'coordinates_{position:04d}.pt',
                dict(sample_id=entry['sample_id'], predictions=predictions))
            for prediction in predictions:
                with torch.no_grad():
                    metrics = score(prediction['coordinates'].to(device), record, condition, physical)
                rows.append(dict(system=entry['system'], sample_id=entry['sample_id'], seed=prediction['seed'],
                                 variant=args.variant, metrics=metrics))
            atomic_write_json(args.output/'rows.json', rows)
            print(json.dumps(dict(completed_records=position+1, rows=len(rows),
                                  seconds=time.monotonic()-started)), flush=True)
        aggregates = defaultdict(list)
        for row in rows:
            for key, value in row['metrics'].items():
                aggregates[key].append((row['system'], value))
        atomic_write_json(args.output/'summary.json', dict(status='completed', systems=48, rows=len(rows),
            failures=[], seconds=time.monotonic()-started, neural_training_updates=0, test_opened=False,
            arms={args.variant+'|'+key:system_estimate(value) for key,value in aggregates.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--initial-config', type=Path)
    parser.add_argument('--domain', choices=['complex', 'protein'], required=True)
    for key in ('data', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--variant', required=True)
    run(parser.parse_args())
