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
    if args.expected_step is not None and args.checkpoint is None:
        raise ValueError('An explicit checkpoint step requires a trained checkpoint')
    if args.reuse and (args.domain != 'complex' or args.complex_systems != 64):
        raise ValueError('Reuse is for the complex16 to complex64 extension only')
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(8)
    torch.manual_seed(71)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        settings = saved['settings']
        expected_step = settings['updates'] if args.expected_step is None else args.expected_step
        if not 0 < expected_step <= settings['updates'] or saved['step'] != expected_step:
            raise ValueError('Use the explicitly declared checkpoint step; default remains fixed-final')
        checkpoint_step = saved['step']
    else:
        settings = json.loads(args.initial_config.read_text())
        saved = None
        checkpoint_step = 0
    model = PretrainedGeometryBridge(load_geometry(settings, device), recompute_frames=True).to(device)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
        del saved
    model.eval().requires_grad_(False)
    seeds = [17, 29, 43, 59, 71, 83, 97, 101] if args.domain == 'complex' else [17, 29]
    protocol = dict(variant=args.variant, domain=args.domain, history_frames=4, query_frames=12,
        training_query_frames=settings['data']['query_frames'], solver_steps=10, seeds=seeds,
        checkpoint=str(args.checkpoint), initial_config=str(args.initial_config),
        training_checkpoint_step=checkpoint_step, planned_training_updates=settings['updates'],
        checkpoint_is_final=checkpoint_step == settings['updates'],
        intermediate_learning_curve=args.expected_step is not None and checkpoint_step != settings['updates'],
        model=model.contract(), original_pvb_modified=False, inference_refinement=False,
        development=True, independent_test=False, test_payload_opened=False,
        limitations=['Inherited released static pretraining; not training from scratch',
            'Four training queries versus twelve joint evaluation queries',
            'No independent family/scaffold generalization claim', 'Short horizons do not establish thermodynamics'])
    atomic_write_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    rows = []
    if args.domain == 'complex':
        inventory = None
        cached = {}
        if args.complex_systems == 64:
            inventory = json.loads((args.data/'summary.json').read_text())
            if inventory['status'] != 'completed' or len(inventory['systems']) != 64:
                raise ValueError('Completed official validation64 inventory required')
            protocol['strata'] = dict(old16=inventory['old16'], new48=inventory['new48'])
        if args.reuse:
            previous = json.loads((args.reuse/'protocol.json').read_text())
            completed = json.loads((args.reuse/'summary.json').read_text())
            for key in ('variant', 'domain', 'history_frames', 'query_frames', 'solver_steps', 'seeds', 'model'):
                if previous[key] != protocol[key]:
                    raise ValueError(f'Cached model/protocol differs: {key}')
            for key in ('checkpoint', 'initial_config'):
                if previous[key] != protocol[key]:
                    raise ValueError(f'Cached model source differs: {key}')
            if completed['status'] != 'completed' or completed['conditions'] != 32:
                raise ValueError('Completed original16 cache required')
            cached = {row['sample_id']: (index, row) for index, row in enumerate(
                json.loads((args.reuse/'rows.json').read_text()))}
            protocol['reuse'] = str(args.reuse)
        protocol['complex_systems'] = args.complex_systems
        atomic_write_json(args.output/'protocol.json', protocol)
        new_forecasts, reused_forecasts = 0, 0
        dataset = ClipMMapDataset(args.data/'valid')
        selected, counts = [], defaultdict(int)
        for index, fields in enumerate(dataset._clip_index_fields):
            sample = fields[0]
            if '_dt_80ps_' not in sample:
                continue
            system = sample.split('_')[1]
            if counts[system] < 2:
                selected.append(index); counts[system] += 1
        if len(counts) != args.complex_systems or len(selected) != 2*args.complex_systems:
            raise ValueError('Expected two native80ps conditions per selected development system')
        if inventory and set(counts) != set(inventory['old16']+inventory['new48']):
            raise ValueError('Selected systems differ from fixed validation64 inventory')
        try:
            for position, index in enumerate(selected):
                record = dataset[index]
                if record['sample_id'] in cached:
                    previous_index, row = cached[record['sample_id']]
                    source = args.reuse/f'coordinates_{previous_index:04d}.pt'
                    saved_coordinates = torch.load(source, map_location='cpu', weights_only=False)
                    if (saved_coordinates['sample_id'] != record['sample_id'] or
                            saved_coordinates['seeds'] != seeds or saved_coordinates['variant'] != args.variant):
                        raise ValueError('Cached coordinates differ from requested forecast')
                    (args.output/f'coordinates_{position:04d}.pt').symlink_to(source.resolve())
                    rows.append(row)
                    reused_forecasts += len(seeds)
                    del saved_coordinates
                    atomic_write_json(args.output/'rows.json', rows)
                    continue
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
                new_forecasts += len(seeds)
                atomic_write_json(args.output/'rows.json', rows)
                print(json.dumps(dict(conditions=len(rows), forecasts=len(rows)*len(seeds),
                    new_forecasts=new_forecasts, reused_forecasts=reused_forecasts,
                    seconds=time.monotonic()-started, metrics=metrics)), flush=True)
            means = {key:float(np.mean([r['metrics'][key] for r in rows if r['metrics'][key] is not None]))
                     for key in rows[0]['metrics']}
            strata = {}
            if inventory:
                for name, ids in [('old16', inventory['old16']), ('new48', inventory['new48']),
                                  ('all64', inventory['old16']+inventory['new48'])]:
                    grouped = defaultdict(list)
                    for row in rows:
                        if row['system'] in ids:
                            for key, value in row['metrics'].items():
                                if value is not None:
                                    grouped[key].append((row['system'], value))
                    strata[name] = {key:system_estimate(values) for key, values in grouped.items()}
            atomic_write_json(args.output/'summary.json', dict(status='completed', systems=len(counts),
                conditions=len(rows), forecasts=len(rows)*len(seeds), variant=args.variant, mean_metrics=means,
                new_forecasts=new_forecasts, reused_forecasts=reused_forecasts, strata=strata,
                seconds=time.monotonic()-started, training_updates=0,
                independent_test=False, test_payload_opened=False))
        finally:
            dataset.close()
    else:
        manifest = json.loads((args.data/'complete.json').read_text())
        entries = [e for e in manifest['files'] if e['lag_ps'] == args.lag_ps]
        if len(entries) != 96 or len({e['system'] for e in entries}) != 48:
            raise ValueError('Expected existing48-system96-condition development inventory')
        protocol['records'] = entries
        protocol['query_dt_ps'] = args.lag_ps
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
    parser.add_argument('--expected-step', type=int,
        help='Explicitly declared learning-curve checkpoint; omission still requires fixed-final weights')
    parser.add_argument('--domain', choices=['complex', 'protein'], required=True)
    for key in ('data', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--lag-ps', type=int, choices=[80, 100, 1000], default=100)
    parser.add_argument('--complex-systems', type=int, choices=[16, 64], default=16)
    parser.add_argument('--reuse', type=Path, help='Completed same-model original16 forecast directory')
    run(parser.parse_args())
