"""Free-running native80ps development paths, not teacher-forced clip stitching."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.bridge_rollout import recursive_bridge_forecast
from molvid.evaluation.ca_detail import system_estimate
from molvid.evaluation.complex_motion import score_complex, aligned_future
from molvid.evaluation.native_forecast import load_pvb, pvb_complex_observation, pvb_forecast
from molvid.flow.pretrained_bridge import PretrainedGeometryBridge
from molvid.flow.coordinate_state import (CoordinateStateDenoiser, COORDINATE_RECIPE_SCHEMAS,
                                          coordinate_checkpoint_schema)
from molvid.runtime import atomic_write_json
from molvid.training.bridge_runner import load_geometry
from molvid.training.predictive_data import atomic_torch_save


def temporal_steps(values, record):
    """Ligand COM increments, including the bridge's fixed12-frame boundaries."""
    mask = np.asarray(record['align_mask'], dtype=bool)
    ligand = np.asarray(record['component_id']) == 1
    reference = np.asarray(record['x'])[3]
    pred = np.stack([aligned_future(x, reference, mask) for x in values])
    truth = aligned_future(np.asarray(record['x'])[4:], reference, mask)
    pred_center, true_center = pred[:, :, ligand].mean(2), truth[:, ligand].mean(1)
    increments = np.linalg.norm(np.diff(pred_center, axis=1), axis=-1)
    true_increment = np.linalg.norm(np.diff(true_center, axis=0), axis=-1)
    boundary = np.arange(1, increments.shape[1]+1) % 12 == 0
    return dict(ligand_center_increment_mae_A=float(np.abs(increments-true_increment).mean()),
        predicted_boundary_increment_A=float(increments[:, boundary].mean()),
        predicted_nonboundary_increment_A=float(increments[:, ~boundary].mean()),
        MD_boundary_increment_A=float(true_increment[boundary].mean()),
        MD_nonboundary_increment_A=float(true_increment[~boundary].mean()))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(8)
    torch.manual_seed(71)
    protocol = json.loads((args.data/'summary.json').read_text())
    if protocol['status'] != 'completed' or protocol['test_payload_opened'] or protocol['future_frames'] != 92:
        raise ValueError('Use completed continuous development96 data')
    initialization_report = None
    if args.pvb_checkpoint:
        model = load_pvb(args.pvb_checkpoint, args.pvb_root, device)
        model_kind = 'unaltered_released_PVB_MISATO_native_H1_autoregressive'
        sampler_steps = 10
    else:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if saved['step'] != saved['settings']['updates']:
            raise ValueError('Fixed-final candidate required')
        coordinate_state = saved['settings'].get('schema') in COORDINATE_RECIPE_SCHEMAS
        if coordinate_state:
            if saved.get('schema') != coordinate_checkpoint_schema(saved['settings']):
                raise ValueError('Wrong coordinate-state checkpoint schema')
            model = CoordinateStateDenoiser(saved['settings']).to(device)
            initialization_report = saved.get('initialization_report')
            sampler_steps = saved['settings']['sampler_intervals']
        else:
            model = PretrainedGeometryBridge(load_geometry(saved['settings'], device), recompute_frames=True,
                temporal_insertion_layer=saved['settings'].get('temporal_insertion_layer')).to(device)
            sampler_steps = 10
        model.load_state_dict(saved['model'], strict=True)
        model.eval().requires_grad_(False)
        model_kind = model.contract()
        del saved
    seeds = [17,29,43,59]
    atomic_write_json(args.output/'protocol.json', dict(variant=args.variant, model=model_kind,
        coordinate_initialization_report=initialization_report,
        checkpoint=str(args.pvb_checkpoint or args.checkpoint), seeds=seeds, systems=16,
        observed_frames=1 if args.pvb_checkpoint else 4, future_frames=92, native_dt_ps=80,
        horizon_ps=7360, solver_steps=sampler_steps, learned_bridge_block_frames=12,
        vector_field_evaluations_per_block=2*sampler_steps-1 if isinstance(model, CoordinateStateDenoiser) else sampler_steps,
        reference_source_modified=False, teacher_forcing=False, projection=False,
        independent_test=False, test_payload_opened=False,
        limitations=['Reused development16, one starting condition and one MD future per system',
            'Finite7.36ns stability is not thermodynamic equilibrium or long-time kinetics',
            'PVB nativeH1 versus bridgeH4; training exposures differ',
            'Twelve-frame chunk boundaries are reported for all models, not only bridge']))
    store = ClipMMapDataset(args.data/'valid')
    rows, started = [], time.monotonic()
    try:
        for index in range(len(store)):
            record = store[index]
            batch = collate_clip_records([record])
            observed = slice_clip_frames(batch, 0, 4)
            if args.pvb_checkpoint:
                pvbatch, origin = pvb_complex_observation(Path(record['source_pdb']), observed,
                                                          record['atom_source_index'], device)
            predictions = []
            for seed in seeds:
                with torch.no_grad():
                    prediction = (pvb_forecast(model, pvbatch, origin, seed=seed, frames=92)
                        if args.pvb_checkpoint else recursive_bridge_forecast(
                            model, observed, batch.time_ps[:,4:], seed=seed, block_frames=12, steps=sampler_steps))
                predictions.append(prediction.cpu())
            values = torch.stack(predictions)
            atomic_torch_save(args.output/f'coordinates_{index:04d}.pt', dict(sample_id=record['sample_id'],
                seeds=seeds, coordinates=values, variant=args.variant))
            metrics = score_complex(values.numpy(), record)
            metrics.update(temporal_steps(values.numpy(), record))
            for label, start, stop in [('early',0,12), ('middle',36,48), ('late',80,92)]:
                subset = dict(record, x=np.concatenate((np.asarray(record['x'])[:4],
                                                       np.asarray(record['x'])[4+start:4+stop])))
                for key,value in score_complex(values[:,start:stop].numpy(), subset).items():
                    metrics[f'{label}.{key}'] = value
            rows.append(dict(system=record['system_id'], sample_id=record['sample_id'], metrics=metrics))
            atomic_write_json(args.output/'rows.json', rows)
            print(json.dumps(dict(completed_systems=len(rows), forecasts=4*len(rows),
                                  seconds=time.monotonic()-started)), flush=True)
        estimates = {}
        for key in rows[0]['metrics']:
            pairs = [(r['system'], r['metrics'][key]) for r in rows if r['metrics'][key] is not None]
            estimates[key] = system_estimate(pairs)
        atomic_write_json(args.output/'summary.json', dict(status='completed', systems=len(rows),
            forecasts=4*len(rows), future_frames=92, variant=args.variant,
            seconds=time.monotonic()-started, estimates=estimates, thermodynamic_claim=False))
    finally:
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('data','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--pvb-checkpoint', type=Path)
    parser.add_argument('--pvb-root', type=Path, default=Path('/workspace/PVB_origin'))
    parser.add_argument('--variant', required=True)
    run(parser.parse_args())
