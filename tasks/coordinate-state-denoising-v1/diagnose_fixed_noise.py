"""Fixed-weight supervised diagnostic, explicitly NOT source-sampled efficacy.

Same H4/Q12 records and epsilon draws for both recipes. The native RF/EDM
intermediate distributions differ; bin positions and task losses are not
matched-noise causal comparisons. No predictions here count as forecasts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.runtime import atomic_write_json


def spaced_ids(ids):
    ordered = sorted(set(ids))
    if len(ordered) < 8:
        raise ValueError('Require eight distinct systems per domain')
    return [ordered[i] for i in np.linspace(0, len(ordered)-1, 8, dtype=int)]


def records(protein, complex_data):
    entries = [e for e in json.loads((protein/'complete.json').read_text())['files']
               if e['lag_ps'] == 100]
    for system in spaced_ids(e['system'] for e in entries):
        entry = sorted((e for e in entries if e['system'] == system),
                       key=lambda e: e['sample_id'])[0]
        yield 'protein', torch.load(protein/entry['path'], map_location='cpu', weights_only=False)['record']
    inventory = json.loads((complex_data/'summary.json').read_text())
    selected = spaced_ids(inventory['new48'])
    store = ClipMMapDataset(complex_data/'valid')
    try:
        for system in selected:
            candidates = [(fields[0], i) for i, fields in enumerate(store._clip_index_fields)
                          if fields[0].split('_')[1] == system and '_dt_80ps_' in fields[0]]
            if not candidates:
                raise ValueError(f'Missing complex condition: {system}')
            yield 'complex', store[min(candidates)[1]]
    finally:
        store.close()


@torch.no_grad()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU model fallback')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(71)
    selected = list(records(args.protein, args.complex_data))
    protocol = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        checkpoint_paths={s: str(getattr(args, s)) for s in ['flow', 'edm']},
        selection='Eight evenly spaced sorted protein development IDs and eight complex new48 IDs; lexically first native anchor; no metric-based selection',
        records=[dict(domain=d, system=r['system_id'], sample=r['sample_id']) for d,r in selected],
        noise_seeds=[17,29], flow_s=[.05,.275,.5,.725,.95], edm_grid='Five logarithmically spaced TRAIN sigma bounds',
        history=4, queries=12, expected_model_calls=320, new_forecasts=0, optimizer_updates=0,
        condition='Observed history and chemistry only; clean future used solely for legitimate interpolation/noising and scoring',
        limitations=['Supervised noisy/interpolated inputs contain future information, unlike primary source sampling.',
            'A clean estimate X_s+(1-s)u is diagnostic, not an integrated trajectory; near s1 it is close to Y by construction.',
            'B clean estimate X_ref+D is not the next noisy sampler state.',
            'No cross-recipe ranking by task loss or equal bin index; intermediate distributions differ.',
            'Small reused development subset, not primary panel, equilibrium evidence or independent confirmation.',
            'Full12-frame window joint calls; each call rebuilds12 candidate-frame geometries.'])
    atomic_write_json(args.output/'protocol.json', protocol)
    began=time.monotonic()
    rows=[]
    resources={}
    for scheme in ['flow','edm']:
        saved=torch.load(getattr(args,scheme),map_location='cpu',weights_only=False)
        settings=saved['settings']
        if (saved.get('schema') != 'molvid.coordinate_state.checkpoint.v1'
                or settings['scheme'] != scheme or saved['step'] != settings['updates']):
            raise ValueError('Require the corresponding complete fixed-final coordinate checkpoint')
        model=CoordinateStateDenoiser(settings).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True)
        del saved
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        levels=protocol['flow_s'] if scheme=='flow' else np.geomspace(model.sigma_min,model.sigma_max,5).tolist()
        for domain,record in selected:
            batch=collate_clip_records([record]).to('cuda')
            condition=model.prepare_condition(slice_clip_frames(batch,0,4))
            target,times=batch.x[4:],batch.time_ps[:,4:]
            reference=condition.reference_A[None]
            valid=batch.loss_mask
            bonds=condition.bonds
            bond_valid=valid[bonds[0]] & valid[bonds[1]]
            i,j=bonds[:,bond_valid]
            target_lengths=(target[:,i]-target[:,j]).norm(dim=-1)
            mse=lambda x: float(x[:,valid].float().square().mean())
            for seed in protocol['noise_seeds']:
                generator=torch.Generator(device='cuda').manual_seed(seed)
                epsilon=torch.randn(target.shape,device='cuda',generator=generator)
                for level in levels:
                    progress=target.new_tensor([level])
                    if scheme=='flow':
                        source=reference+model.rho_A*epsilon
                        current=(1-level)*source+level*target
                        native_target=target-source
                        weight=1/model.sigma_data**2
                    else:
                        current=target+level*epsilon
                        native_target=target-reference
                        weight=(level**2+model.sigma_data**2)/(level*model.sigma_data)**2
                    prediction=model(current,condition=condition,query_time_ps=times,progress=progress)
                    clean=current+(1-level)*prediction if scheme=='flow' else reference+prediction
                    raw_mse=mse(prediction-native_target)
                    if not bool(torch.isfinite(clean).all()):
                        raise FloatingPointError('Nonfinite diagnostic clean estimate')
                    rows.append(dict(scheme=scheme,domain=domain,system=record['system_id'],
                        sample=record['sample_id'],noise_seed=seed,progress=level,
                        raw_native_target_MSE_A2=raw_mse,weighted_native_objective=weight*raw_mse,
                        clean_coordinate_component_RMSE_A=mse(clean-target)**.5,
                        input_coordinate_component_RMSE_A=mse(current-target)**.5,
                        persistence_coordinate_component_RMSE_A=mse(reference-target)**.5,
                        bond_length_RMSE_A=float(((clean[:,i]-clean[:,j]).norm(dim=-1)-target_lengths).square().mean().sqrt())))
            atomic_write_json(args.output/'rows.json',rows)
            print(json.dumps(dict(scheme=scheme,system=record['system_id'],diagnostic_rows=len(rows),new_forecasts=0)),flush=True)
        torch.cuda.synchronize()
        resources[scheme]=dict(seconds=time.monotonic()-start,model_calls=model.forward_calls,
            candidate_geometry_frames=model.geometry_calls-4*len(selected),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        del model,condition,prediction,clean,batch,target,current,epsilon
        torch.cuda.empty_cache()
    atomic_write_json(args.output/'summary.json',dict(status='completed',rows=len(rows),resources=resources,
        seconds=time.monotonic()-began,new_forecasts=0,optimizer_updates=0,primary_efficacy=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['flow','edm','protein','complex-data','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
