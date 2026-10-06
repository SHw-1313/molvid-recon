"""Real full-complex CUDA generation, generic atoms rather than protein CA adapters."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.complex_motion import score_complex
from molvid.evaluation.native_forecast import coordinate_forecast
from molvid.flow.coordinate import CoordinateFlowModel
from molvid.provenance import freeze_run_identity
from molvid.runtime import configure_device,atomic_write_json
from molvid.training.predictive_data import load_parent,atomic_torch_save
from molvid.training.execution_evidence import ExecutionEvidence
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    summary=json.loads((args.data/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('complex data conversion incomplete')
    dataset=ClipMMapDataset(args.data/'valid')
    selected=[];counts={}
    for index,fields in enumerate(dataset._clip_index_fields):
        sample=fields[0]
        if '_dt_80ps_' not in sample:continue
        system=sample.split('_')[1]
        if counts.get(system,0)<2:
            selected.append(index);counts[system]=counts.get(system,0)+1
    if len(counts)!=16 or set(counts.values())!={2}:
        raise ValueError('two native80ps windows for each16 development complex required')
    device=configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.9)
    parent,_,_=load_parent(device)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    require_final_coordinate_checkpoint(saved)
    model=CoordinateFlowModel(parent,train_query_geometry=(saved['contract']['model']['query_encoder']==
        'trainable_query_copy_frozen_observation_current_radius_graph'),
        dit_depth=saved['contract']['model'].get('dit_depth',4))
    model.set_query_geometry_recompute('query_geometry_execution' in saved['contract']['model'])
    if model.contract()!=saved['contract']['model']:raise ValueError('incompatible model contract')
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    del saved,parent
    seeds=[17,29,43,59,71,83,97,101]
    freeze_run_identity(args.output,inputs=[args.checkpoint,args.data/'summary.json',args.data/'valid/index.txt'],
        purpose='Full-complex native80ps development forecasting',protocol=dict(forecasts_planned=256,
            training_updates=0,systems=16,conditions=32,history_frames=4,query_frames=12,
            native_dt_ps=80,inference_seeds=seeds,solver_steps=16,variant=args.variant,
            test_payload_opened=False,independent_test=False,pretraining_exposure='not presumed absent',
            limitations=['official validation, not family/scaffold independent','short horizon, not equilibrium',
                'single MD future per condition; no conditional MD variance claim']))
    rows=[];started=time.monotonic()
    try:
        with ExecutionEvidence(args.output):
            for position,index in enumerate(selected):
                record=dataset[index];batch=collate_clip_records([record])
                observed=slice_clip_frames(batch,0,4)
                predictions=[]
                for seed in seeds:
                    prediction=coordinate_forecast(model,observed,batch.time_ps[:,4:],device=device,seed=seed)
                    predictions.append(prediction.cpu())
                values=torch.stack(predictions)
                metrics=score_complex(values.numpy(),record)
                persistence=np.repeat(np.asarray(record['x'])[3:4],12,axis=0)
                baseline=score_complex(np.stack([persistence,persistence]),record)
                atomic_torch_save(args.output/f'coordinates_{position:04d}.pt',dict(sample_id=record['sample_id'],
                    seeds=seeds,coordinates=values,variant=args.variant))
                rows.append(dict(system=record['system_id'],sample_id=record['sample_id'],metrics=metrics,
                    persistence=baseline))
                atomic_write_json(args.output/'rows.json',rows)
                print(json.dumps(dict(conditions=len(rows),forecasts=len(rows)*len(seeds),
                    seconds=time.monotonic()-started,metrics=metrics)),flush=True)
        atomic_write_json(args.output/'summary.json',dict(status='completed',systems=16,conditions=32,
            forecasts=256,training_updates=0,variant=args.variant,seconds=time.monotonic()-started,
            mean_metrics={key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']},
            mean_persistence={key:float(np.mean([r['persistence'][key] for r in rows])) for key in rows[0]['persistence']}))
    finally:
        dataset.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('data','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',required=True)
    run(parser.parse_args())
