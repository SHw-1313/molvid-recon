"""One capacity arm/case using the existing observed-only Flow forecast entry."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.training.predictive_data import atomic_torch_save
from tools.evaluate_pretrained_bridge import forecast


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True)
    for name in ('record','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or not torch.cuda.is_available():
        raise RuntimeError('Existing torch-ito and real authorized CUDA required')
    torch.set_num_threads(2)
    raw=torch.load(a.record,map_location='cpu',weights_only=False); record=raw.get('record',raw)
    if record.get('history_frames')!=4 or record.get('future_frames')!=50 or len(record['x'])!=54:
        raise ValueError('Only prepared H4/Q50 records')
    batch=collate_clip_records([record]); observed=slice_clip_frames(batch,0,4)
    query=batch.time_ps[:,4:]; relative=query[0]-batch.time_ps[0,3]
    dt={'atlas':100.,'misato':80.}[record['source']]
    if not torch.allclose(relative,torch.arange(1,51,dtype=relative.dtype)*dt):
        raise ValueError('Unchanged native physical clocks required')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=saved['settings']
    if saved['step']!=4096 or cfg['updates']!=4096 or cfg['scheme']!='flow' or cfg.get('query_qk_mode')!='group':
        raise ValueError('Expected completed capacity MD4096 group-QK Flow checkpoint')
    parent=saved['initialization_report']['parent_checkpoint']
    expected=a.checkpoint.parent.parent/'static'/'coordinate_final.pt'
    if Path(parent).resolve()!=expected.resolve(): raise ValueError('Unexpected arm-specific static parent')
    model=CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True)
    a.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    result=dict(model='capacity_'+a.arm,arm=a.arm,status='started',checkpoint=str(a.checkpoint.resolve()),
        training_parent=parent,source_commit=saved.get('source_commit'),capacity=model.capacity,
        parameters=model.contract()['parameters'],training_updates=4096,static_updates=4096,
        record=str(a.record.resolve()),history_frames=4,query_frames=50,query_dt_ps=dt,
        query_time_ps=relative.tolist(),seeds=[17,29],coordinate_unit='angstrom',raw=True,
        sampler_intervals=18,planned_vector_field_evaluations_per_draw=35,source_rho_A=model.rho_A,
        CUDA_device=torch.cuda.get_device_name(0),draws={})
    del saved
    def status(**kwargs):
        result.update(kwargs,elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        (a.output/'protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    status()
    try:
        for seed in (17,29):
            before=model.forward_calls; xyz=forecast(model,observed,query,torch.device('cuda:0'),seed)
            torch.cuda.synchronize(); nfe=model.forward_calls-before
            atomic_torch_save(a.output/f'coordinates_seed{seed}.pt',dict(record=record,
                sample_id=record['sample_id'],atom_identity=record['atom_identity'],coordinate_unit='angstrom',
                history_frames=4,query_time_ps=relative,predictions=[dict(seed=seed,coordinates=xyz.cpu())]))
            result['draws'][str(seed)]={'actual_vector_field_evaluations':nfe}
            status(last_saved_seed=seed,actual_nfe=nfe)
            if xyz.shape!=(50,batch.atom_count,3) or not torch.isfinite(xyz).all() or nfe!=35:
                raise ValueError('Raw output retained; invalid shape/finiteness/NFE')
        status(status='completed')
    except Exception as e:
        status(status='failed',error=repr(e));raise

if __name__=='__main__': main()
