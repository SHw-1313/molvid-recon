"""One atom/group H4/Q50 condition using the existing Flow sampler.

The MAIN CLI combines a fixed static_A label with PVB. This limited entry keeps
its Flow path and adds explicit mode provenance and passive query-residual sums.
No device selection, training, scoring or additional diagnostic generation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.training.predictive_data import atomic_torch_save
from tools.evaluate_pretrained_bridge import forecast

PARENT = Path('/data1/sihao/molvid-derived/coordinate-static-pretrain-v1/seed71/flow/static/coordinate_final.pt')


class QueryResidualSums:
    """Detached component-wise feature RMS; no retained features or RNG calls."""
    def __init__(self, model):
        self.progress = None
        self.totals = None
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.temporal.query_attention.register_forward_hook(self.after)]

    def reset(self):
        self.progress = None
        self.totals = None

    def before(self, module, inputs, kwargs):
        progress = kwargs['progress'].detach()
        if progress.numel() != 1:
            raise ValueError('Residual summaries require this single-system short-generation entry')
        self.progress = progress[0]

    def after(self, module, inputs, output):
        h, v = inputs[0], inputs[2]
        dh, dv = output
        # Query attention returns dh/dv before the residual addition. Accumulate
        # feature sum-of-squares on device; only tiny totals move to CPU at save.
        sums = torch.stack([x.detach().float().square().sum() for x in (h, dh, v, dv)]).double()
        values = torch.cat((sums, sums.new_tensor([h.numel(), v.numel(), 1.])))
        s = self.progress
        buckets = torch.stack((s < 1/3, (s >= 1/3) & (s < 2/3), s >= 2/3)).to(values.dtype)
        if self.totals is None:
            self.totals = values.new_zeros((3, 7))
        self.totals.add_(buckets[:, None] * values[None])
        # Returning None preserves the original output object and values.

    def summary(self):
        result = dict(units='latent_feature_components; vector RMS includes xyz components',
            scope='all atoms/query frames/channels; no feature tensors retained',
            bins='early [0,1/3), middle [1/3,2/3), late [2/3,1]', buckets={})
        if self.totals is None:
            return result
        for name, row in zip(('early','middle','late'), self.totals.detach().cpu().tolist()):
            hi, ho, vi, vo, hn, vn, calls = row
            values = dict(forward_calls=int(calls), scalar_input_sum_square=hi, scalar_residual_sum_square=ho,
                vector_input_sum_square=vi, vector_residual_sum_square=vo,
                scalar_component_count=int(hn), vector_component_count=int(vn),
                scalar_input_rms=math.sqrt(hi/hn) if hn else None,
                scalar_residual_rms=math.sqrt(ho/hn) if hn else None,
                vector_input_rms=math.sqrt(vi/vn) if vn else None,
                vector_residual_rms=math.sqrt(vo/vn) if vn else None,
                scalar_relative_rms=math.sqrt(ho/hi) if hi > 0 else None,
                vector_relative_rms=math.sqrt(vo/vi) if vi > 0 else None)
            result['buckets'][name] = {k:v if v is None or math.isfinite(v) else None for k,v in values.items()}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('atom','group'), required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or not torch.cuda.is_available():
        raise RuntimeError('Use enter-container/torch-ito and authorized CUDA; no CPU model fallback')
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = torch.load(args.record, map_location='cpu', weights_only=False)
    record = raw.get('record',raw)
    if record.get('history_frames')!=4 or record.get('future_frames')!=50 or len(record['x'])!=54:
        raise ValueError('Use the existing prepared H4/Q50 record')
    batch = collate_clip_records([record])
    observed = slice_clip_frames(batch,0,4)
    query = batch.time_ps[:,4:]
    relative = query[0]-batch.time_ps[0,3]
    dt = {'atlas':100.,'misato':80.}.get(record['source'])
    if dt is None or not torch.allclose(relative,torch.arange(1,51,dtype=relative.dtype)*dt):
        raise ValueError('Expected unchanged ATLAS100ps / MISATO80ps query times')
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    started = time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(model='Flow-'+args.arm,query_qk_mode=args.arm,
        checkpoint=str(args.checkpoint.resolve()),declared_training_parent_checkpoint=str(PARENT),
        record=str(args.record.resolve()),history_frames=4,query_frames=50,query_dt_ps=dt,
        query_time_ps=relative.tolist(),seeds=[17,29],coordinate_unit='angstrom',raw=True,
        generation_training_updates=0,device=torch.cuda.get_device_name(0),residual_summary={})
    def status(**values):
        elapsed=time.monotonic()-started
        protocol.update(values,elapsed_seconds=elapsed,GPU_hours=elapsed/3600)
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    status(status='started')
    hooks = None
    try:
        saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        settings = saved['settings']
        actual_parent = saved.get('initialization_report',{}).get('parent_checkpoint')
        if not actual_parent or Path(actual_parent).resolve()!=PARENT.resolve():
            raise ValueError('Checkpoint initialization_report does not identify the authorized static Flow parent')
        protocol['actual_training_parent']=str(Path(actual_parent).resolve())
        if saved.get('schema')!='molvid.coordinate_state.checkpoint.v1' or saved['step']!=settings['updates'] or saved['step']!=512:
            raise ValueError('Expected the completed matched 512-update checkpoint')
        if settings['scheme']!='flow' or settings.get('query_qk_mode','atom')!=args.arm or settings['sampler_intervals']!=18:
            raise ValueError('Checkpoint does not match requested Flow Q/K mode and 35NFE sampler')
        model = CoordinateStateDenoiser(settings).to(device)
        model.load_state_dict(saved['model'],strict=True)
        model.eval().requires_grad_(False)
        protocol.update(training_checkpoint_step=saved['step'],sampler_intervals=18,
            planned_vector_field_evaluations_per_draw=35,source_rho_A=model.rho_A,
            model_source=str(Path(sys.modules[CoordinateStateDenoiser.__module__].__file__).resolve()))
        del saved
        hooks = QueryResidualSums(model)
        for seed in (17,29):
            hooks.reset(); before=model.forward_calls
            xyz = forecast(model,observed,query,device,seed)
            torch.cuda.synchronize()
            actual_nfe = model.forward_calls-before
            # Preserve the raw draw before shape/finiteness checks, including failures.
            atomic_torch_save(args.output/f'coordinates_seed{seed}.pt',dict(record=record,
                sample_id=record['sample_id'],atom_identity=record['atom_identity'],
                coordinate_unit='angstrom',history_frames=4,query_time_ps=relative,
                predictions=[dict(seed=seed,coordinates=xyz.cpu())]))
            residual = hooks.summary()
            residual['actual_vector_field_evaluations']=actual_nfe
            protocol['residual_summary'][str(seed)]=residual
            status(last_saved_seed=seed,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            if tuple(xyz.shape)!=(50,batch.x.shape[1],3) or not torch.isfinite(xyz).all() or actual_nfe!=35:
                raise ValueError('Raw draw retained, but shape/finiteness/actual35NFE validation failed')
        status(status='completed')
    except Exception as error:
        if hooks is not None:
            protocol['partial_residual_summary']=hooks.summary()
        status(status='failed',error=repr(error))
        raise
    finally:
        if hooks is not None: hooks.close()


if __name__=='__main__':
    main()
