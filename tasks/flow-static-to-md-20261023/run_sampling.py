"""One fixed-checkpoint sampling mode/case; retain completed draws on explicit append."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CHECKPOINT = Path('/workspace/MOLVID-FLOW-TIME-20261021/runs/flow_time_modeling_20261021/control/coordinate_final.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('old_static','new_static','new_static_replay'),required=True)
    p.add_argument('--mode', choices=['joint', 'q12', 'coarse_fine'], required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[17,29])
    p.add_argument('--append', action='store_true')
    p.add_argument('--allocator-GiB', type=float, default=32.)
    a = p.parse_args()
    import torch
    import molvid
    from molvid.codec.frame import slice_clip_frames
    from molvid.data.batch import collate_clip_records
    from molvid.flow.coordinate_state import CoordinateStateDenoiser
    from sampling_protocols import sample_with_protocol
    from molvid.training.predictive_data import atomic_torch_save
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or not torch.cuda.is_available():
        raise RuntimeError('Existing torch-ito environment and explicitly authorized CUDA required')
    if Path(molvid.__file__).resolve().parent != ROOT/'molvid':
        raise RuntimeError('Wrong workspace package: '+molvid.__file__)
    if len(set(a.seeds)) != len(a.seeds) or not a.seeds:
        raise ValueError('Unique seeds required')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(a.allocator_GiB*1024**3/total,1.),0)
    torch.set_num_threads(2)
    raw = torch.load(a.record,map_location='cpu',weights_only=False)
    record = raw.get('record',raw)
    if record.get('history_frames') != 4 or record.get('future_frames') != 50 or len(record['x']) != 54:
        raise ValueError('Expected the existing observed H4 + reference Q50 record')
    batch = collate_clip_records([record])
    observed = slice_clip_frames(batch,0,4)
    query = batch.time_ps[:,4:]
    relative = query[0]-batch.time_ps[0,3]
    dt = {'atlas':100.,'misato':80.}[record['source']]
    if not torch.allclose(relative,torch.arange(1,51,dtype=relative.dtype)*dt):
        raise ValueError('Native physical clock mismatch')
    saved = torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg = copy.deepcopy(saved['settings'])
    if saved['step'] != 512 or cfg['scheme'] != 'flow' or cfg.get('temporal_experiment',{}).get('mode','control') != 'control':
        raise ValueError('Expected completed control512 Flow checkpoint, not short-trained or async')
    if cfg['sampler_intervals'] != 18 or cfg.get('query_qk_mode') != 'group':
        raise ValueError('Native sampler or group-QK recipe changed')
    identity = dict(model=a.arm+'_'+a.mode,arm=a.arm,sampling_mode=a.mode,checkpoint=str(a.checkpoint.resolve()),
        checkpoint_step=512,record=str(a.record.resolve()),history_frames=4,query_frames=50,
        query_dt_ps=dt,query_time_ps=relative.tolist(),coordinate_unit='angstrom',raw=True,
        sampler_intervals=18,coarse_stride=5 if a.mode=='coarse_fine' else None)
    path = a.output/'protocol.json'
    if a.output.exists():
        if not a.append or not path.is_file():
            raise FileExistsError('Keep existing outputs; append only explicit unfinished seed coverage')
        protocol = json.loads(path.read_text())
        for k,v in identity.items():
            if protocol.get(k) != v: raise ValueError('Append identity mismatch: '+k)
        if protocol['status'] != 'completed':
            raise ValueError('Do not silently retry a failed/incomplete process')
    else:
        a.output.mkdir(parents=True)
        protocol = dict(identity,status='started',seeds=[],draws={},processes=[],
            scope='Matched512 MD updates; parent quality/static replay contrast; native bidirectional query',
            inference_query_causal=False,
            coarse_fine_caveat='Coarse and fine bidirectional; pure RF path clamp, no future-anchor history context, not two-ended inpainting',
            source_commit=saved.get('source_commit'),inference_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),allocator_limit_GiB=a.allocator_GiB)
    for seed in a.seeds:
        if str(seed) in protocol['draws'] or (a.output/f'coordinates_seed{seed}.pt').exists():
            raise FileExistsError('Do not regenerate or overwrite completed seed '+str(seed))
    started = time.monotonic()
    process = dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seeds=a.seeds,
        status='running',CUDA_device=torch.cuda.get_device_name(0))
    protocol['processes'].append(process)
    def save(status=None, **extra):
        protocol.update(extra)
        if status is not None: protocol['status']=status
        process.update(elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        path.write_text(json.dumps(protocol,indent=2)+'\n')
    save('started')
    try:
        model = CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False)
        loaded = model.load_state_dict(saved['model'],strict=True)
        protocol.update(parameters=model.contract()['parameters'],source_rho_A=model.rho_A,
            checkpoint_load=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys))
        del saved
        # Only the observed batch and physical query labels enter the model.
        with torch.no_grad(): condition=model.prepare_condition(observed.to('cuda:0'))
        for seed in a.seeds:
            begin=time.monotonic();before=model.forward_calls
            generator=torch.Generator(device='cuda:0').manual_seed(seed)
            verification_generator=torch.Generator(device='cuda:0').manual_seed(seed)
            epsilon=torch.randn((60 if a.mode=='q12' else 50,*condition.reference_A.shape),device='cuda:0',generator=verification_generator)
            first_source=[]
            def capture_source(module,args,kwargs):
                if not first_source:first_source.append(args[0].detach().clone())
            hook=model.register_forward_pre_hook(capture_source,with_kwargs=True)
            with torch.no_grad():
                xyz=sample_with_protocol(model,condition,query.to('cuda:0'),protocol=a.mode,
                    intervals=18,generator=generator,stride=5).float()
            torch.cuda.synchronize()
            hook.remove()
            expected=condition.reference_A[None]+model.rho_A*epsilon
            expected_first=expected[4::5] if a.mode=='coarse_fine' else expected[:12] if a.mode=='q12' else expected
            if not torch.equal(first_source[0],expected_first):raise ValueError('Actual source does not match the recorded standard-normal tensor')
            atomic_torch_save(a.output/f'source_seed{seed}.pt',dict(epsilon=epsilon.cpu(),reference_A=condition.reference_A.cpu(),rho_A=model.rho_A,first_source_exact=True,noise_shape=list(epsilon.shape)))
            del first_source,epsilon,expected,expected_first
            info=copy.deepcopy(model.last_sampling_info)
            calls=model.forward_calls-before
            # Save raw output before quality checks; no geometry filtering.
            atomic_torch_save(a.output/f'coordinates_seed{seed}.pt',dict(record=record,
                sample_id=record['sample_id'],atom_identity=record['atom_identity'],coordinate_unit='angstrom',
                history_frames=4,query_time_ps=relative,
                predictions=[dict(seed=seed,coordinates=xyz.cpu())]))
            artifacts=getattr(model,'last_sampling_artifacts',None)
            if artifacts:
                atomic_torch_save(a.output/(f'full60_seed{seed}.pt' if a.mode=='q12' else f'anchors_seed{seed}.pt'),
                    {k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in artifacts.items()})
            if xyz.shape != (50,batch.atom_count,3) or not torch.isfinite(xyz).all():
                raise ValueError('Raw retained: shape/nonfinite failure')
            if info['total_NFE'] != calls:
                raise ValueError('Raw retained: actual field counter mismatch')
            protocol['draws'][str(seed)]=dict(actual_vector_field_evaluations=calls,
                sampling_info=info,elapsed_seconds=time.monotonic()-begin)
            protocol['seeds'].append(seed)
            save(last_saved_seed=seed)
            print(json.dumps(dict(mode=a.mode,case=record['sample_id'],seed=seed,status='completed',
                seconds=time.monotonic()-begin,nfe=calls)),flush=True)
        process['status']='completed';save('completed')
    except Exception as error:
        process['status']='failed';save('failed',error=repr(error));raise


if __name__=='__main__':main()
