"""Train-only real-source handedness gradients; no optimizer or weight changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.evaluation.gradient_balance import gradient_snapshot, gradient_comparison
from molvid.evaluation.stereochemistry import ca_handedness
from molvid.flow.coordinate import CoordinateFlowModel, coordinate_flow_loss
from molvid.latent.types import QuerySpec
from molvid.losses.stereochemistry import observed_ca_reference, handedness_margin_loss
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.coordinate_rollout import checkpointed_coordinate_rollout
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_parent
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint SHA mismatch')
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    require_final_coordinate_checkpoint(saved)
    runtime = Path(__file__).resolve().parents[1]/'molvid'
    for name,digest in saved['contract']['runtime_source_sha256'].items():
        if sha256_file(runtime/name) != digest:
            raise ValueError(f'trained model runtime differs: {name}')
    items = load_ca_cache(args.cache,'train')
    by_system = {}
    for item in sorted(items,key=lambda i:i['record']['sample_id']):
        by_system.setdefault(item['metadata']['system'],item)
    systems = sorted(by_system,key=lambda s:hashlib.sha256(('coordinate-chirality-20261005:'+s).encode()).hexdigest())[:8]
    if len(systems) != 8:
        raise ValueError('eight training proteins required')
    device = configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.manual_seed(20261005)
    parent,_,_ = load_parent(device)
    model = CoordinateFlowModel(parent).to(device).eval(); del parent
    if model.contract() != saved['contract']['model']:
        raise ValueError('trained model contract differs')
    model.load_state_dict(saved['model'],strict=True)
    before = saved['model']; del saved['optimizer'],saved['rng']
    protocol = dict(systems=systems,training_only=True,history_frames=4,query_frames=2,
        flow_times=[.25,.75],actual_source_steps=args.aux_steps,reference_solver_steps=16,
        source_seeds=[4000+i for i in range(8)],precision='FP32',model_mode='eval_autograd_enabled',
        margin=.1,minimum_reference_volume_A3=.2,optimizer_updates=0,test_opened=False,
        centers='protein Cα with all N/C/CB bonds; not ligand stereochemistry',
        checkpoint_sha256=args.checkpoint_sha256,checkpoint_cursor=2048,
        gradient_groups='existing dynamic/static comparison field names: static means chirality auxiliary here',
        full_velocity_field_checkpointing=True,
        purpose='train-only gradient feasibility, not model efficacy or kinetics')
    freeze_run_identity(args.output,inputs=[args.checkpoint,args.cache/'complete.json',args.cache/'protocol.json'],
        purpose=protocol['purpose'],protocol=protocol)
    atomic_write_json(args.output/'protocol.json',protocol)
    rows=[]; started=time.monotonic()
    with ExecutionEvidence(args.output):
        for i,system in enumerate(systems):
            if time.monotonic()-started > 900:
                raise TimeoutError('15-minute bounded GPU probe')
            record=by_system[system]['record']
            cpu=slice_clip_frames(collate_clip_records([record]),0,6)
            observed_cpu=slice_clip_frames(cpu,0,4)
            model.prepare_batch(observed_cpu); observed=observed_cpu.to(device)
            query=QuerySpec(cpu.time_ps[:,4:].to(device),cpu.frame_mask[:,4:].to(device))
            target=cpu.x[4:].to(device)
            reference=observed_ca_reference(observed=observed.x,atom_identity=record['atom_identity'],
                block_id=observed.block_id,abid=observed.abid,frame_mask=observed.frame_mask,
                loss_mask=observed.loss_mask,bond_index=observed.bond_index)
            if not reference.indices.numel():
                raise ValueError('selected training protein has no eligible centers')
            def metrics(x):
                return dict(loss=float(handedness_margin_loss(x,reference,frame_mask=query.frame_mask).detach()),
                    **ca_handedness(x.detach(),observed=observed.x,atom_identity=record['atom_identity'],block_id=observed.block_id))
            model.zero_grad(set_to_none=True)
            condition=model.prepare_condition(observed)
            source=model.source(condition=condition,query=query,
                generator=torch.Generator(device=device).manual_seed(4000+i))
            state=checkpointed_coordinate_rollout(model,source,condition=condition,query=query,steps=args.aux_steps)
            auxiliary=handedness_margin_loss(state,reference,frame_mask=query.frame_mask)
            generated_auxiliary=metrics(state); auxiliary.backward()
            aux_grad=gradient_snapshot(model)
            if any(n.startswith('target_teacher.') for n in aux_grad):
                raise RuntimeError('frozen teacher acquired gradients')
            del state,auxiliary,condition
            with torch.no_grad():
                condition=model.prepare_condition(observed)
                generated16=metrics(model.rollout(source,condition=condition,query=query,steps=16))
            del condition
            for progress in (.25,.75):
                model.zero_grad(set_to_none=True)
                condition=model.prepare_condition(observed)
                noisy=torch.lerp(source,target,progress)
                prediction=model(noisy,condition=condition,query=query,
                    flow_time=target.new_full((1,),progress),checkpoint_blocks=True)
                loss=coordinate_flow_loss(prediction,target-source,batch=observed,query=query)
                endpoint=noisy+(1-progress)*prediction
                endpoint_metrics=metrics(endpoint)
                loss.backward(); dynamic_grad=gradient_snapshot(model)
                comparison=gradient_comparison(dynamic_grad,aux_grad)
                rows.append(dict(system=system,sample_id=record['sample_id'],flow_time=progress,
                    source=metrics(source),interpolation=metrics(noisy),endpoint=endpoint_metrics,
                    generated_auxiliary=generated_auxiliary,generated16=generated16,target=metrics(target),
                    rf_loss_A2=float(loss.detach()),groups=comparison))
                del condition,noisy,prediction,endpoint,loss,dynamic_grad
            del aux_grad
            atomic_write_json(args.output/'rows.json',rows)
            print(json.dumps(dict(completed_systems=i+1,rows=len(rows),seconds=time.monotonic()-started)),flush=True)
        model.zero_grad(set_to_none=True)
        if any(not torch.equal(value.detach().cpu(),before[name]) for name,value in model.state_dict().items()):
            raise RuntimeError('read-only probe altered model state')
        atomic_write_json(args.output/'summary.json',dict(status='completed',systems=8,rows=len(rows),
            optimizer_updates=0,rf_backwards=16,actual_source_auxiliary_backwards=8,
            model_state_bitwise_unchanged=True,seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),test_opened=False,effectiveness_tested=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=Path('/workspace/molvid-ca-detail-20261005/runs/ca_detail_20261005/cache'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--aux-steps',type=int,choices=[4,16],default=4)
    run(p.parse_args())
