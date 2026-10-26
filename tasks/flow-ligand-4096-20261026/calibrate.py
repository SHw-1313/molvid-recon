"""Four fixed restored TRAIN windows and one full-complex native-tail profile."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from checks import PARENT,TASK,RUN,equal,fixed_record,difference
from molvid.codec.frame import slice_clip_frames
from molvid.data.bridge_mixture import BridgeMixture
from molvid.flow.coordinate_state import CoordinateStateDenoiser,coordinate_state_objective
from molvid.flow.local_geometry_loss import ligand_geometry_band_loss,_topology
from molvid.training.coordinate_state import LigandSourceGeometry,_global_rng_state,validate_continuation


def selected_windows(saved,settings):
    mixture=BridgeMixture(settings['data'],seed=71)
    mixture.load_state_dict(deepcopy(saved['mixture']))
    counts={4:0,8:0};items=[];skipped=[]
    try:
        for step in range(512,640):
            host,task,_,_,identity=mixture.next()
            assert task=='dynamic' and host.frames==16
            history=(4,8)[step%2];queries=16-history
            item=dict(global_step=step+1,sample_id=host.sample_id[0],identity=identity,
                history=history,queries=queries,atoms=host.atom_count,time_ps=host.time_ps[0].tolist())
            eligible=identity['domain']=='complex' and counts[history]<2 and bool(
                (host.loss_mask & (host.component_id>0)).any())
            if eligible:
                counts[history]+=1
                items.append(dict(batch=host,metadata=item,skipped_draw_shapes=skipped));skipped=[]
            else:
                skipped.append((host.batch_size,queries,host.atom_count))
            if counts=={4:2,8:2}: break
        if counts!={4:2,8:2}: raise RuntimeError(f'Insufficient fixed TRAIN windows: {counts}')
        return items
    finally: mixture.close()


def grad_norm(model):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise ValueError('Missing/nonfinite model gradient')
    return float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())


def coverage(batch):
    valid=batch.loss_mask & (batch.component_id>0)
    internal=batch.component_id[batch.bond_index[0]]==batch.component_id[batch.bond_index[1]]
    selected=replace(batch,loss_mask=valid,bond_index=batch.bond_index[:,internal])
    bonds,_,_,angles=_topology(selected,batch.atom_count)
    assert bool((batch.component_id[bonds]>0).all()) and bool((batch.component_id[angles]>0).all())
    assert bool((batch.component_id[bonds]==batch.component_id[bonds[:1]]).all())
    assert bool((batch.component_id[angles]==batch.component_id[angles[:1]]).all())
    return dict(valid_ligand_atoms=int(valid.sum()),bonds=bonds.shape[1],angles=angles.shape[1],
        positive_components=sorted(set(batch.component_id[valid].tolist())),
        atom_source_index=batch.atom_source_index[valid].tolist(),atomic_numbers=(batch.atype[valid]+1).tolist(),
        same_component_internal=True,no_residue_or_block_filter=True)


def native_reference(model,observed,times,generator_state):
    generator=torch.Generator(device='cuda');generator.set_state(generator_state)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=True,cache_enabled=False):
        condition=model.prepare_condition(observed);before=model.forward_calls
        result=model.sample(condition,times,generator=generator)
    assert model.forward_calls-before==35 and bool(torch.isfinite(result).all())
    return result,generator.get_state()


def auxiliary(model,helper,observed,target,times,*,head_terms=False,native_repeat=False):
    result={};global_before=_global_rng_state();before_calls=model.forward_calls
    with helper.isolated_rng():
        before_generator=helper.generator.get_state().clone();native=native2=None
        if native_repeat:
            native,native_rng=native_reference(model,observed,times,before_generator)
            native2,native2_rng=native_reference(model,observed,times,before_generator)
            assert torch.equal(native_rng,native2_rng)
            result['native_repeat']=dict(old_old=difference(native2,native),nfe_each=35)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
            calls=model.forward_calls
            endpoint=model.sample_truncated(observed,times,tail_intervals=4,generator=helper.generator)
            assert model.forward_calls-calls==35
            loss,metrics=ligand_geometry_band_loss(endpoint,target,observed,helper.config['band'])
        result['metrics']=metrics
        assert metrics['local_geometry_bonds']>0 and metrics['local_geometry_angles']>0
        assert torch.isfinite(loss) and float(loss.detach())>0
        if native_repeat:
            assert torch.equal(helper.generator.get_state(),native_rng)
            result['native_repeat'].update(native_tail=difference(endpoint.detach(),native),
                native_tail_vs_repeat=difference(endpoint.detach(),native2),
                source_generator_exact=True,tail_intervals=4,
                endpoint_comparison='Diagnostic against same-model numerical baseline; no bit-exact claim')
        coordinate_gradient=torch.autograd.grad(loss,endpoint,retain_graph=True)[0]
        ligand=observed.loss_mask & (observed.component_id>0);protein=observed.component_id==0
        assert bool(torch.isfinite(coordinate_gradient).all()) and float(coordinate_gradient[:,ligand].norm())>0
        assert not bool(coordinate_gradient[:,protein].any())
        result['endpoint_gradient']=dict(ligand_norm=float(coordinate_gradient[:,ligand].norm()),protein_norm=0.)
        if head_terms:
            result['isolated_head_gradients']={};parameters=list(model.head.parameters())
            for term in ('bond','angle'):
                bands=dict(helper.config['band'])
                # Diagnostic isolation on the same generated graph; production
                # bands and the unweighted calibration loss remain unchanged.
                if term=='bond': bands['cos_floor']=10.
                else: bands['bond_floor_A']=1e12
                term_loss,_=ligand_geometry_band_loss(endpoint,target,observed,bands)
                gradients=torch.autograd.grad(term_loss,parameters,retain_graph=True,allow_unused=True)
                values=[g for g in gradients if g is not None]
                norm=float(torch.stack([g.float().square().sum() for g in values]).sum().sqrt())
                assert all(bool(torch.isfinite(g).all()) for g in values) and norm>0
                result['isolated_head_gradients'][term]=dict(norm=norm,loss=float(term_loss.detach()),
                    diagnostic_only=True,other_term_silenced=True,new_source_forward=False)
        loss.backward();result['unweighted_aux_gradient_norm']=grad_norm(model)
        result['unweighted_aux_loss']=float(loss.detach());assert result['unweighted_aux_gradient_norm']>0
        del endpoint,loss,coordinate_gradient,native,native2
    assert equal(global_before,_global_rng_state())
    result['ordinary_rng_preserved']=True
    result['forward_calls_including_recomputation']=model.forward_calls-before_calls
    return result


def run(args,result):
    settings=json.loads(args.config.read_text())
    saved=torch.load(args.parent,map_location='cpu',weights_only=False)
    validate_continuation(saved,settings,71)
    windows=[] if args.max_only else selected_windows(saved,settings)
    result['selected_train_windows']=[x['metadata'] for x in windows]
    args.output.mkdir(parents=True,exist_ok=True)
    if windows:
        (args.output/'selected_train_windows.json').write_text(json.dumps(dict(parent=str(args.parent),
            selection='first two eligible complex H4 and first two H8 after saved step512 cursor',
            windows=result['selected_train_windows']),indent=2)+'\n')
    torch.manual_seed(71)
    model=CoordinateStateDenoiser(settings).cuda().train();model.load_state_dict(saved['model'],strict=True)
    helper=LigandSourceGeometry(settings['ligand_geometry'],71,'cuda')
    generator=torch.Generator(device='cuda');generator.set_state(saved['generator_rng'])
    result['windows']=[];head_seen=set()
    for number,item in enumerate(windows):
        began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for systems,queries,atoms in item['skipped_draw_shapes']:
            torch.rand(systems,device='cuda',generator=generator)
            torch.randn((queries,atoms,3),device='cuda',generator=generator)
        batch=item['batch'].to('cuda');history=item['metadata']['history']
        observed=slice_clip_frames(batch,0,history);target=batch.x[history:];times=batch.time_ps[:,history:]
        row=dict(item['metadata'],coverage=coverage(batch));result['windows'].append(row)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
            condition=model.prepare_condition(observed)
            rf,metrics=coordinate_state_objective(model,condition,target,times,generator=generator)
        rf.backward();row['rf_gradient_norm']=grad_norm(model);row['rf_loss']=float(rf.detach());row['rf_metrics']=metrics
        assert row['rf_gradient_norm']>0
        del condition,rf
        row['source_aux']=auxiliary(model,helper,observed,target,times,
            head_terms=history not in head_seen,native_repeat=number==0)
        head_seen.add(history)
        row['rf_to_aux_ratio']=row['rf_gradient_norm']/row['source_aux']['unweighted_aux_gradient_norm']
        torch.cuda.synchronize();row['wall_seconds']=time.monotonic()-began
        row['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        (args.output/'calibration_progress.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        model.zero_grad(set_to_none=True);del batch,observed,target,times
        gc.collect();torch.cuda.empty_cache()
    if windows:
        ratios=[row['rf_to_aux_ratio'] for row in result['windows']]
        weight=min(.5*statistics.median(ratios),min(ratios))
        assert math.isfinite(weight) and weight>0
        result.update(weight=weight,rule='min(0.5*median(RF_norm/aux_norm),min(RF_norm/aux_norm))',
            weighted_aux_to_rf=[weight/r for r in ratios],optimizer_steps=0)
    if not args.skip_max:
        began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        host,metadata=fixed_record(settings,'6D6U')
        assert host.atom_count==16999,'Fixed maximum complex changed; do not substitute'
        batch=host.to('cuda');observed=slice_clip_frames(batch,0,4)
        result['max_complex']=dict(metadata,history=4,queries=12,coverage=coverage(batch))
        result['max_complex']['source_aux']=auxiliary(model,helper,observed,batch.x[4:],batch.time_ps[:,4:])
        torch.cuda.synchronize();result['max_complex']['wall_seconds']=time.monotonic()-began
        result['max_complex']['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    if windows:
        config=deepcopy(settings);config['ligand_geometry'].update(weight=result['weight'],weight_status='calibrated')
        if args.config_output.exists(): raise FileExistsError(args.config_output)
        args.config_output.write_text(json.dumps(config,indent=2)+'\n');result['frozen_config']=str(args.config_output)
    result['passed']=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=TASK/'configs/ligand_uncalibrated.json')
    parser.add_argument('--config-output',type=Path,default=TASK/'configs/ligand.json')
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--output',type=Path,default=RUN/'calibration')
    parser.add_argument('--cap-GiB',type=float,default=64.)
    parser.add_argument('--max-only',action='store_true');parser.add_argument('--skip-max',action='store_true')
    args=parser.parse_args()
    if args.max_only and args.skip_max: raise ValueError('No work selected')
    receipt=args.output/('max_profile.json' if args.max_only else 'calibration.json')
    if receipt.exists(): raise FileExistsError(receipt)
    result=dict(passed=False,parent=str(args.parent),config=str(args.config),cap_GiB=args.cap_GiB);began=time.monotonic()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('Budgeted CUDA required')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(args.cap_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        run(args,result)
    except Exception as error:
        result['error']=repr(error);raise
    finally:
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
        result['wall_seconds']=time.monotonic()-began
        args.output.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(receipt),passed=result['passed'],wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__': main()
