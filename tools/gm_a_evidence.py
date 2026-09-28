"""A-line CUDA evidence: train-only tables, branch smoke, shared calibration.

No model update occurs in tables/calibration. The model's differentiable forward
owns actual-source training samples; the no-grad generation API is not used.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.cli.train_frame_joint import _open_data, _sampler
from molvid.config import load_config
from molvid.data.batch import collate_clip_records
from molvid.data.chemistry import ATOM_TYPE
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.flow.source import sample_frame_source
from molvid.losses.distribution import _sample_metadata, _sample_pairs
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig, physical_auxiliary_loss, motion_features
from molvid.runtime import atomic_write_json, canonical_hash, configure_device, sha256_file, seed_all
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.execution import execution_schedule
from molvid.training.joint import FrameJointTrainer, JointLossConfig, SampledAuxiliaryConfig, _gradient_norm


PARENT = Path('/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/formal/GM/frame_joint_step_00021208.pt')
PARENT_SHA = 'aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540'
SEED = 20260928
TERMS = ('sampled_clash', 'sampled_rmsf', 'sampled_displacement', 'sampled_bond_band')


def selected_entries(train, raw, *, per_ordinary=3, per_fixed=2):
    sampler = _sampler(train, raw['training'], seed=SEED, rank=0, world=1)
    entries = execution_schedule(sampler.global_batches, sampler.specs, histories=(4,8), seed=SEED, epoch=0)
    counts = Counter()
    chosen = []
    for entry in entries:
        key = (entry.view, entry.time_bucket_id, entry.history_frames)
        required = per_ordinary if entry.view == 'ordinary' else per_fixed
        if counts[key] < required:
            counts[key] += 1
            chosen.append(entry)
        if len(chosen) == 6 * per_ordinary + 3 * per_fixed:
            break
    systems = {sampler.specs[i].sample_id.rsplit('_R',1)[0] for e in chosen for i in e.indices}
    if len(chosen) != 6 * per_ordinary + 3 * per_fixed or len(systems) < 8:
        raise RuntimeError('calibration coverage is incomplete')
    return chosen, sampler


def prepare(model, train, entry, device):
    cpu = collate_clip_records([train[i] for i in entry.indices])
    return prepare_frame_joint_batch(model.target_teacher, cpu, device=device, normalizer=model,
                                     history_frames=entry.history_frames)


def forward_samples(model, batch, *, seed, steps, amp=True):
    main = torch.Generator(device='cuda').manual_seed(seed)
    aux = torch.Generator(device='cuda').manual_seed(seed+100000)
    flow = FrameRectifiedFlowObjective()
    draw = flow.sample(batch.normalized_target, batch.source_center, generator=main)
    sources = tuple(sample_frame_source(batch.source_center, generator=aux)[0] for _ in range(2))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        output = model(draw.interpolated, context=batch.observed_context, query=batch.query,
                       flow_time=draw.flow_time, sampled_sources=sources, sampled_steps=steps,
                       checkpoint_sampled_steps=True)
        objective = flow.loss(output.velocity, draw.target_velocity).total
    return output, objective


def fit_tables(model, train, raw, data_hash, device, output):
    from rdkit import Chem, __version__ as rdkit_version
    table = Chem.GetPeriodicTable()
    radii = {str(i):float(table.GetRvdw(symbol)) for i,symbol in enumerate(ATOM_TYPE)}
    entries, sampler = selected_entries(train, raw)
    source = {'data_hash':data_hash, 'entries':[e.__dict__ for e in entries], 'seed':SEED}
    provisional = PhysicalAuxiliaryConfig.resolve({
        'clash_weight':1.0, 'vdw_radii_A':radii, 'allowed_overlap_A':0.8,
        'one_four_overlap_A':1.2, 'table_provenance':{'split':'train','source_hash':canonical_hash(source)},
    })
    bands = defaultdict(list)
    feature_moments = defaultdict(list)
    rows = []
    for index, entry in enumerate(entries):
        batch = prepare(model, train, entry, device)
        topology = batch.observed_context.topology
        with torch.no_grad():
            losses, counts, diag = physical_auxiliary_loss((batch.target_coordinates,)*2, batch, provisional)
            for sample in range(topology.batch_size):
                nodes, _, reference, frames, _ = _sample_metadata(batch, sample)
                ordinary = entry.view == 'ordinary' and len(frames) >= 8
                feats = motion_features(batch.target_coordinates, batch, sample, rmsf=ordinary, displacement=True)
                for name, feat in feats.items():
                    feature_moments[name].append(float(feat.square().mean()))
                pairs = _sample_pairs(batch, sample, nodes)
                a,b = pairs
                lengths = torch.linalg.vector_norm(batch.target_coordinates[frames][:,a].float()-batch.target_coordinates[frames][:,b].float(),dim=-1)
                reference_lengths = torch.linalg.vector_norm(reference[a].float()-reference[b].float(),dim=-1)
                differences = (lengths-reference_lengths).abs().cpu().numpy()
                kinds = {tuple(sorted(p)):k for p,k in zip(topology.covalent_bond_index.cpu().T.tolist(),topology.covalent_bond_type.cpu().tolist())}
                elements = topology.atom_type.cpu().tolist()
                for col,(aa,bb) in enumerate(pairs.cpu().T.tolist()):
                    lo,hi = sorted((elements[aa],elements[bb]))
                    pair = f'{lo}:{hi}'
                    key = f'{pair}:{kinds[tuple(sorted((aa,bb)))]}'
                    for group in (key,pair,'*'):
                        bands[group].extend(differences[:,col].tolist())
        row = {'entry':entry.__dict__, 'MD_clash_severity_A2_per_atom':float(losses['sampled_clash']),
               'condition_count':int(counts['sampled_clash']), **diag}
        rows.append(row)
        print(json.dumps({'tables_batch':index+1,'MD_severity':row['MD_clash_severity_A2_per_atom']}),flush=True)
        del batch
    widths = {key:float(np.quantile(values,0.99)) for key,values in bands.items() if len(values)>=100 or key=='*'}
    stats = {key:{'count':len(values),'q99_A':float(np.quantile(values,0.99)),
                  'retained':key in widths} for key,values in bands.items()}
    mean_md = float(np.mean([r['MD_clash_severity_A2_per_atom'] for r in rows]))
    # Predeclared before any generation: substantial native MD penalty means
    # the physical threshold definition must be revisited, not force-fit lambda.
    if mean_md > 0.001 or any(r['exact_overlap_pair_frames'] for r in rows):
        atomic_write_json(output/'MD_threshold_failure.json', {'rows':rows,'mean_severity':mean_md})
        raise RuntimeError('native MD threshold sanity failed; inspect topology/type rules')
    config = {**provisional.contract(), 'rmsf_weight':1.0,'displacement_weight':1.0,'bond_band_weight':1.0,
              'rmsf_scale_A':math.sqrt(np.mean(feature_moments['sampled_rmsf'])),
              'displacement_scale_A2':math.sqrt(np.mean(feature_moments['sampled_displacement'])),
              'bond_band_widths_A':widths}
    result = {'schema':'molvid.A.train_physical_tables.v1','physical':config,'source':source,
              'MD_checks':rows,'band_statistics':stats,'band_quantile':0.99,'minimum_type_count':100,
              'band_fallback_order':['element_pair+actual_bond_type','element_pair','global'],
              'radii_source':'https://rdkit.org/docs/source/rdkit.Chem.PeriodicTable.html',
              'rdkit_version':rdkit_version,'element_id_mapping':'ATOM_TYPE[id]; atomic_number=id+1',
              'overlap_allowances_A':{'general':0.8,'graph_distance_3':1.2},
              'MD_severity_guardrail_A2_per_atom':0.001,'test_opened':False}
    atomic_write_json(output/'physical_tables.json',result)
    return result


def calibrate(model, train, raw, tables, device, output, steps):
    entries,_ = selected_entries(train, raw)
    config = PhysicalAuxiliaryConfig.resolve(tables['physical'])
    params = [p for p in model.dit.parameters() if p.requires_grad]
    decoder = [p for p in model.decoder.parameters() if p.requires_grad]
    before = module_state_hash(model)
    rows = []
    for index,entry in enumerate(entries):
        batch = prepare(model,train,entry,device)
        result,flow = forward_samples(model,batch,seed=SEED+index,steps=steps)
        losses,counts,diag = physical_auxiliary_loss(tuple(x.coordinates for x in result.sampled),batch,config)
        flow_grad = torch.autograd.grad(flow,params,retain_graph=True,allow_unused=True)
        flow_norm = math.sqrt(sum(float(g.float().square().sum()) for g in flow_grad if g is not None))
        row = {'entry':entry.__dict__,'flow_norm':flow_norm,'near_zero_flow':flow_norm<1e-8,'groups':{}}
        for name in TERMS:
            if int(counts[name]) == 0:
                row['groups'][name] = {'eligible':False}
                continue
            grads = torch.autograd.grad(losses[name],params,retain_graph=True,allow_unused=True)
            norm = math.sqrt(sum(float(g.float().square().sum()) for g in grads if g is not None))
            dot = sum(float((g.float()*f.float()).sum()) for g,f in zip(grads,flow_grad) if g is not None and f is not None)
            decoder_norm = float(_gradient_norm(losses[name],decoder,retain_graph=True))
            row['groups'][name] = {'eligible':True,'raw':float(losses[name].detach()),'dit_norm':norm,
                                    'decoder_norm':decoder_norm,'active':norm>0,'flow_ratio':norm/flow_norm if flow_norm>=1e-8 else None,
                                    'flow_cosine':dot/(norm*flow_norm) if norm>0 and flow_norm>=1e-8 else None}
        rows.append(row)
        atomic_write_json(output/'calibration_progress.json',{'rows':rows})
        print(json.dumps({'calibration_batch':index+1,'groups':row['groups']}),flush=True)
        del result,flow,losses,batch,flow_grad,grads
        if index == 23 and not any(r['groups']['sampled_clash'].get('active') for r in rows):
            expanded,_ = selected_entries(train, raw, per_ordinary=6, per_fixed=4)
            seen = {(e.indices,e.history_frames) for e in entries}
            entries.extend(e for e in expanded if (e.indices,e.history_frames) not in seen)
            if len(entries) > 48:
                raise RuntimeError('calibration expansion exceeded 48 train batches')
    coefficients={}
    distributions={}
    for name,target,cap in zip(TERMS,(0.02,0.0375,0.0375,0.02),(0.20,0.25,0.25,0.20)):
        ratios=[r['groups'][name]['flow_ratio'] for r in rows if r['groups'][name].get('active') and not r['near_zero_flow']]
        if not ratios:
            raise RuntimeError(f'no active gradient calibration for {name}; expand train-only coverage to 48')
        median,p90 = np.quantile(ratios,[0.5,0.9])
        coefficients[name]=float(min(target/median,cap/p90))
        distributions[name]={'active_count':len(ratios),'eligible_count':sum(r['groups'][name].get('eligible',False) for r in rows),
                             'active_ratio_median':float(median),'active_ratio_p90':float(p90),'weight':coefficients[name]}
        overall=[r['groups'][name]['flow_ratio'] for r in rows if r['groups'][name].get('eligible') and not r['near_zero_flow']]
        distributions[name].update({'overall_ratio_median':float(np.median(overall)),
                                   'overall_ratio_p90':float(np.quantile(overall,.9)),
                                   'zero_gradient_count':sum(value==0 for value in overall)})
    combined=[sum(coefficients[n]*r['groups'][n].get('flow_ratio',0) for n in ('sampled_rmsf','sampled_displacement') if r['groups'][n].get('flow_ratio') is not None) for r in rows if not r['near_zero_flow']]
    combined_p90=float(np.quantile(combined,.9))
    motion_scale=min(1.,.5/combined_p90) if combined_p90>0 else 1.
    for name in ('sampled_rmsf','sampled_displacement'):
        coefficients[name]*=motion_scale
    resolved={**tables['physical'],**dict(zip(('clash_weight','rmsf_weight','displacement_weight','bond_band_weight'),(coefficients[n] for n in TERMS)))}
    after=module_state_hash(model)
    if before!=after:
        raise RuntimeError('calibration changed model weights')
    atomic_write_json(output/'resolved_auxiliary.json',{'schema':'molvid.A.resolved_auxiliary.v1','physical':resolved,
        'euler_steps':steps,'draws':2,'cadence':8,'parent_sha256':PARENT_SHA,'model_unchanged':True,
        'physical_tables_sha256':sha256_file(output/'physical_tables.json'),'rows':rows,'distributions':distributions,
        'motion_combined_p90_before_cap':combined_p90,'motion_shared_scale':motion_scale,'split':'train','test_opened':False,
        'cadence_amortization':'multiply active-step ratios by 1/8; actual rounding in execution manifests'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('tables','calibrate'),required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/frame_gm_p2_formal_GM_260923.yaml'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=(4,8),default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.time()
    device=configure_device('cuda',deterministic=True)
    seed_all(SEED)
    raw=load_config(args.config,schema='molvid.frame_joint.train.v1')
    train,valid,data_hash,_,_,close=_open_data(raw['data'])
    try:
        loaded=load_frame_joint_inference(PARENT,expected_sha256=PARENT_SHA,codec_path=raw['codec']['checkpoint'],
                                        codec_sha256=raw['codec']['sha256'],device=device)
        model=loaded['model']
        model.train(); model.target_teacher.eval()
        if args.mode=='tables':
            fit_tables(model,train,raw,data_hash,device,args.output)
        else:
            tables=json.loads((args.output/'physical_tables.json').read_text())
            calibrate(model,train,raw,tables,device,args.output,args.steps)
        atomic_write_json(args.output/f'{args.mode}_cost.json',{'elapsed_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,
            'device':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'test_opened':False})
    finally:
        close()


if __name__=='__main__':
    main()
