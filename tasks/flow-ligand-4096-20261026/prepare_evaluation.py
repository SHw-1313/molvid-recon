"""Task adapter: score new C/L trajectories and reference preserved same-protocol runs."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flow_ligand_4096_20261026'
MAIN=Path('/workspace/MOLVID-MAIN-20261016')
MAIN_RUN=MAIN/'runs/main_integration_20261016'
ENGINE=Path('/workspace/complexmd_eval_logicmd55_misato89_v0.3.1_20261014/engine')
P1=Path('/workspace/MOLVID-FLOW-STATIC-TO-MD-20261023/runs/flow_static_to_md_20261023')
CAP=Path('/workspace/MOLVID-FLOW-CAPACITY-20261020/runs/flow_capacity_20261020')
CAUSAL=Path('/workspace/MOLVID-FLOW-CAUSAL-20261022')
CASES=('atlas_1whz_A','atlas_3vnb_A','atlas_6e5y_A','4J8T','2XSB','5TV1')
MODES=('joint','q12','q4','coarse_fine')
ARMS=('control','ligand')
ORDER_SOURCES=(P1/'temporal_order_20261023/sidecar',CAP/'temporal_order_20261021/sidecar')
sys.path.insert(0,str(MAIN));sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
import compare_geometry_continuation as shared
from molvid.evaluation.common_panel import load_draws,prepare_case,write_json

def read(path):
    return json.loads(Path(path).read_text())

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    before=list(sys.path)
    try:
        sys.path.insert(0,str(Path(path).parent));spec.loader.exec_module(m)
    finally:sys.path[:]=before
    return m

legacy=load_module('ligand_artifact_reader',Path('/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022/prepare_scores.py'))

def cpu_only():
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Use existing torch-ito inside container, CUDA_VISIBLE_DEVICES empty')

def specifications(run,modes=MODES,arms=ARMS,step=4096):
    result={f'{arm}_{mode}':dict(new=True,stage=arm,mode=mode,model=f'{arm}_{mode}',
        folder=str(run/'generation'/arm/mode if step==4096 else run/'generation_probes'/arm/f'step{step}'/mode),
        checkpoint=str(run/arm/('coordinate_final.pt' if step==4096 else f'checkpoint_step{step}.pt')),checkpoint_step=step)
        for arm in arms for mode in modes}
    for stage in ('old_static','new_static'):
        for mode in modes:
            if mode=='q4':continue  # Not generated for these P1 weights: no substitute checkpoint.
            result[f'P1_{stage}_512_{mode}']=dict(new=False,stage=f'P1_{stage}_512',mode=mode,
                source_arm=f'{stage}_{mode}',source_model=f'{stage}_{mode}',source_root=str(P1),
                checkpoint=str(P1/stage/'coordinate_final.pt'),checkpoint_step=512)
    if 'joint' in modes:
        result['old_query_post_4096_joint']=dict(new=False,stage='old_query_post_4096',mode='joint',
            source_arm='query_post',source_model='capacity_query_post',source_root=str(CAP),
            checkpoint=str(CAP/'query_post/md/coordinate_final.pt'),checkpoint_step=4096)
    return result

def condition(sid,mode):
    dt=100. if sid.startswith('atlas_') else 80.
    return dict(dataset='ATLAS' if sid.startswith('atlas_') else 'MISATO',dt_ps=dt,
        history_frames=4,history_interval_ps=dt,query_frames=50,generation_protocol=mode,
        reference_window='original_short50_after_H4',reference_first_lag_ps=dt,reference_last_lag_ps=50*dt)

def validate(p,spec,seed,prepared,partial):
    if p.get('status')!='completed' and not (partial and str(seed) in p.get('draws',{})):
        raise ValueError('No completed per-draw receipt')
    if (p.get('model'),p.get('arm'),p.get('sampling_mode'))!=(spec['model'],spec['stage'],spec['mode']):
        raise ValueError('Model/arm/sampling protocol mismatch')
    if p.get('inference_query_causal',False) is not False:raise ValueError('Expected original bidirectional inference')
    if Path(p['checkpoint']).resolve()!=Path(spec['checkpoint']).resolve() or p.get('checkpoint_step')!=spec['checkpoint_step']:
        raise ValueError('Wrong continuation checkpoint or total step')
    if (p.get('history_frames'),p.get('query_frames'))!=(4,50) or seed not in p.get('seeds',[]):raise ValueError('H4/Q50/seed mismatch')
    if p.get('query_time_ps')!=prepared.time_ps.tolist() or p.get('query_dt_ps')!=float(prepared.time_ps[0]):raise ValueError('Native physical clock mismatch')
    if p.get('coordinate_unit')!='angstrom' or p.get('raw') is not True or p.get('sampler_intervals')!=18:raise ValueError('Raw/native Heun mismatch')
    detail=p['draws'][str(seed)];info=detail['sampling_info'];nfe={'joint':35,'q12':175,'q4':455,'coarse_fine':70}[spec['mode']]
    if detail.get('actual_vector_field_evaluations')!=nfe or info.get('total_NFE')!=nfe:raise ValueError('Actual NFE mismatch')
    if spec['mode']=='q12' and (info.get('block_lengths')!=[12]*5 or info.get('generated_frames')!=60 or info.get('scored_frames')!=50):raise ValueError('Q12 must preserve raw60, score first50')
    if spec['mode']=='q4' and info.get('block_lengths')!=[4]*12+[2]:raise ValueError('Q4 original 13-block protocol mismatch')
    return detail

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES));p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));p.add_argument('--seeds',type=int,nargs='+',choices=[17,29],default=[17,29])
    p.add_argument('--checkpoint-step',type=int,choices=[1024,2048,4096],default=4096)
    p.add_argument('--reuse-evaluation',type=Path)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args();cpu_only()
    dest=a.output_root.resolve()/'common_input';dest.mkdir(parents=True,exist_ok=False)
    source=read(MAIN_RUN/'short50_generated_20261017/manifest_ready.json');templates={c['id']:c for c in source['cases']}
    if a.checkpoint_step!=4096 and a.modes!=['joint']:raise ValueError('Intermediate probes are joint only')
    specs=specifications(a.run_root.resolve(),a.modes,a.arms,a.checkpoint_step);cases=[];missing=[];sampling=[];cache={};cached=[]
    if a.reuse_evaluation:
        previous=read(a.reuse_evaluation/'common_input/import_details.json')
        if previous['run_root']!=str(a.run_root.resolve()) or previous['checkpoint_step']!=a.checkpoint_step:raise ValueError('Reusable evaluation has different run/checkpoint step')
        for c in read(a.reuse_evaluation/'comparison/viewer_panel_index.json')['cases']:
            for m in c['models']:
                if m['name'] in specs and specs[m['name']]['new']:
                    for d in m['draws']:cache[(c['id'],m['name'],d['seed'])]=d
    for sid in a.cases:
        case={k:copy.deepcopy(v) for k,v in templates[sid].items() if k!='models'};prepared=prepare_case(case,MAIN_RUN/'short50_generated_20261017');case['models']=[]
        if prepared.history_frames!=4 or len(prepared.reference)!=50:raise ValueError('Original H4/Q50 reference required')
        for arm,spec in specs.items():
            if not spec['new']:continue
            folder=Path(spec['folder'])/sid;predictions=[];unavailable=[]
            model=dict(name=arm,history_frames=4,query_time_ps=prepared.time_ps.tolist(),coordinate_unit='angstrom',requested_seeds=a.seeds)
            for seed in a.seeds:
                raw=folder/f'coordinates_seed{seed}.pt'
                try:
                    protocol=read(folder/'protocol.json');detail=validate(protocol,spec,seed,prepared,a.allow_partial)
                    draw=load_draws(dict(model,cache=str(raw),seeds=[seed]),prepared,dest)[0];xyz=torch.as_tensor(draw['coordinates'])
                    if not torch.isfinite(xyz).all():raise ValueError('Nonfinite raw retained, not accepted as valid draw')
                    artifact=legacy.validate_artifact(folder,seed,'cf_clean' if spec['mode']=='coarse_fine' else spec['mode'],xyz,prepared)
                    src=torch.load(folder/f'source_seed{seed}.pt',map_location='cpu',weights_only=False)
                    if src.get('first_source_exact') is not True or list(src['epsilon'].shape)!=[60 if spec['mode']=='q12' else 50,*xyz.shape[1:]]:
                        raise ValueError('Preserved source/standard-normal evidence mismatch')
                    prior=cache.get((sid,arm,seed))
                    if prior:
                        saved=shared.load_npz(prior['npz'])
                        if not torch.equal(torch.as_tensor(saved['positions']),xyz):raise ValueError('Previously scored coordinates differ from preserved raw')
                        for key in ('metrics_json','local_geometry_json'):
                            if not Path(prior[key]).is_file():raise ValueError('Previously scored draw missing '+key)
                        cached.append(dict(case=sid,arm=arm,seed=seed,draw=prior,evaluation=str(a.reuse_evaluation.resolve())))
                    else:predictions.append(dict(seed=seed,coordinates=xyz))
                    sampling.append(dict(case=sid,arm=arm,seed=seed,
                        raw=str(raw),protocol=str(folder/'protocol.json'),source_artifact=str(folder/f'source_seed{seed}.pt'),**detail,**artifact))
                except Exception as exc:
                    item=dict(case=sid,arm=arm,seed=seed,reason=f'{type(exc).__name__}: {exc}',raw=str(raw));missing.append(item);unavailable.append(item)
            model.update(seeds=[d['seed'] for d in predictions],missing_draws=unavailable)
            if predictions:
                path=dest/f'{arm}_{sid}.pt';torch.save(dict(sample_id=prepared.record['sample_id'],atom_identity=prepared.topology['atom_identity'],
                    coordinate_unit='angstrom',history_frames=4,query_time_ps=prepared.time_ps.tolist(),predictions=predictions),path);model['cache']=str(path)
            else:model.update(status='NA',reason='No completed valid requested draw; no parent substitution')
            case['models'].append(model)
        cases.append(case)
    write_json(dest/'manifest.json',dict(source,cases=cases))
    write_json(dest/'import_details.json',dict(study='ligand4096',run_root=str(a.run_root.resolve()),requested_cases=a.cases,
        requested_seeds=a.seeds,requested_modes=a.modes,requested_training_arms=a.arms,checkpoint_step=a.checkpoint_step,arm_specs=specs,sampling=sampling,missing=missing,
        expected_new_draws=len(a.cases)*len(a.seeds)*len(a.modes)*len(a.arms),full_study_new_draws=96,cached_new_draws=cached,
        baseline_capability={'old_query_post_4096':{'available_protocols':['joint'],'others':'NA: not generated for this checkpoint'},
                             'P1_old_static_512':{'available_protocols':['joint','q12','coarse_fine'],'q4':'NA: not generated'},
                             'P1_new_static_512':{'available_protocols':['joint','q12','coarse_fine'],'q4':'NA: not generated','coarse_fine_missing':'5TV1 seed17/29, original timeout; no retry'}},
        reference_geometry_policy='Reuse original MD50 topology and normal bands; training auxiliary thresholds never replace evaluation thresholds'))
    print(json.dumps(dict(new_valid_draws=len(sampling),missing=len(missing),manifest=str(dest/'manifest.json'))))
    if missing and not a.allow_partial:raise SystemExit('Incomplete. Use explicit --allow-partial to score existing valid draws.')

if __name__=='__main__':main()
