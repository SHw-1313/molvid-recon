"""Finite P1/P2 imports. Common metric definitions remain in the existing MAIN evaluator."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys

STUDY='P1'
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/('runs/flow_static_to_md_20261023' if STUDY=='P1' else 'runs/flow_causal_train_20261023')
MAIN=Path('/workspace/MOLVID-MAIN-20261016');MAIN_RUN=MAIN/'runs/main_integration_20261016'
BASE=Path('/workspace/MOLVID-FLOW-CAUSAL-20261022/runs/flow_causal_20261022/completion_q12')
SAMPLING=BASE
CASES=('atlas_1whz_A','atlas_3vnb_A','atlas_6e5y_A','4J8T','2XSB','5TV1')
MODES=('joint','q12','coarse_fine')
sys.path.insert(0,str(MAIN));sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
import compare_geometry_continuation as shared
from molvid.evaluation.common_panel import load_draws,prepare_case,write_json


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def specifications(run):
    if STUDY=='P1':
        return {stage+'_'+mode:dict(mode=mode,stage=stage,model=stage+'_'+mode,mask=False,
            folder=str(run/'generation'/stage/mode),checkpoint=str(run/stage/'coordinate_final.pt'),new=True)
            for stage in ('old_static','new_static','new_static_replay') for mode in MODES}
    result={}
    for training in ('bi','causal'):
        for mask in (0,1):
            for mode in MODES:
                infer='causal' if mask else 'bi';old=('causal' if mask else 'bidirectional')+'_'+mode
                result[f'train_{training}_infer_{infer}_{mode}']=dict(mode=mode,stage=training,model=old,mask=bool(mask),new=training=='causal',
                    folder=str(run/'generation'/f'mask{mask}'/mode),checkpoint=str(run/'causal/coordinate_final.pt'),source_arm=old)
    return result


SPECS=specifications(RUN);ARMS=tuple(SPECS);NEW_ARMS=tuple(k for k,v in SPECS.items() if v['new']);REUSED={k:v['source_arm'] for k,v in SPECS.items() if not v['new']}
legacy=load_module('artifact_validation_readonly',Path('/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022/prepare_scores.py'))


def cpu_only():
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Existing torch-ito CPU with CUDA hidden required')


def validate(p,spec,seed,prepared,partial):
    if p.get('status')!='completed' and not (partial and str(seed) in p.get('draws',{})):raise ValueError('No completed draw receipt')
    if (p.get('model'),p.get('sampling_mode'),p.get('inference_query_causal'))!=(spec['model'],spec['mode'],spec['mask']):raise ValueError('Training/inference mode mismatch')
    if Path(p['checkpoint']).resolve()!=Path(spec['checkpoint']).resolve() or p.get('checkpoint_step')!=512:raise ValueError('Wrong checkpoint/step')
    if STUDY=='P1' and p.get('arm')!=spec['stage']:raise ValueError('Wrong P1 stage')
    if (p.get('history_frames'),p.get('query_frames'))!=(4,50) or seed not in p.get('seeds',[]):raise ValueError('H4Q50 seed mismatch')
    if p.get('query_time_ps')!=prepared.time_ps.tolist() or p.get('query_dt_ps')!=float(prepared.time_ps[0]):raise ValueError('Native clock mismatch')
    if p.get('coordinate_unit')!='angstrom' or p.get('raw') is not True or p.get('sampler_intervals')!=18:raise ValueError('Raw/native solver mismatch')
    detail=p['draws'][str(seed)];info=detail['sampling_info'];nfe={'joint':35,'q12':175,'coarse_fine':70}[spec['mode']]
    if detail.get('actual_vector_field_evaluations')!=nfe or info.get('total_NFE')!=nfe:raise ValueError('Actual NFE mismatch')
    if spec['mode']=='q12' and (info.get('block_lengths')!=[12]*5 or info.get('generated_frames')!=60 or info.get('scored_frames')!=50):raise ValueError('Q12 raw60/first50 mismatch')
    return detail


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));p.add_argument('--seeds',type=int,nargs='+',choices=[17,29],default=[17,29]);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();cpu_only()
    out=(a.output_root or a.run_root).resolve();dest=out/'common_input';dest.mkdir(parents=True,exist_ok=False)
    specs=specifications(a.run_root.resolve());source=shared.read(MAIN_RUN/'short50_generated_20261017/manifest_ready.json');templates={c['id']:c for c in source['cases']}
    cases=[];missing=[];sampling=[]
    for sid in a.cases:
        case={k:copy.deepcopy(v) for k,v in templates[sid].items() if k!='models'};prepared=prepare_case(case,MAIN_RUN/'short50_generated_20261017');case['models']=[]
        for arm,spec in specs.items():
            if not spec['new']:continue
            folder=Path(spec['folder'])/sid;predictions=[];unavailable=[]
            model=dict(name=arm,history_frames=4,query_time_ps=prepared.time_ps.tolist(),coordinate_unit='angstrom',requested_seeds=a.seeds)
            for seed in a.seeds:
                path=folder/f'coordinates_seed{seed}.pt'
                try:
                    protocol=shared.read(folder/'protocol.json');detail=validate(protocol,spec,seed,prepared,a.allow_partial)
                    d=load_draws(dict(model,cache=str(path),seeds=[seed]),prepared,dest)[0];xyz=torch.as_tensor(d['coordinates'])
                    if not torch.isfinite(xyz).all():raise ValueError('Nonfinite raw retained, no valid score')
                    artifact=legacy.validate_artifact(folder,seed,'cf_clean' if spec['mode']=='coarse_fine' else spec['mode'],xyz,prepared)
                    src=torch.load(folder/f'source_seed{seed}.pt',map_location='cpu',weights_only=False)
                    if src.get('first_source_exact') is not True or list(src['epsilon'].shape)!=[60 if spec['mode']=='q12' else 50,*xyz.shape[1:]]:raise ValueError('Source epsilon evidence mismatch')
                    predictions.append(dict(seed=seed,coordinates=xyz));sampling.append(dict(case=sid,arm=arm,seed=seed,protocol=str(folder/'protocol.json'),source_artifact=str(folder/f'source_seed{seed}.pt'),**detail,**artifact))
                except Exception as exc:
                    failure=dict(case=sid,arm=arm,seed=seed,reason=f'{type(exc).__name__}: {exc}',raw=str(path));missing.append(failure);unavailable.append(failure)
            model.update(seeds=[r['seed'] for r in predictions],missing_draws=unavailable)
            if predictions:
                path=dest/f'{arm}_{sid}.pt';torch.save(dict(sample_id=prepared.record['sample_id'],atom_identity=prepared.topology['atom_identity'],coordinate_unit='angstrom',history_frames=4,query_time_ps=prepared.time_ps.tolist(),predictions=predictions),path);model['cache']=str(path)
            else:model.update(status='NA',reason='No completed valid requested draw')
            case['models'].append(model)
        cases.append(case)
    write_json(dest/'manifest.json',dict(source,cases=cases))
    write_json(dest/'import_details.json',dict(study=STUDY,run_root=str(a.run_root.resolve()),requested_cases=a.cases,requested_seeds=a.seeds,arm_specs=specs,sampling=sampling,missing=missing,
        baseline_panel=str(BASE/'comparison/viewer_panel_index.json'),baseline_comparison=str(BASE/'comparison'),reference_geometry_policy='reuse original MD geometry; only new trajectories evaluated'))
    print(f'{len(sampling)} new valid draws; {len(missing)} missing')
    if missing and not a.allow_partial:raise ValueError('Use explicit --allow-partial for actual valid coverage')


if __name__=='__main__':main()
