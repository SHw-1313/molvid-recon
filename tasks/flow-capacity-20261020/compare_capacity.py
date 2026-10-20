"""Thin seven-arm capacity Q50 import/comparison using existing MAIN helpers.

prepare -> MAIN/tools/evaluate_common_panel.py -> compare --export-viewer.
Only the 84 current-study draws enter scoring; no old baseline is rescored.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path('/workspace/MOLVID-MAIN-20261016')
MAIN_RUN = MAIN/'runs/main_integration_20261016'
RUN = ROOT/'runs/flow_capacity_20261020'
# This CPU entry deliberately uses MAIN evaluation modules. Generation uses the
# separate run_flow_short.py and its own branch's model; no module-path mixing.
sys.path.insert(0,str(MAIN))
sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
import compare_geometry_continuation as shared
from molvid.evaluation.common_panel import load_draws,prepare_case,write_json

ARMS = ('control','observed','query_pre','temporal','collective','query_post','wide')
PAIRS = tuple((arm+'-control',arm,'control') for arm in ARMS if arm!='control')
EXPECTED_DRAWS = 84
CASES,SEEDS = shared.CASES,shared.SEEDS


def validate_protocol(protocol, args, arm, seed, time_ps):
    """Validate the capacity runner's existing protocol; never load a model."""
    if protocol.get('model')!='capacity_'+arm or protocol.get('arm')!=arm:
        raise ValueError('Capacity arm/model label mismatch')
    if protocol.get('status')!='completed' and not args.allow_partial:
        raise ValueError('Generation protocol is not completed')
    for key in ('training_updates','static_updates'):
        if protocol.get(key)!=4096:
            raise ValueError('Expected completed static4096 then MD4096: '+key)
    expected={'checkpoint':args.run_root/arm/'md/coordinate_final.pt',
              'training_parent':args.run_root/arm/'static/coordinate_final.pt'}
    for key,path in expected.items():
        if Path(protocol[key]).resolve()!=path.resolve():
            raise ValueError('Wrong arm-specific '+key)
    if (protocol.get('history_frames'),protocol.get('query_frames'))!=(4,50):
        raise ValueError('Expected native H4/Q50')
    if protocol.get('query_time_ps')!=[float(x) for x in time_ps]:
        raise ValueError('Physical query clock differs from reference')
    if protocol.get('query_dt_ps')!=float(time_ps[0]):
        raise ValueError('Native query interval mismatch')
    if protocol.get('coordinate_unit')!='angstrom' or protocol.get('raw') is not True:
        raise ValueError('Expected unchanged raw Angstrom coordinates')
    if sorted(protocol.get('seeds',[]))!=list(SEEDS):
        raise ValueError('Requested seeds must be 17/29')
    if protocol.get('sampler_intervals')!=18 or protocol.get('planned_vector_field_evaluations_per_draw')!=35:
        raise ValueError('Expected existing 18-interval / 35NFE sampler')
    if protocol.get('draws',{}).get(str(seed),{}).get('actual_vector_field_evaluations')!=35:
        raise ValueError('Missing or invalid actual per-seed 35NFE evidence')
    if not isinstance(protocol.get('capacity'),dict) or protocol.get('parameters',0)<=0:
        raise ValueError('Missing capacity/parameter identity')


def absolute_summaries(rows):
    result=shared.absolute_summaries(rows,ARMS,PAIRS)
    # Carry through existing scalar outputs omitted by the old helper suffix
    # filter. No metric is newly computed and no preferred direction is added.
    keys=('common.acf_contact_count_rmse','common.bb_rmsd0_pred_mean_A',
          'common.bb_rmsd0_ref_mean_A',
          *[f'geometry.{part}.clashes.clashes_per_1000_atoms' for part in ('all','protein','ligand')])
    for row in result['absolute_per_case']:
        selected=[r for r in rows if r['arm']==row['arm'] and r['case']==row['case'] and r['status']=='ok']
        row['metrics'].update({key:shared.mean([r.get(key) for r in selected]) for key in keys})
    for row in result['absolute_per_dataset']:
        selected=[r for r in result['absolute_per_case'] if r['arm']==row['arm'] and r['dataset']==row['dataset']]
        row['metrics'].update({key:shared.mean([r['metrics'][key] for r in selected]) for key in keys})
        row['case_counts'].update({key:sum(shared.finite(r['metrics'][key]) for r in selected) for key in keys})
    return result


def state_equal(a, b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) or isinstance(b,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def training_pairing(run_root):
    """Read final CPU checkpoint metadata only; do not instantiate any model."""
    import torch
    keys=('exposure','mixture','generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng')
    rows,errors,static_exposure=[],[],{}
    for stage in ('static','md'):
        control=None
        for arm in ARMS:
            path=run_root/arm/stage/'coordinate_final.pt'
            row=dict(arm=arm,stage=stage,checkpoint=str(path),status='ok')
            try:
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved.get('schema')!='molvid.coordinate_state.checkpoint.v1':
                    raise ValueError('Unexpected checkpoint schema')
                row.update(step=saved['step'],seed=saved['seed'],exposure=saved['exposure'])
                if saved['step']!=4096 or saved['settings']['updates']!=4096 or saved['seed']!=71:
                    raise ValueError('Expected completed4096 and shared seed71')
                if saved['exposure']['updates']!=4096:
                    raise ValueError('Exposure update count differs from completed checkpoint')
                own,other=('static_examples','dynamic_examples') if stage=='static' else ('dynamic_examples','static_examples')
                if saved['exposure'][own]!=4096 or saved['exposure'][other]!=0:
                    raise ValueError('Stage exposure is not 4096 pure static or pure dynamic examples')
                if stage=='static':
                    static_exposure[arm]=saved['exposure']
                else:
                    parent=saved['initialization_report']
                    row['actual_training_parent']=parent['parent_checkpoint']
                    if Path(parent['parent_checkpoint']).resolve()!=(run_root/arm/'static/coordinate_final.pt').resolve():
                        raise ValueError('MD checkpoint does not start from its own static final')
                    if parent['parent_step']!=4096 or parent['parent_seed']!=71 or not state_equal(parent['parent_exposure'],static_exposure.get(arm)):
                        raise ValueError('MD parent step/seed/exposure inconsistent with static final')
                current={key:saved[key] for key in keys};current['data_settings']=saved['settings']['data']
                del saved
                if arm=='control':control=current
                if control is None:raise ValueError('No valid control checkpoint for pairing')
                equal={key:state_equal(current[key],control[key]) for key in (*keys,'data_settings')}
                # Full-width initialization consumes a different CPU RNG stream;
                # this is recorded, not confused with unpaired training noise.
                required=[key for key in equal if not (arm=='wide' and key=='torch_rng')]
                row.update(equal_to_control=equal,required_equal_fields=required,
                    wide_torch_rng_difference_allowed=(arm=='wide'),model_weights_compared=False)
                failed=[key for key in required if not equal[key]]
                if failed:raise ValueError('Final paired-state mismatch: '+','.join(failed))
            except Exception as exc:
                row.update(status='error',reason=f'{type(exc).__name__}: {exc}')
                errors.append(dict(arm=arm,stage=stage,reason=row['reason']))
            rows.append(row)
    return dict(status='pass' if not errors else 'error',expected_checkpoints=14,checks=rows,errors=errors,
        scope='Final exposure/data-mixture/normal-generator/global-RNG equality within each stage. Different architecture weights are never compared. Final-state equality is not a new per-step replay.',
        shared_initialization='Five location additions use the separately tested forked initialization. This final metadata check does not rerun initialization tests.',
        GPU_seconds=0)


def prepare(args):
    import torch
    source = shared.read(args.reference_manifest)
    if tuple(c['id'] for c in source['cases'])!=CASES:
        raise ValueError('Expected the fixed six-case H4/Q50 manifest')
    # The finite launcher calls this once after generation. Do not begin a
    # full import/scoring pass before all requested artifacts are complete.
    if not args.allow_partial:
        pending=[]
        for case in CASES:
            for arm in ARMS:
                folder=args.run_root/'generation'/arm/case
                try:
                    protocol=shared.read(folder/'protocol.json')
                    if protocol.get('status')!='completed' or not all((folder/f'coordinates_seed{seed}.pt').is_file() for seed in SEEDS):
                        pending.append(arm+'/'+case)
                except (FileNotFoundError,json.JSONDecodeError):pending.append(arm+'/'+case)
        if pending:raise ValueError('All84 required before full import; pending: '+','.join(pending))
    output=args.output or args.run_root/'common_input'
    output.mkdir(parents=True,exist_ok=False)
    pairing=training_pairing(args.run_root)
    write_json(output/'import_details.json',dict(requested_draws=EXPECTED_DRAWS,available_draws=0,
        import_started=False,training_pairing=pairing))
    if pairing['errors']:raise ValueError('Final checkpoint pairing failed; details retained before trajectory import')
    cases,missing=[],[]
    for original in source['cases']:
        case={k:copy.deepcopy(v) for k,v in original.items() if k!='models'}
        prepared=prepare_case(case,args.reference_manifest.parent)
        if prepared.history_frames!=4 or len(prepared.reference)!=50:
            raise ValueError('Reference is not the fixed H4/Q50')
        case['models']=[]
        for arm in ARMS:
            folder=args.run_root/'generation'/arm/case['id']
            protocol_path=folder/'protocol.json'
            model=dict(name='capacity_'+arm,history_frames=4,query_time_ps=prepared.time_ps.tolist(),
                coordinate_unit='angstrom',requested_seeds=list(SEEDS),protocol=str(protocol_path))
            predictions,unavailable,sources=[],[],[]
            for seed in SEEDS:
                path=folder/f'coordinates_seed{seed}.pt'
                try:
                    protocol=shared.read(protocol_path)
                    validate_protocol(protocol,args,arm,seed,prepared.time_ps)
                    draw=load_draws(dict(model,cache=str(path),seeds=[seed]),prepared,output)[0]
                    predictions.append(dict(seed=seed,coordinates=torch.as_tensor(draw['coordinates'])))
                    sources.append(dict(seed=seed,source=str(path),protocol=str(protocol_path),
                        checkpoint=protocol['checkpoint'],training_parent=protocol['training_parent'],
                        capacity=protocol['capacity'],parameters=protocol['parameters'],actual_NFE=35))
                except Exception as exc:
                    row=dict(seed=seed,reason=f'{type(exc).__name__}: {exc}',expected_source=str(path),
                        raw_artifacts=[str(path)] if path.is_file() else [])
                    unavailable.append(row);missing.append(dict(case=case['id'],arm=arm,**row))
            model.update(seeds=[d['seed'] for d in predictions],missing_draws=unavailable,
                completion_status='partial' if unavailable else 'complete')
            if predictions:
                cache=output/f'{arm}_{case["id"]}.pt'
                torch.save(dict(sample_id=prepared.record['sample_id'],atom_identity=prepared.topology['atom_identity'],
                    coordinate_unit='angstrom',history_frames=4,query_time_ps=prepared.time_ps.tolist(),
                    predictions=predictions,sources=sources),cache)
                model['cache']=str(cache)
            else:
                model.update(status='NA',reason='No valid draw; per-seed reasons and raw paths retained')
            case['models'].append(model)
        cases.append(case)
    write_json(output/'import_details.json',dict(requested_draws=EXPECTED_DRAWS,available_draws=EXPECTED_DRAWS-len(missing),missing=missing,
        generated_draws=0,external_baselines_rescored=False,training_pairing=pairing,import_started=True,
        protocol_required_fields=['arm','capacity','parameters','training_updates','static_updates','draws[seed].actual_vector_field_evaluations']))
    if missing and not args.allow_partial:
        raise ValueError('Missing/invalid draws retained; explicit --allow-partial required')
    write_json(output/'manifest.json',dict(cases=cases,analysis=source['analysis'],reference_manifest=str(args.reference_manifest)))
    print(output/'manifest.json')


def compare(args):
    panel=shared.read(args.panel)
    if tuple(c['id'] for c in panel['cases'])!=CASES:
        raise ValueError('Expected fixed six-case panel')
    output=args.output or args.run_root/'comparison'
    output.mkdir(parents=True,exist_ok=False)
    rows,residues,frames,regions,errors=[],[],[],[],list(panel.get('errors',[]))
    visible=dict(panel,cases=[],errors=[])
    for case in panel['cases']:
        sid=case['id']; reference=shared.load_npz(case['reference_npz'])
        context=shared.reference_context(reference)
        ref_geometry=shared.read(case['reference_local_geometry_json'])
        base=dict(case=sid,dataset='ATLAS' if sid.startswith('atlas_') else 'MISATO')
        regions.append(dict(case=sid,excluded_residues=context['excluded'],residues=[
            dict(index=i,identity=list(key),MD_centroid_rmsf_A=float(context['flexibility'][i]),
                flexible_md_q75=bool(context['regions']['flexible_md_q75'][i])) for i,key in enumerate(context['keys'])]))
        identity=dict(base,arm='MD',seed=None)
        motion,rr,ff=shared.supplemental(reference,ref_geometry,context,identity)
        rows.append(dict(identity,status='ok',**shared.existing_metrics({'raw':{}},ref_geometry,reference['time_ps']),**motion))
        residues.extend(rr);frames.extend(ff)
        visible_case=dict(case,models=[]);visible['cases'].append(visible_case)
        for arm in ARMS:
            model=next((m for m in case['models'] if m['name']=='capacity_'+arm),None)
            visible_case['models'].append(copy.deepcopy(model) if model else dict(name='capacity_'+arm,status='NA',reason='Arm absent',draws=[]))
            draws={int(d['seed']):d for d in (model or {}).get('draws',[])}
            for seed in SEEDS:
                identity=dict(base,arm=arm,seed=seed);draw=draws.get(seed)
                try:
                    if draw is None: raise ValueError('Requested seed absent: '+str((model or {}).get('missing_draws','not present')))
                    trajectory=shared.load_npz(draw['npz']);shared.matching(reference,trajectory)
                    metrics,geometry=shared.read(draw['metrics_json']),shared.read(draw['local_geometry_json'])
                    if int(trajectory['seed'])!=seed or int(trajectory['history_frames'])!=4 or metrics['query_frames']!=50 or metrics['history_frames']!=4:
                        raise ValueError('Seed/history/query mismatch')
                    if geometry['thresholds']!=ref_geometry['thresholds'] or geometry['reference_calibration']!=ref_geometry['reference_calibration']:
                        raise ValueError('Geometry reference/tolerance changed across arms')
                    if not all(part in geometry['metrics'][kind] for part in shared.PARTITIONS for kind in ('bonds','angles')):
                        raise ValueError('Missing backbone/sidechain bond-angle partitions')
                    motion,rr,ff=shared.supplemental(trajectory,geometry,context,identity)
                    rows.append(dict(identity,status='ok',npz=draw['npz'],metrics_json=draw['metrics_json'],
                        local_geometry_json=draw['local_geometry_json'],**shared.existing_metrics(metrics,geometry,reference['time_ps']),**motion))
                    residues.extend(rr);frames.extend(ff)
                except Exception as exc:
                    reason=f'{type(exc).__name__}: {exc}'
                    rows.append(dict(identity,status='NA',reason=reason,npz=(draw or {}).get('npz')))
                    errors.append(dict(identity,reason=reason,raw_artifact=(draw or {}).get('npz')))
    deltas,case_means,dataset_means,flags=shared.contrast_metrics(rows,PAIRS)
    # Reuse the original residue contrast helper with its control anchor alias.
    aliases={arm:('B' if arm=='control' else arm) for arm in (*ARMS,'MD')}
    residue_pairs=tuple((label,aliases[after],aliases[before]) for label,after,before in PAIRS)
    res_delta=shared.residue_contrasts([dict(r,arm=aliases[r['arm']]) for r in residues],residue_pairs)
    suppression=[]
    for case in CASES:
        for seed in SEEDS:
            for label,_,_ in PAIRS:
                selected=[r for r in res_delta if r['case']==case and r['seed']==seed
                          and r['contrast']==label and r['flexible_md_q75']]
                suppression.append(dict(case=case,seed=seed,contrast=label,flexible_residues=len(selected),
                    fraction_already_under_MD_centroid_suppressed=shared.mean([
                        int(r['already_under_MD_centroid_suppressed']) for r in selected])))
    for filename,values in [('draw_metrics.csv',rows),('paired_deltas.csv',deltas),('residue_motion.csv',residues),
                            ('residue_deltas.csv',res_delta),('motion_frames.csv',frames)]:
        shared.save_csv(output/filename,values)
    write_json(output/'reference_regions.json',regions)
    write_json(output/'comparison_summary.json',dict(method=shared.METHOD,panel=str(args.panel),
        helper_source=str(Path(shared.__file__).resolve()),arm_definitions={a:'capacity_'+a for a in ARMS},
        expected_draws=EXPECTED_DRAWS,coverage=[{k:r.get(k) for k in ('case','arm','seed','status','reason','npz')} for r in rows if r['arm']!='MD'],
        errors=errors,per_case=case_means,per_dataset=dataset_means,protection_flags=flags,
        protection_interpretation='Matched current-study control; all arms static4096 then MD4096; original Flow objective. Wide has more parameters and is not an equal-parameter location comparison.',
        prior_baselines_rescored=False, aggregation='Mean across available seeds per case, then equal-case mean within each dataset; actual counts and NA retained',
        fixed_flexible_region_suppression=suppression,raw_trajectories_modified=False,
        **absolute_summaries(rows)))
    visible['errors']=errors;write_json(output/'viewer_panel_index.json',visible)
    if args.export_viewer:
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        viewer=args.run_root/'comparison_viewer';assets=MAIN_RUN/'viewer_assets'
        export_panel_viewer(output/'viewer_panel_index.json',viewer,viewer_js=assets/'3Dmol-official.js')
        for name in ('LICENSE','3Dmol-min.js.LICENSE.txt','source.json'):
            shutil.copyfile(assets/name,viewer/name)
    print(json.dumps(dict(output=str(output),expected_draws=EXPECTED_DRAWS,available_draws=sum(r['status']=='ok' and r['arm']!='MD' for r in rows),errors=len(errors))))
    if errors and not args.allow_partial:
        raise ValueError('Partial results retained; explicit --allow-partial required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','compare'))
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--reference-manifest',type=Path,default=MAIN_RUN/'short50_generated_20261017/manifest_ready.json')
    parser.add_argument('--panel',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--export-viewer',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Use enter-container/torch-ito with CUDA_VISIBLE_DEVICES empty')
    args.run_root=args.run_root.resolve();args.reference_manifest=args.reference_manifest.resolve()
    args.panel=(args.panel or args.run_root/'common_scores/panel_index.json').resolve()
    if args.output: args.output=args.output.resolve()
    if args.mode=='compare': shared.load_engine(shared.ENGINE)
    started,cpu=time.monotonic(),time.process_time()
    {'prepare':prepare,'compare':compare}[args.mode](args)
    print(json.dumps(dict(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,GPUh=0)))


if __name__=='__main__':
    main()
