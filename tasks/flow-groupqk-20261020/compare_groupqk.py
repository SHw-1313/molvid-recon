"""Thin atom/group Q50 import and comparison using fixed MAIN scoring helpers.

prepare -> MAIN/tools/evaluate_common_panel.py -> compare --export-viewer.
Only the 24 new draws enter scoring; external baselines are never rerun.
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
RUN = ROOT/'runs/flow_groupqk_20261020'
# This CPU entry deliberately uses MAIN evaluation modules. Generation uses the
# separate run_flow_short.py and its own branch's model; no module-path mixing.
sys.path.insert(0,str(MAIN))
sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
import compare_geometry_continuation as shared
from molvid.evaluation.common_panel import load_draws,prepare_case,write_json

ARMS = ('atom','group')
PAIRS = (('group-atom','group','atom'),)
CASES,SEEDS = shared.CASES,shared.SEEDS


def prepare(args):
    import torch
    source = shared.read(args.reference_manifest)
    if tuple(c['id'] for c in source['cases'])!=CASES:
        raise ValueError('Expected the fixed six-case H4/Q50 manifest')
    output=args.output or args.run_root/'common_input'
    output.mkdir(parents=True,exist_ok=False)
    cases,missing=[],[]
    for original in source['cases']:
        case={k:copy.deepcopy(v) for k,v in original.items() if k!='models'}
        prepared=prepare_case(case,args.reference_manifest.parent)
        if prepared.history_frames!=4 or len(prepared.reference)!=50:
            raise ValueError('Reference is not the fixed H4/Q50')
        case['models']=[]
        for arm in ARMS:
            folder=args.run_root/'short50_generated'/arm/case['id']
            protocol_path=folder/'protocol.json'
            model=dict(name='Flow-'+arm,history_frames=4,query_time_ps=prepared.time_ps.tolist(),
                coordinate_unit='angstrom',requested_seeds=list(SEEDS),protocol=str(protocol_path))
            predictions,unavailable,sources=[],[],[]
            for seed in SEEDS:
                path=folder/f'coordinates_seed{seed}.pt'
                try:
                    protocol=shared.read(protocol_path)
                    if protocol['model']!='Flow-'+arm or protocol['query_qk_mode']!=arm:
                        raise ValueError('Flow atom/group protocol label mismatch')
                    if Path(protocol['checkpoint']).resolve()!=(args.run_root/arm/'coordinate_final.pt').resolve():
                        raise ValueError('Wrong arm checkpoint')
                    if protocol['sampler_intervals']!=18 or protocol['residual_summary'][str(seed)]['actual_vector_field_evaluations']!=35:
                        raise ValueError('Expected unchanged actual35NFE sampler')
                    draw=load_draws(dict(model,cache=str(path),seeds=[seed]),prepared,output)[0]
                    predictions.append(dict(seed=seed,coordinates=torch.as_tensor(draw['coordinates'])))
                    sources.append(dict(seed=seed,source=str(path),protocol=str(protocol_path)))
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
    write_json(output/'import_details.json',dict(requested_draws=24,available_draws=24-len(missing),missing=missing,
        generated_draws=0,external_baselines_rescored=False))
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
    rows,residues,frames,regions,errors=[],[],[],[],[]
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
            model=next((m for m in case['models'] if m['name']=='Flow-'+arm),None)
            visible_case['models'].append(copy.deepcopy(model) if model else dict(name='Flow-'+arm,status='NA',reason='Arm absent',draws=[]))
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
    # The existing RF helper anchors residue pairing on B. Alias labels only in
    # its input; output contrasts remain explicitly group-atom, never RF B/G.
    aliases={'atom':'B','group':'G','MD':'MD'}
    res_delta=shared.residue_contrasts([dict(r,arm=aliases[r['arm']]) for r in residues],(('group-atom','G','B'),))
    suppression=[]
    for case in CASES:
        for seed in SEEDS:
            selected=[r for r in res_delta if r['case']==case and r['seed']==seed and r['flexible_md_q75']]
            suppression.append(dict(case=case,seed=seed,contrast='group-atom',flexible_residues=len(selected),
                fraction_already_under_MD_centroid_suppressed=shared.mean([int(r['already_under_MD_centroid_suppressed']) for r in selected])))
    for filename,values in [('draw_metrics.csv',rows),('paired_deltas.csv',deltas),('residue_motion.csv',residues),
                            ('residue_deltas.csv',res_delta),('motion_frames.csv',frames)]:
        shared.save_csv(output/filename,values)
    write_json(output/'reference_regions.json',regions)
    write_json(output/'comparison_summary.json',dict(method=shared.METHOD,panel=str(args.panel),
        helper_source=str(Path(shared.__file__).resolve()),arm_definitions={a:'Flow-'+a for a in ARMS},
        expected_draws=24,coverage=[{k:r.get(k) for k in ('case','arm','seed','status','reason','npz')} for r in rows if r['arm']!='MD'],
        errors=errors,per_case=case_means,per_dataset=dataset_means,protection_flags=flags,
        protection_interpretation='Descriptive matched group-atom flags; same static parent and 512 updates; no new loss',
        fixed_flexible_region_suppression=suppression,raw_trajectories_modified=False,
        **shared.absolute_summaries(rows,ARMS,PAIRS)))
    visible['errors']=errors;write_json(output/'viewer_panel_index.json',visible)
    if args.export_viewer:
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        viewer=args.run_root/'comparison_viewer';assets=MAIN_RUN/'viewer_assets'
        export_panel_viewer(output/'viewer_panel_index.json',viewer,viewer_js=assets/'3Dmol-official.js')
        for name in ('LICENSE','3Dmol-min.js.LICENSE.txt','source.json'):
            shutil.copyfile(assets/name,viewer/name)
    print(json.dumps(dict(output=str(output),expected_draws=24,available_draws=sum(r['status']=='ok' and r['arm']!='MD' for r in rows),errors=len(errors))))
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
