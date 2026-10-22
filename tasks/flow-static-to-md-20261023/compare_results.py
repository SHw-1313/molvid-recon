"""Task-only P1/P2 aggregation using the established geometry/motion/seam functions."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from prepare_scores import (ARMS,BASE,CASES,MAIN_RUN,MODES,RUN,STUDY,cpu_only,
    load_module,prepare_case,shared,write_json)
old=load_module('readonly_inpaint_slices',Path('/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022/compare_inpaint.py'))


def merge(out):
    from molvid.evaluation.local_geometry import prepare_local_geometry
    imports=shared.read(out/'common_input/import_details.json');manifest=shared.read(out/'common_input/manifest.json')
    templates={c['id']:c for c in manifest['cases']};incoming=shared.read(out/'common_scores/panel_index.json');fresh={c['id']:c for c in incoming['cases']}
    prior={c['id']:c for c in shared.read(imports['baseline_panel'])['cases']};cases=[];errors=list(incoming['errors'])
    for sid in imports['requested_cases']:
        baseline=prior[sid];current=fresh.get(sid,{})
        if current.get('reference_npz'):shared.matching(shared.load_npz(baseline['reference_npz']),shared.load_npz(current['reference_npz']),coordinates=True)
        prepared=prepare_case(templates[sid],out/'common_input');scorer=prepare_local_geometry(prepared.reference,prepared.topology)
        refgeo=shared.read(baseline['reference_local_geometry_json']);models=[]
        for arm,spec in imports['arm_specs'].items():
            source=next((m for m in (current.get('models',[]) if spec['new'] else baseline['models']) if m['name']==(arm if spec['new'] else spec['source_arm'])),None)
            model=copy.deepcopy(source) if source else dict(name=arm,status='NA',reason='Requested draw missing',draws=[])
            model['name']=arm;model['draws']=[d for d in model['draws'] if d['seed'] in imports['requested_seeds']]
            if not spec['new']:model.update(reused_comparison=imports['baseline_comparison'],source_arm=spec['source_arm'])
            else:
                for draw in model['draws']:
                    try:
                        xyz=shared.load_npz(draw['npz']);report=scorer.evaluate(xyz['positions'])
                        if report['thresholds']!=refgeo['thresholds'] or report['reference_calibration']!=refgeo['reference_calibration']:raise ValueError('Existing MD geometry calibration differs')
                        path=Path(draw['npz']).parent/'local_geometry.json'
                        if path.exists():raise FileExistsError('No duplicate geometry evaluation output')
                        write_json(path,report);draw['local_geometry_json']=str(path)
                    except Exception as exc:errors.append(dict(case=sid,arm=arm,seed=draw['seed'],error=repr(exc)))
            models.append(model)
        cases.append(dict(baseline,models=models))
    dest=out/'comparison_inputs';dest.mkdir(exist_ok=False)
    write_json(dest/'panel_index.json',dict(incoming,cases=cases,errors=errors,manifest=str(out/'common_input/manifest.json'),
        reference_geometry_reused=True,old_generated_geometry_score_calls=0))
    print(json.dumps(dict(new_local_geometry_draws=sum(len(m['draws']) for c in cases for m in c['models'] if imports['arm_specs'][m['name']]['new']),errors=errors)))


def contrasts(specs):
    pairs=[]
    def add(a,b):pairs.append((a+'-vs-'+b,a,b))
    if STUDY=='P1':
        for mode in MODES:
            add('new_static_'+mode,'old_static_'+mode);add('new_static_replay_'+mode,'new_static_'+mode);add('new_static_replay_'+mode,'old_static_'+mode)
        groups=('old_static','new_static','new_static_replay')
    else:
        for mode in MODES:
            for infer in ('bi','causal'):add(f'train_causal_infer_{infer}_{mode}',f'train_bi_infer_{infer}_{mode}')
            for train in ('bi','causal'):add(f'train_{train}_infer_causal_{mode}',f'train_{train}_infer_bi_{mode}')
        groups=tuple(f'train_{t}_infer_{i}' for t in ('bi','causal') for i in ('bi','causal'))
    for group in groups:
        for mode in ('q12','coarse_fine'):add(group+'_'+mode,group+'_joint')
    return tuple(pairs)


def compare(a):
    import torch
    from molvid.evaluation.local_geometry import prepare_local_geometry
    out=a.output_root;imports=shared.read(out/'common_input/import_details.json');specs=imports['arm_specs'];arms=tuple(specs)
    panel=shared.read(out/'comparison_inputs/panel_index.json');cases=imports['requested_cases'];seeds=imports['requested_seeds']
    temporal={(r['case'],r['arm'],r['seed']):r for r in shared.read(out/'temporal_order_20261023/sidecar/draw_metrics.json')}
    tables={k:old.read_table(Path(imports['baseline_comparison'])/f'{k}.csv') for k in ('draw_metrics','residue_motion','motion_frames','seam_frames')}
    templates={c['id']:c for c in shared.read(out/'common_input/manifest.json')['cases']}
    costs={(r['case'],r['arm'],r['seed']):r for r in imports['sampling']}
    rows=[];residues=[];frames=[];seams=[];slices=[];tails=[];errors=list(panel['errors'])
    for case in panel['cases']:
        sid=case['id'];dataset='ATLAS' if sid.startswith('atlas_') else 'MISATO';reference=shared.load_npz(case['reference_npz']);context=None;scorer=None
        refgeo=shared.read(case['reference_local_geometry_json']);work=[('MD',None,None)]
        for model in case['models']:work.extend((model['name'],s,next((d for d in model['draws'] if d['seed']==s),None)) for s in seeds)
        for arm,seed,draw in work:
            identity=dict(case=sid,dataset=dataset,arm=arm,seed=seed)
            try:
                if arm!='MD' and draw is None:raise ValueError('No completed valid requested draw; not replaced by parent')
                xyz=reference if arm=='MD' else shared.load_npz(draw['npz']);shared.matching(reference,xyz)
                geo=refgeo if arm=='MD' else shared.read(draw['local_geometry_json'])
                reused=arm=='MD' or not specs[arm]['new'];source_arm='MD' if arm=='MD' else specs[arm].get('source_arm')
                if reused:
                    source=next(r for r in tables['draw_metrics'] if (r['case'],r['arm'],r['seed'])==(sid,source_arm,seed) and r['status']=='ok');row=dict(source,**identity)
                    copied={k:[dict(r,**identity) for r in tables[k] if (r['case'],r['arm'],r['seed'])==(sid,source_arm,seed)] for k in ('residue_motion','motion_frames','seam_frames')}
                    rr,ff,sf=(copied[k] for k in ('residue_motion','motion_frames','seam_frames'))
                else:
                    metrics=shared.read(draw['metrics_json'])
                    if context is None:context=shared.reference_context(reference)
                    motion,rr,ff=shared.supplemental(xyz,geo,context,identity);extra,sf=old.seam_metrics(xyz,context,identity)
                    row=dict(identity,status='ok',npz=draw['npz'],metrics_json=draw['metrics_json'],local_geometry_json=draw['local_geometry_json'],
                        **shared.existing_metrics(metrics,geo,reference['time_ps']),**motion,**extra)
                order=temporal[(sid,arm,seed)];target=temporal[(sid,'MD',None)]
                if order['status']!='ok':raise ValueError('Temporal sidecar failed')
                row.update({'temporal.'+k:v for k,v in order['metrics'].items()})
                row.update({'temporal.'+k+'_absolute_MD_error':abs(v-target['metrics'][k]) if shared.finite(v) and shared.finite(target['metrics'].get(k)) else None for k,v in order['metrics'].items() if k.endswith('order_gain')})
                cost=costs.get((sid,arm,seed))
                if cost:row.update({'sampling.total_NFE':cost['sampling_info']['total_NFE'],'sampling.atom_frame_evaluations':cost['sampling_info']['atom_frame_evaluations']})
                rows.append(row);residues.extend(rr);frames.extend(ff);seams.extend(sf);slices.extend(old.geometry_slices(geo,identity))
                if cost and cost.get('full60_artifact'):
                    if scorer is None:
                        prepared=prepare_case(templates[sid],out/'common_input');scorer=prepare_local_geometry(prepared.reference,prepared.topology)
                    full=torch.load(cost['full60_artifact'],map_location='cpu',weights_only=False)['full_coordinates_A'].numpy();tail=scorer.evaluate(full[50:]);combined=copy.deepcopy(geo)
                    for kind,parts in combined['metrics'].items():
                        for component,m in parts.items():
                            if m.get('available'):m['bad_atoms_by_frame']+=tail['metrics'][kind][component]['bad_atoms_by_frame']
                    tails.extend(old.geometry_slices(combined,identity,count=60,scope='new_Q12_full60_geometry_only'))
            except Exception as exc:
                error=dict(identity,status='NA',reason=f'{type(exc).__name__}: {exc}');errors.append(error)
                if not any((r['case'],r['arm'],r['seed'])==(sid,arm,seed) for r in rows):rows.append(error)
    lookup={(r['case'],r['arm'],r['seed']):r for r in rows}
    for row in rows:
        if row['status']!='ok':continue
        md=lookup[(row['case'],'MD',None)];spec=specs.get(row['arm']);joint=None
        if spec:
            prefix=row['arm'][:-len(spec['mode'])];joint=lookup.get((row['case'],prefix+'joint',row['seed']))
        for key in [k for k in row if k.startswith('seam.') and not k.endswith(('count','fraction','positions'))]:
            row[key+'_minus_MD_same_positions']=row[key]-md[key] if shared.finite(row[key]) and shared.finite(md.get(key)) else None
            row[key+'_minus_joint_same_positions']=row[key]-joint[key] if joint and shared.finite(row[key]) and shared.finite(joint.get(key)) else None
    shared.CASES=tuple(cases);shared.SEEDS=tuple(seeds);pairs=contrasts(specs)
    delta,per_case,per_dataset,flags=shared.contrast_metrics(rows,pairs);absolute=shared.absolute_summaries(rows,('MD',*arms),pairs)
    extra={k for r in rows for k,v in r.items() if shared.finite(v) and (k.startswith(('seam.','temporal.')) or k in ('common.acf_contact_count_rmse','common.bb_rmsd0_pred_mean_A') or k.endswith('clashes_per_1000_atoms'))}
    for row in absolute['absolute_per_case']:
        selected=[r for r in rows if r['case']==row['case'] and r['arm']==row['arm'] and r['status']=='ok'];row['metrics'].update({k:shared.mean([r.get(k) for r in selected]) for k in extra})
    for row in absolute['absolute_per_dataset']:
        selected=[r for r in absolute['absolute_per_case'] if (r['dataset'],r['arm'])==(row['dataset'],row['arm'])];row['metrics'].update({k:shared.mean([r['metrics'][k] for r in selected]) for k in extra});row['case_counts'].update({k:sum(shared.finite(r['metrics'][k]) for r in selected) for k in extra})
    coverage=[{k:r.get(k) for k in ('case','arm','seed','status','reason','npz')} for r in rows if r['arm']!='MD'];complete=not errors and not imports['missing']
    dest=out/'comparison';dest.mkdir(exist_ok=False)
    for name,items in [('draw_metrics',rows),('residue_motion',residues),('motion_frames',frames),('seam_frames',seams),('paired_deltas',delta)]:shared.save_csv(dest/f'{name}.csv',items)
    summary=dict(study=STUDY,expected_draws=len(cases)*len(seeds)*len(arms),available_draws=sum(r['status']=='ok' for r in coverage),coverage=coverage,errors=errors,
        completion_status='complete' if complete else 'incomplete',full_study_complete=complete and len(cases)==6 and len(seeds)==2,arm_specs=specs,
        per_case=per_case,per_dataset=per_dataset,protection_flags=flags,frame_slice_summary=old.frame_slices(frames),geometry_frame_slices=slices,full60_geometry_frame_slices=tails,
        reference_reuse='Original MD geometry/motion and old generated metrics reused; only new draw geometry/motion plus new Q12 tail10 evaluated.',
        protocol='H4, Q50 main; Q12 five full blocks/raw60; pure-clamp coarse10/fine50 without clean future context; native ATLAS100ps/MISATO80ps.',
        aggregation='Two seeds averaged within case, then cases equally weighted within dataset. Amplitude closeness uses existing absolute log-ratio error; temporal values target matched MD.',
        slice_definition='Same global reference and physical edges on all arms/MD. No block alignment. Geometry slice counts bad atoms/frames; not feature anomaly fractions. Fifth Q12 main block is partial2/12.',**absolute)
    write_json(dest/'comparison_summary.json',summary);write_json(dest/'viewer_panel_index.json',dict(panel,errors=errors))
    if a.export_viewer:
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        assets=MAIN_RUN/'viewer_assets';viewer=out/'comparison_viewer';export_panel_viewer(dest/'viewer_panel_index.json',viewer,viewer_js=assets/'3Dmol-official.js')
        for name in ('LICENSE','3Dmol-min.js.LICENSE.txt','source.json'):shutil.copyfile(assets/name,viewer/name)
    print(json.dumps(dict(available=summary['available_draws'],expected=summary['expected_draws'],complete=complete,errors=errors)))
    if not complete and not a.allow_partial:raise ValueError('Incomplete real coverage retained')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=RUN);p.add_argument('--merge-panel',action='store_true');p.add_argument('--distance',action='store_true');p.add_argument('--allow-partial',action='store_true');p.add_argument('--export-viewer',action='store_true');a=p.parse_args();cpu_only()
    if a.merge_panel:merge(a.output_root)
    elif a.distance:distance(a.output_root)
    else:compare(a)


def distance(out):
    """Only new trajectories enter the original D helper; reuse old numeric D."""
    from types import SimpleNamespace
    h=load_module('readonly_distance',Path('/workspace/MOLVID-FLOW-CAUSAL-20261022/tasks/flow-causal-20261022/history_distance.py'))
    imports=shared.read(out/'common_input/import_details.json');specs=imports['arm_specs'];family='static_to_md' if STUDY=='P1' else 'causal_training'
    side=out/'temporal_order_20261023/sidecar';selected=out/'distance_new_inputs';selected.mkdir(exist_ok=False)
    for name in ('summary.json','references.json'):shutil.copyfile(side/name,selected/name)
    drawrows=shared.read(side/'draw_metrics.json');write_json(selected/'draw_metrics.json',[r for r in drawrows if r['arm'] in specs and specs[r['arm']]['new']])
    h.run(SimpleNamespace(sidecar=selected,family=family,index=None,output=out/'distance_new'))
    calculated=shared.read(out/'distance_new/results.json');trajs=calculated['trajectories'];reused=0
    baseline=shared.read(BASE/'distance/results.json')
    for original in baseline['trajectories']:
        if original['case'] not in imports['requested_cases']:continue
        aliases=[]
        for alias in original['aliases']:
            if alias['arm']=='MD':aliases=[dict(alias,family=family)];break
            if alias['seed'] not in imports['requested_seeds']:continue
            for arm,spec in specs.items():
                if not spec['new'] and spec['source_arm']==alias['arm']:aliases.append(dict(alias,family=family,arm=arm))
        if not aliases:continue
        row=copy.deepcopy(original);row['id']='reused_'+row['id'];row['aliases']=aliases;trajs.append(row);reused+=1
    cases,aggregate=h.aggregate_curves(trajs);dest=out/'distance';dest.mkdir(exist_ok=False)
    summary=dict(calculated['summary'],unique_generated_trajectories=sum(not t['is_reference'] for t in trajs),unique_MD_references=sum(t['is_reference'] for t in trajs),
        aliases=sum(len(t['aliases']) for t in trajs),new_generated_computations=sum(not t['is_reference'] for t in calculated['trajectories'])-sum(not t['is_reference'] for t in trajs if t['id'].startswith('reused_')),
        reused_trajectories=reused,new_MD_computations=0,old_generated_computations=0,
        G_identity_max_error=max((v['G_identity_max_error'] or 0 for t in trajs for v in t['observables'].values()),default=0))
    h.write(dest/'results.json',dict(summary=summary,trajectories=trajs,per_case=cases,aggregate=aggregate));h.write(dest/'summary.json',summary)
    for name,items in [('per_case',cases),('aggregate',[{k:v for k,v in r.items() if k!='case_ids'} for r in aggregate]),('key_lag_summary',[{k:v for k,v in r.items() if k!='case_ids'} for r in aggregate if r['lag_frames'] in h.KEY_LAGS])]:h.csv_rows(dest/(name+'.csv'),items)
    print(json.dumps(summary))


if __name__=='__main__':main()
