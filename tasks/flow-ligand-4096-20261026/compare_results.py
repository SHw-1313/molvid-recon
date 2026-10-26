"""Finite C/L comparison. Existing common/geometry/motion/G/D definitions are reused."""
import argparse
import copy
import math
from pathlib import Path
import shutil
import numpy as np
from prepare_evaluation import (ROOT,RUN,MAIN_RUN,ENGINE,P1,CAP,CAUSAL,CASES,MODES,
    cpu_only,condition,read,write_json,load_module,prepare_case,shared)

old=load_module('ligand_existing_seams',Path('/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022/compare_inpaint.py'))
CONDITION_FIELDS=('dataset','dt_ps','history_frames','history_interval_ps','query_frames','generation_protocol','reference_window','reference_first_lag_ps','reference_last_lag_ps')

def panel(root):
    return read(Path(root)/'comparison/viewer_panel_index.json')

def merge(out):
    from molvid.evaluation.local_geometry import prepare_local_geometry
    imports=read(out/'common_input/import_details.json');templates={c['id']:c for c in read(out/'common_input/manifest.json')['cases']}
    incoming=read(out/'common_scores/panel_index.json');fresh={c['id']:c for c in incoming['cases']}
    roots={str(P1),*[s['source_root'] for s in imports['arm_specs'].values() if not s['new']]}
    prior={r:{c['id']:c for c in panel(r)['cases']} for r in roots};cases=[];errors=list(incoming['errors'])
    cached={(r['case'],r['arm'],r['seed']):r for r in imports.get('cached_new_draws',[])}
    for sid in imports['requested_cases']:
        baseline=prior[str(P1)][sid];current=fresh.get(sid,{})
        reference=shared.load_npz(baseline['reference_npz']);refgeo=read(baseline['reference_local_geometry_json'])
        if current.get('reference_npz'):shared.matching(reference,shared.load_npz(current['reference_npz']),coordinates=True)
        prepared=prepare_case(templates[sid],out/'common_input');scorer=None;models=[]
        for arm,spec in imports['arm_specs'].items():
            src=current if spec['new'] else prior[spec['source_root']][sid]
            if src.get('reference_npz'):shared.matching(reference,shared.load_npz(src['reference_npz']),coordinates=True)
            label=arm if spec['new'] else spec['source_model']
            source=next((m for m in src.get('models',[]) if m['name']==label),None)
            model=copy.deepcopy(source) if source else dict(draws=[],status='NA',reason='Historical protocol absent; no new generation')
            model.update(name=arm,requested_seeds=imports['requested_seeds'],generation_protocol=spec['mode'])
            model['draws']=[d for d in model['draws'] if d['seed'] in imports['requested_seeds']]
            for seed in imports['requested_seeds']:
                if (sid,arm,seed) in cached:
                    model['draws'].append(copy.deepcopy(cached[(sid,arm,seed)]['draw']))
                    model['status']='completed'
            for draw in model['draws']:
                if not spec['new'] or (sid,arm,draw['seed']) in cached:continue
                try:
                    xyz=shared.load_npz(draw['npz']);shared.matching(reference,xyz)
                    path=Path(draw['npz']).parent/'local_geometry.json'
                    if path.exists():report=read(path)
                    else:
                        if scorer is None:scorer=prepare_local_geometry(prepared.reference,prepared.topology,thresholds=refgeo['thresholds'])
                        report=scorer.evaluate(xyz['positions']);write_json(path,report)
                    if report['thresholds']!=refgeo['thresholds'] or report['reference_calibration']!=refgeo['reference_calibration']:
                        raise ValueError('Original MD50 evaluation bands changed')
                    draw['local_geometry_json']=str(path)
                except Exception as exc:errors.append(dict(case=sid,arm=arm,seed=draw['seed'],reason=repr(exc)))
            models.append(model)
        cases.append(dict(baseline,models=models))
    dest=out/'comparison_inputs';dest.mkdir(exist_ok=False)
    write_json(dest/'panel_index.json',dict(incoming,cases=cases,errors=errors,manifest=str(out/'common_input/manifest.json'),reference_geometry_reused=True))

def ligand_motion(xyz,reference,context):
    """Small descriptive decomposition using existing Kabsch/RMSF primitives.

    Internal distances are rigid invariant; self-fit RMSF also removes rotation.
    Fixed pocket = protein heavy atoms within 5 A of any ligand atom in MD frame0.
    The ligand/pocket centers are unweighted heavy-atom centroids, NOT mass COM.
    """
    heavy=~np.isin(np.char.upper(reference['elements'].astype(str)),['H','D'])
    ligand=reference['ligand_mask'].astype(bool)&heavy
    if ligand.sum()<3:return {'ligand_motion.available':False,'ligand_motion.reason':'protein_only or fewer than three ligand heavy atoms'},None
    target=reference['positions'][0];protein=(~reference['ligand_mask'].astype(bool))&heavy
    near=np.linalg.norm(target[protein,None]-target[None,ligand],axis=-1).min(1)<=5.
    pocket=np.flatnonzero(protein)[near]
    aligned=shared.align(xyz['positions'],target,context['mask']);ref=shared.align(reference['positions'],target,context['mask'])
    self_xyz=shared.align(xyz['positions'],target,ligand)[:,ligand]
    self_ref=shared.align(reference['positions'],target,ligand)[:,ligand]
    left,right=np.triu_indices(int(ligand.sum()),1)
    def measures(coords,self_coords):
        atom=coords[:,ligand];dist=np.linalg.norm(atom[:,left]-atom[:,right],axis=-1)
        result=dict(internal_self_aligned_RMSF_mean_A=float(shared.rmsf(self_coords).mean()),
            internal_pair_distance_time_std_mean_A=float(dist.std(0).mean()))
        if len(pocket):
            relative=atom.mean(1)-coords[:,pocket].mean(1)
            observed=reference['observed'][-1] if 'observed' in reference else None
            # The NPZ uses observed_positions; no fallback to generated frame0.
            observed=reference.get('observed_positions',observed)
            if observed is not None and observed.ndim==3:observed=observed[-1]
            if observed is None:raise ValueError('Original observed coordinates required for pocket displacement')
            start=shared.align(observed[None],target,context['mask'])[0]
            initial=start[ligand].mean(0)-start[pocket].mean(0)
            result.update(pocket_relative_centroid_RMSF_A=float(shared.rmsf(relative[:,None]).mean()),
                pocket_relative_displacement_from_last_observed_mean_A=float(np.linalg.norm(relative-initial,axis=-1).mean()),
                pocket_relative_displacement_from_last_observed_last_A=float(np.linalg.norm(relative[-1]-initial)))
        return result
    pred=measures(aligned,self_xyz);md=measures(ref,self_ref)
    values={'ligand_motion.available':True,'ligand_motion.ligand_atoms':int(ligand.sum()),'ligand_motion.fixed_pocket_atoms':len(pocket)}
    for key,value in pred.items():
        values['ligand_motion.'+key]=value;values['ligand_motion.MD_'+key]=md[key]
        values['ligand_motion.'+key+'_amplitude_ratio']=value/md[key] if md[key]>1e-12 else None
    return values,dict(ligand_atom_indices=np.flatnonzero(ligand).tolist(),fixed_pocket_atom_indices=pocket.tolist(),
        definition='MD first-future frame, fixed protein heavy atoms <=5A; unweighted centroids; same global backbone fit, internal self-fit separate',
        interpretation='Descriptive amplitudes, not lower-is-better. Internal deformation may be chemically invalid; read with geometry. Fixed pocket is not recentered per generated neighbor selection.')

def summarize(rows,specs,cases,seeds):
    """Seed mean -> equal case, with the complete condition key retained."""
    numeric=sorted({k for r in rows for k,v in r.items() if (k.startswith(('common.','geometry.','motion.','shape.','local_bad.','temporal.','seam.','sampling.','ligand_motion.'))) and shared.finite(v)})
    grouped={}
    for r in rows:
        key=tuple(r.get(k) for k in ('case','arm',*CONDITION_FIELDS));grouped.setdefault(key,[]).append(r)
    per_case=[]
    for key,draws in grouped.items():
        meta=dict(zip(('case','arm',*CONDITION_FIELDS),key));ok=[r for r in draws if r['status']=='ok']
        per_case.append(dict(meta,draws=len(ok),requested_draws=1 if meta['arm']=='MD' else len(seeds),
            metrics={k:shared.mean([r.get(k) for r in ok]) for k in numeric},
            missing=[dict(seed=r['seed'],reason=r.get('reason')) for r in draws if r['status']!='ok']))
    groups={}
    for r in per_case:groups.setdefault(tuple(r[k] for k in ('arm',*CONDITION_FIELDS)),[]).append(r)
    aggregate=[]
    for key,items in groups.items():
        aggregate.append(dict(zip(('arm',*CONDITION_FIELDS),key),case_ids=[r['case'] for r in items if r['draws']],
            cases=sum(r['draws']>0 for r in items),metrics={k:shared.mean([r['metrics'].get(k) for r in items]) for k in numeric},
            metric_case_counts={k:sum(shared.finite(r['metrics'].get(k)) for r in items) for k in numeric}))
    # Only compare arms within an identical generation protocol and physical clock.
    contrasts=[]
    for mode in {s['mode'] for s in specs.values()}:
        names=[n for n,s in specs.items() if s['mode']==mode]
        if f'ligand_{mode}' in names and f'control_{mode}' in names:contrasts.append((f'ligand_{mode}',f'control_{mode}'))
        for new in (f'control_{mode}',f'ligand_{mode}'):
            if new not in names:continue
            contrasts.extend((new,prior) for prior in names if not specs[prior]['new'])
    draw_lookup={(r['case'],r['arm'],r['seed']):r for r in rows if r['arm']!='MD'};paired=[]
    for after,before in contrasts:
        for sid in cases:
            good=[s for s in seeds if draw_lookup.get((sid,after,s),{}).get('status')=='ok' and draw_lookup.get((sid,before,s),{}).get('status')=='ok']
            av={k:shared.mean([draw_lookup[(sid,after,s)].get(k) for s in good]) for k in numeric}
            bv={k:shared.mean([draw_lookup[(sid,before,s)].get(k) for s in good]) for k in numeric}
            paired.append(dict(case=sid,after=after,before=before,paired_seeds=good,**condition(sid,specs[after]['mode']),
                before_metrics=bv,after_metrics=av,delta={k:av[k]-bv[k] if shared.finite(av[k]) and shared.finite(bv[k]) else None for k in numeric}))
    return per_case,aggregate,paired

def compare(a):
    shared.load_engine(ENGINE)
    out=a.output_root;imports=read(out/'common_input/import_details.json');specs=imports['arm_specs'];cases=imports['requested_cases'];seeds=imports['requested_seeds']
    index=read(out/'comparison_inputs/panel_index.json');temporal={(r['case'],r['arm'],r['seed']):r for r in read(out/'temporal_order/sidecar/draw_metrics.json') if r['status']=='ok'}
    cached={(r['case'],r['arm'],r['seed']):r for r in imports.get('cached_new_draws',[])}
    roots={str(P1),*[s['source_root'] for s in specs.values() if not s['new']],*[r['evaluation'] for r in cached.values()]}
    tables={root:{key:old.read_table(Path(root)/f'comparison/{key}.csv') if (Path(root)/f'comparison/{key}.csv').exists() else []
        for key in ('draw_metrics','residue_motion','motion_frames','seam_frames')} for root in roots}
    costs={(r['case'],r['arm'],r['seed']):r for r in imports['sampling']}
    rows=[];frames=[];residues=[];seams=[];slices=[];errors=list(index['errors']);missing=[];ligand_definitions={}
    for case in index['cases']:
        sid=case['id'];reference=shared.load_npz(case['reference_npz']);refgeo=read(case['reference_local_geometry_json']);context=shared.reference_context(reference)
        work=[('MD',None,None)]+[(m['name'],seed,next((d for d in m['draws'] if d['seed']==seed),None)) for m in case['models'] for seed in seeds]
        for arm,seed,draw in work:
            ident=dict(case=sid,dataset=condition(sid,'joint')['dataset'],arm=arm,seed=seed)
            spec=specs.get(arm);mode=spec['mode'] if spec else 'MD_reference'
            try:
                if arm!='MD' and draw is None:
                    missing.append(dict(ident,new=spec['new'],reason='No completed valid draw; historical absence is not a new failure'))
                    rows.append(dict(ident,**{k:v for k,v in condition(sid,mode).items() if k!='dataset'},status='NA',reason=missing[-1]['reason']));continue
                xyz=reference if arm=='MD' else shared.load_npz(draw['npz']);shared.matching(reference,xyz)
                geo=refgeo if arm=='MD' else read(draw['local_geometry_json']);prior=cached.get((sid,arm,seed));reuse=arm=='MD' or not spec['new'] or prior is not None
                if reuse:
                    root=str(P1) if arm=='MD' else prior['evaluation'] if prior else spec['source_root'];source_arm='MD' if arm=='MD' else arm if prior else spec['source_arm']
                    source=next(r for r in tables[root]['draw_metrics'] if (r['case'],r['arm'],r['seed'])==(sid,source_arm,seed) and r['status']=='ok')
                    row=dict(source,**ident)
                    copied={key:[dict(r,**ident) for r in tables[root][key] if (r['case'],r['arm'],r['seed'])==(sid,source_arm,seed)] for key in ('residue_motion','motion_frames','seam_frames')}
                    rr,ff,sf=(copied[k] for k in ('residue_motion','motion_frames','seam_frames'))
                    if not sf:
                        extra,sf=old.seam_metrics(xyz,context,ident);row.update(extra)
                    row['reused_from']=str(Path(root)/'comparison/draw_metrics.csv')
                else:
                    motion,rr,ff=shared.supplemental(xyz,geo,context,ident);extra,sf=old.seam_metrics(xyz,context,ident)
                    row=dict(ident,status='ok',**shared.existing_metrics(read(draw['metrics_json']),geo,reference['time_ps']),**motion,**extra)
                row.update(condition(sid,mode),npz=case['reference_npz'] if arm=='MD' else draw['npz'],
                    local_geometry_json=case['reference_local_geometry_json'] if arm=='MD' else draw['local_geometry_json'],
                    checkpoint=None if arm=='MD' else spec['checkpoint'],checkpoint_step=None if arm=='MD' else spec['checkpoint_step'])
                if draw:row['metrics_json']=draw['metrics_json']
                order=temporal[(sid,arm,seed)];target=temporal[(sid,'MD',None)]
                row.update({'temporal.'+k:v for k,v in order['metrics'].items()})
                row.update({'temporal.'+k+'_absolute_MD_error':abs(v-target['metrics'][k]) if shared.finite(v) and shared.finite(target['metrics'].get(k)) else None for k,v in order['metrics'].items() if k.endswith('order_gain')})
                ligand,definition=ligand_motion(xyz,reference,context);row.update(ligand)
                if definition:ligand_definitions[sid]=definition
                cost=costs.get((sid,arm,seed))
                if cost:row.update({'sampling.total_NFE':cost['sampling_info']['total_NFE'],'sampling.atom_frame_evaluations':cost['sampling_info']['atom_frame_evaluations']})
                rows.append(row);frames.extend(ff);residues.extend(rr);seams.extend(sf);slices.extend(old.geometry_slices(geo,ident))
            except Exception as exc:
                error=dict(ident,status='NA',reason=f'{type(exc).__name__}: {exc}');errors.append(error);rows.append(dict(error,**{k:v for k,v in condition(sid,mode).items() if k!='dataset'}))
    lookup={(r['case'],r['arm'],r['seed']):r for r in rows}
    for row in rows:
        if row['status']!='ok':continue
        md=lookup[(row['case'],'MD',None)];spec=specs.get(row['arm']);joint=lookup.get((row['case'],spec['stage']+'_joint',row['seed'])) if spec else None
        for key in [k for k in row if k.startswith('seam.') and not k.endswith(('count','fraction','positions'))]:
            row[key+'_minus_MD_same_positions']=row[key]-md[key] if shared.finite(row[key]) and shared.finite(md.get(key)) else None
            row[key+'_minus_joint_same_positions']=row[key]-joint[key] if joint and shared.finite(row[key]) and shared.finite(joint.get(key)) else None
    per_case,aggregate,paired=summarize(rows,specs,cases,seeds)
    newrows=[r for r in rows if r['arm']!='MD' and specs[r['arm']]['new']];oldrows=[r for r in rows if r['arm']!='MD' and not specs[r['arm']]['new']]
    available=sum(r['status']=='ok' for r in newrows);complete=available==imports['expected_new_draws'] and not imports['missing'] and not errors
    dest=out/'comparison';dest.mkdir(exist_ok=False)
    for name,items in [('draw_metrics',rows),('residue_motion',residues),('motion_frames',frames),('seam_frames',seams)]:shared.save_csv(dest/f'{name}.csv',items)
    summary=dict(study='ligand4096',completion_status='complete' if complete else 'incomplete',
        full_study_complete=complete and imports['checkpoint_step']==4096 and imports['expected_new_draws']==96,
        expected_new_draws=imports['expected_new_draws'],available_new_draws=available,
        expected_reference_draws=len(oldrows),available_reference_draws=sum(r['status']=='ok' for r in oldrows),
        errors=errors,missing=missing,import_missing=imports['missing'],arm_specs=specs,baseline_capability=imports['baseline_capability'],
        absolute_per_case=per_case,absolute_per_condition=aggregate,paired_per_case=paired,
        frame_slice_summary=old.frame_slices(frames),geometry_frame_slices=slices,ligand_motion_definitions=ligand_definitions,
        aggregation='Never pool clock/dataset/protocol/reference windows. Seed mean within case; equal cases per full condition. Paired tables use common available seeds only.',
        amplitude_interpretation='Report actual ratios; closeness to MD uses established abs(log(ratio)), not larger-is-better. Internal motion requires valid geometry.',
        geometry_policy='Original fixed MD50 topology/thresholds/calibration. Existing old common/geometry/motion/G/D values reused.',
        slice_definition='Same global fit and physical positions on MD/joint/rollout. Bad-atom counts and bad-frame fractions are not feature anomaly fractions. Q12 fifth block has only2 main-window frames.',
        caveats=['ATLAS has no ligand and never enters ligand means','Single MD window and two draw seeds are not independent training replicas','Ligand self-fit removes whole-ligand rotation; internal pairs do not establish chirality','Pocket displacement uses fixed MD-defined heavy-atom centroids, not mass COM or dynamically selected contacts'])
    write_json(dest/'comparison_summary.json',summary);write_json(dest/'viewer_panel_index.json',dict(index,errors=errors))
    if a.export_viewer:
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        assets=MAIN_RUN/'viewer_assets';viewer=out/'comparison_viewer';export_panel_viewer(dest/'viewer_panel_index.json',viewer,viewer_js=assets/'3Dmol-official.js')
        for name in ('LICENSE','3Dmol-min.js.LICENSE.txt','source.json'):shutil.copyfile(assets/name,viewer/name)
    print({'new':available,'expected_new':imports['expected_new_draws'],'errors':errors,'complete':complete})
    if not complete and not a.allow_partial:raise SystemExit('Explicit partial output required; existing rows retained')

def distance(out):
    from types import SimpleNamespace
    h=load_module('ligand_original_distance',CAUSAL/'tasks/flow-causal-20261022/history_distance.py')
    imports=read(out/'common_input/import_details.json');specs=imports['arm_specs'];side=out/'temporal_order/sidecar'
    selected=out/'distance_new_inputs';selected.mkdir(exist_ok=False)
    for name in ('summary.json','references.json'):shutil.copyfile(side/name,selected/name)
    cached={(r['case'],r['arm'],r['seed']):r for r in imports.get('cached_new_draws',[])}
    write_json(selected/'draw_metrics.json',[r for r in read(side/'draw_metrics.json') if r['arm'] in specs and specs[r['arm']]['new'] and (r['case'],r['arm'],r['seed']) not in cached])
    h.run(SimpleNamespace(sidecar=selected,family='ligand4096',index=None,output=out/'distance_new'))
    result=read(out/'distance_new/results.json');trajs=result['trajectories'];seen=set()
    old_sources=(P1/'distance/results.json',CAUSAL/'runs/flow_causal_20261022/history_distance/results.json',*[Path(p)/'distance/results.json' for p in sorted({r['evaluation'] for r in cached.values()})])
    for source in old_sources:
        for original in read(source)['trajectories']:
            if original['case'] not in imports['requested_cases']:continue
            aliases=[]
            if original['is_reference']:
                if original['dt_ps']!=(100. if original['case'].startswith('atlas_') else 80.):continue
                aliases=[dict(family='ligand4096',arm='MD',source_arm='MD',role='reference',case=original['case'],seed=None)]
            elif original['seed'] in imports['requested_seeds']:
                for name,spec in specs.items():
                    source_arm=name if spec['new'] else spec['source_arm']
                    if spec['new'] and (original['case'],name,original['seed']) not in cached:continue
                    if any(a['arm']==source_arm and (source==old_sources[0] or a['family'] in ('capacity','ligand4096')) for a in original['aliases']):
                        aliases.append(dict(family='ligand4096',arm=name,source_arm=source_arm,role='standard',case=original['case'],seed=original['seed']))
            aliases=[a for a in aliases if (a['case'],a['arm'],a['seed']) not in seen]
            if not aliases:continue
            seen.update((a['case'],a['arm'],a['seed']) for a in aliases)
            row=copy.deepcopy(original);row.update(id=f'reused_{len(trajs):04d}',aliases=aliases,reused_from=str(source));trajs.append(row)
    cases,aggregate=h.aggregate_curves(trajs)
    for row in cases+aggregate:row.update(generation_protocol=specs[row['arm']]['mode'] if row['arm']!='MD' else 'MD_reference',history_frames=4,history_interval_ps=row['dt_ps'],reference_window='original_short50_after_H4')
    dest=out/'distance';dest.mkdir(exist_ok=False)
    summary=dict(result['summary'],plots=[],unique_generated_trajectories=sum(not t['is_reference'] for t in trajs),unique_MD_references=sum(t['is_reference'] for t in trajs),
        old_generated_computations=0,new_MD_computations=0,reused_trajectories=sum('reused_from' in t for t in trajs),
        G_identity_max_error=max((v['G_identity_max_error'] or 0 for t in trajs for v in t['observables'].values()),default=0))
    write_json(dest/'results.json',dict(summary=summary,trajectories=trajs,per_case=cases,aggregate=aggregate));write_json(dest/'summary.json',summary)
    for name,items in [('per_case',cases),('aggregate',aggregate),('key_lag_summary',[r for r in aggregate if r['lag_frames'] in h.KEY_LAGS])]:h.csv_rows(dest/(name+'.csv'),items)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--merge-panel',action='store_true');p.add_argument('--distance',action='store_true')
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--export-viewer',action='store_true');a=p.parse_args();cpu_only()
    if a.merge_panel:merge(a.output_root)
    elif a.distance:distance(a.output_root)
    else:compare(a)

if __name__=='__main__':main()
