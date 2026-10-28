"""Finite kernel evaluation: endpoint ensembles separate from genuine trajectories."""
import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
from prepare_evaluation import (RUN, MAIN, MAIN_RUN, ENGINE, PARENT_EVAL, ENDPOINT_ORIGINS, read, write_json,
                                cpu_only, prepare_case, import_file)
from endpoint_metrics import features, energy_score, distribution_distance, distance_curve
from molvid.evaluation.common_panel import load_engine, score_draw
from molvid.evaluation.local_geometry import prepare_local_geometry
from temporal_eval.metrics import score_path
sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
import compare_geometry_continuation as shared
sys.path.insert(0,'/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022')
old=import_file('kernel_original_seams',Path('/workspace/MOLVID-FLOW-INPAINT-20261022/tasks/flow-inpaint-20261022/compare_inpaint.py'))
KEYS=('dataset','family','protocol','dt_ps','history_frames','query_frames','last_observed_full64_index')


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def identity(entry,arm,seed=None):
    return dict(case=entry['id'],base_case=entry['base_case'],arm=arm,seed=seed,
                **{k:entry[k] for k in KEYS},lag_multiplier=entry.get('lag_multiplier'),
                reference_window=entry['query_full64_indices'],history_time_ps=entry['observed_time_ps'])


class References:
    def __init__(self):
        self.cache={}
        self.templates={e['id']:e for e in read(MAIN_RUN/'short50_generated_20261017/manifest_ready.json')['cases']}
        self.previous={c['id']:c for c in read(PARENT_EVAL/'comparison/viewer_panel_index.json')['cases']}
    def get(self,base):
        if base not in self.cache:
            prepared=prepare_case(self.templates[base],MAIN_RUN/'short50_generated_20261017')
            previous=self.previous[base];geometry=read(previous['reference_local_geometry_json'])
            scorer=prepare_local_geometry(prepared.reference,prepared.topology,thresholds=geometry['thresholds'])
            self.cache[base]=(scorer,shared.reference_context(npz(previous['reference_npz'])))
        return self.cache[base]


def averages(rows,keys,metrics_key=None):
    grouped=defaultdict(list)
    for row in rows:
        if row.get('status')=='ok':grouped[tuple(row.get(k) for k in keys)].append(row)
    result=[]
    for key,rr in grouped.items():
        maps=[r.get(metrics_key,{}) if metrics_key else r for r in rr]
        names=sorted({k for m in maps for k,v in m.items() if shared.finite(v) and (metrics_key or k.startswith(('common.','geometry.','motion.','shape.','local_bad.','temporal.','seam.','ligand_motion.','CA.','internal.','D.')))})
        result.append(dict(zip(keys,key),status='ok',samples=len(rr),metrics={k:shared.mean([m.get(k) for m in maps]) for k in names},
                           metric_n={k:sum(shared.finite(m.get(k)) for m in maps) for k in names}))
    return result


def endpoints(out):
    panel=read(out/'imported/endpoint_panel.json');dest=out/'endpoints';dest.mkdir(exist_ok=False)
    refs=References();rows=[];groups=[];errors=[];md_cache={}
    for case in panel['cases']:
        entry=case['entry'];reference=npz(case['reference_npz']);rep=read(entry['representation']);scorer,_=refs.get(entry['base_case'])
        y=features(reference['positions'],reference['observed'][-1],rep)
        mdkey=(entry['base_case'],entry['last_observed_full64_index'],entry['lag_multiplier'])
        if mdkey not in md_cache:
            path=dest/'MD_geometry'/f'{entry["base_case"]}_t{mdkey[1]}_lag{mdkey[2]}.json';path.parent.mkdir(exist_ok=True)
            write_json(path,scorer.evaluate(reference['positions']));md_cache[mdkey]=str(path)
        for model in case['models']:
            vectors={'CA':[],'internal':[]};seeds=[];numeric=[]
            for draw in model['draws']:
                ident=identity(entry,model['name'],draw['seed'])
                try:
                    data=npz(draw['npz']);shared.matching(reference,data)
                    if data['positions'].shape[0]!=1:raise ValueError('Endpoint must contain exactly one real future frame')
                    geo=scorer.evaluate(data['positions']);path=Path(draw['npz']).parent/'local_geometry.json';write_json(path,geo)
                    draw['local_geometry_json']=str(path)
                    metric=shared.existing_metrics({'raw':{}},geo,data['time_ps']);phi=features(data['positions'],reference['observed'][-1],rep)
                    for name in vectors:vectors[name].append(phi[name][0])
                    seeds.append(draw['seed']);numeric.append(metric)
                    rows.append(dict(ident,status='ok',npz=draw['npz'],local_geometry_json=str(path),**metric))
                except Exception as exc:errors.append(dict(ident,reason=repr(exc)));rows.append(dict(ident,status='NA',reason=repr(exc)))
            group=dict(identity(entry,model['name']),status='ok' if len(seeds)==4 else 'partial',actual_draws=len(seeds),seeds=seeds,
                       expected_seeds=model['requested_seeds'],missing=model['missing_draws'],MD_geometry_json=md_cache[mdkey],metrics={})
            for name,values in vectors.items():
                score=energy_score(np.asarray(values),y[name][0]) if values else dict(status='NA',reason='No valid endpoint draw',draws=0)
                group[name]=score
                group['metrics'].update({name+'.'+k:v for k,v in score.items() if shared.finite(v)})
            names={k for r in numeric for k in r if k.startswith('geometry.') and shared.finite(r[k])}
            group['metrics'].update({k:shared.mean([r.get(k) for r in numeric]) for k in names})
            groups.append(group)
    # Origins have one heldout MD endpoint each, not independent MD replicates.
    origin_rows=[g for g in groups if g['status']=='ok' and g['actual_draws']==4]
    case_keys=['base_case','dataset','arm','history_frames','dt_ps','lag_multiplier']
    domain_keys=['dataset','arm','history_frames','dt_ps','lag_multiplier']
    by_case=averages(origin_rows,case_keys,'metrics')
    for row in by_case:
        selected=[g for g in groups if all(g[k]==row[k] for k in case_keys)]
        counts={origin:0 for origin in ENDPOINT_ORIGINS}
        counts.update({g['last_observed_full64_index']:g['actual_draws'] for g in selected})
        complete_origins=sorted(g['last_observed_full64_index'] for g in selected
                                if g['status']=='ok' and g['actual_draws']==4)
        row.update(actual_origin_indices=complete_origins,actual_origin_count=len(complete_origins),
                   planned_origin_indices=list(ENDPOINT_ORIGINS),planned_origin_count=len(ENDPOINT_ORIGINS),
                   origin_draw_counts={str(k):v for k,v in sorted(counts.items())},
                   missing_or_partial_origin_indices=[k for k in ENDPOINT_ORIGINS if k not in complete_origins],
                   origin_coverage_fraction=len(complete_origins)/len(ENDPOINT_ORIGINS),
                   full_origin_coverage=len(complete_origins)==len(ENDPOINT_ORIGINS),
                   main_summary_ensemble_draws_per_origin=4)
    by_domain=averages(by_case,domain_keys,'metrics')
    for row in by_domain:
        included=[r for r in by_case if all(r[k]==row[k] for k in domain_keys)]
        row.update(actual_cases=[r['base_case'] for r in included],actual_case_count=len(included),
                   case_origin_coverage={r['base_case']:{k:r[k] for k in
                       ('actual_origin_indices','actual_origin_count','planned_origin_indices','planned_origin_count',
                        'origin_draw_counts','full_origin_coverage')} for r in included},
                   main_summary_ensemble_draws_per_origin=4)
    result=dict(status='completed' if all(g['status']=='ok' for g in groups) and not errors else 'partial',groups=groups,draw_rows=rows,
                per_case=by_case,per_dataset=by_domain,errors=errors,available_new_draws=sum(r['status']=='ok' for r in rows),
                expected_new_draws=sum(len(m['requested_seeds']) for c in panel['cases'] for m in c['models']),
                interpretation='Independent Q1 endpoints: no RMSF/ACF/G/D or concatenation. ES uses single heldout MD endpoint; four generated draws do not reveal the full conditional distribution. Partial groups retain actual n and do not count as complete.',
                geometry='Original MD50 topology/normal bands, independent one-frame geometry per draw; ATLAS ligand NA.',
                aggregation='Main summaries include only complete four-draw origins. Equal available complete origins within base case, then equal cases within dataset/history/actual lag; origin indices/counts and actual case membership are explicit. Two/three-draw estimates remain only in groups, excluded from main per_case/per_dataset summaries.')
    write_json(dest/'summary.json',result);shared.save_csv(dest/'draw_metrics.csv',rows)
    write_json(dest/'panel_index.json',panel)
    return result


def variable_motion(data,reference,context,ident):
    target=context['reference']['positions'][0]
    a=shared.representations(shared.align(data['positions'],target,context['mask']),context['groups'])
    b=shared.representations(shared.align(reference['positions'],target,context['mask']),context['groups'])
    observed=shared.align(reference['observed'][-1:],target,context['mask'])[0]
    start={'ca':observed[context['groups'][:,1]],'centroid':observed[context['groups']].mean(1)}
    metrics={};frames=[]
    for region,mask in context['regions'].items():
        if not mask.any():continue
        for name in ('ca','centroid'):
            x,y=a[name][:,mask],b[name][:,mask];rx,ry=shared.rmsf(x),shared.rmsf(y)
            ac,mdac=shared.vector_acf(x),shared.vector_acf(y);prefix=f'motion.{name}.{region}.'
            metrics.update({prefix+'rmsf_mean_A':float(rx.mean()),prefix+'reference_rmsf_mean_A':float(ry.mean()),
                            prefix+'amplitude_ratio':shared.ratio(rx.mean(),ry.mean()),prefix+'pearson':shared.pearson(rx,ry),
                            prefix+'vector_acf_rmse':None if ac is None or mdac is None else float(np.sqrt(np.mean((ac-mdac)**2)))})
            for k in (1,2,3):metrics[prefix+f'vector_acf_lag{k}']=None if ac is None else float(ac[k-1])
            centered=x-x.mean(0)
            for i in range(len(x)):
                frames.append(dict(ident,representation=name,region=region,frame=i,time_ps=float(data['time_ps'][i]),
                                   displacement_from_first_A=float(np.sqrt(np.square(x[i]-x[0]).sum(-1).mean())),
                                   displacement_from_last_observed_A=float(np.sqrt(np.square(x[i]-start[name][mask]).sum(-1).mean())),
                                   centered_motion_rms_A=float(np.sqrt(np.square(centered[i]).sum(-1).mean())),
                                   mean_step_A=None if i==0 else float(np.linalg.norm(x[i]-x[i-1],axis=-1).mean())))
    return metrics,frames


def parent_tables():
    table={(r['case'],r['arm'],r['seed']):r for r in old.read_table(PARENT_EVAL/'comparison/draw_metrics.csv') if r['status']=='ok'}
    temporal={(r['case'],r['arm'],r['seed']):r['metrics'] for r in read(PARENT_EVAL/'temporal_order/sidecar/draw_metrics.json') if r['status']=='ok'}
    distance={}
    for item in read(PARENT_EVAL/'distance/results.json')['trajectories']:
        for a in item['aliases']:distance[(a['case'],a['arm'],a.get('seed'))]=item['observables']
    frames=old.read_table(PARENT_EVAL/'comparison/motion_frames.csv')
    seams=old.read_table(PARENT_EVAL/'comparison/seam_frames.csv')
    return table,temporal,distance,frames,seams


def ligand_motion(data,reference,context):
    """Unchanged ligand4096 comparison definition, now accepting the actual Q.

    Source: MOLVID-FLOW-LIGAND-4096-20261026/tasks/flow-ligand-4096-20261026/
    compare_results.py:ligand_motion. This small function is retained locally
    because importing that task module would import its different prepare module.
    """
    heavy=~np.isin(np.char.upper(reference['elements'].astype(str)),['H','D'])
    ligand=reference['ligand_mask'].astype(bool)&heavy
    if ligand.sum()<3:return {'ligand_motion.available':False,'ligand_motion.reason':'protein_only or fewer than three ligand heavy atoms'}
    target=reference['positions'][0];protein=(~reference['ligand_mask'].astype(bool))&heavy
    near=np.linalg.norm(target[protein,None]-target[None,ligand],axis=-1).min(1)<=5.
    pocket=np.flatnonzero(protein)[near]
    aligned=shared.align(data['positions'],target,context['mask'])
    ref=shared.align(reference['positions'],target,context['mask'])
    self_xyz=shared.align(data['positions'],target,ligand)[:,ligand]
    self_ref=shared.align(reference['positions'],target,ligand)[:,ligand]
    left,right=np.triu_indices(int(ligand.sum()),1)
    def measures(coords,self_coords):
        atoms=coords[:,ligand];dist=np.linalg.norm(atoms[:,left]-atoms[:,right],axis=-1)
        result=dict(internal_self_aligned_RMSF_mean_A=float(shared.rmsf(self_coords).mean()),
                    internal_pair_distance_time_std_mean_A=float(dist.std(0).mean()))
        if len(pocket):
            relative=atoms.mean(1)-coords[:,pocket].mean(1)
            start=shared.align(reference['observed'][-1:],target,context['mask'])[0]
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
    return values


def trajectories(out):
    panel=read(out/'imported/trajectory_panel.json');dest=out/'trajectories';dest.mkdir(exist_ok=False)
    refs=References();parent,pg,pd,pframes,pseams=parent_tables()
    rows=[];frames=[];seams=[];slices=[];orders=[];distances=[];errors=[]
    for case in panel['cases']:
        entry=case['entry'];reference=npz(case['reference_npz']);prepared=prepare_case(entry,RUN/'eval_inputs')
        rep=read(entry['representation']);scorer,context=refs.get(entry['base_case'])
        work=[('MD',None,None)]+[(m['name'],d['seed'],d) for m in case['models'] for d in m['draws']]
        mdorder=None
        for arm,seed,draw in work:
            ident=identity(entry,arm,seed)
            try:
                data=reference if draw is None else npz(draw['npz']);shared.matching(reference,data)
                reused=(arm=='parent_L4096') or (arm=='MD' and entry['family']=='standard_q50')
                geo_path=case.get('reference_local_geometry_json') if draw is None else draw.get('local_geometry_json')
                if reused:
                    source_arm='MD' if arm=='MD' else draw['source_arm'];key=(entry['base_case'],source_arm,seed)
                    row=dict(parent[key],**ident,status='ok',reused_from=str(PARENT_EVAL))
                    order=pg[key]
                    geo=read(geo_path)
                    ff=[dict(r,**ident) for r in pframes if (r['case'],r['arm'],r['seed'])==key]
                    sf=[dict(r,**ident) for r in pseams if (r['case'],r['arm'],r['seed'])==key]
                    for name,values in pd[key].items():
                        for i,value in enumerate(values['D_A2']):
                            distances.append(dict(ident,observable=name,lag_frames=i+1,lag_ps=(i+1)*entry['dt_ps'],n_pairs=values['n_pairs'][i],
                                                  D_A2=value,shuffle_platform_A2=values['shuffle_platform_A2'],D_over_shuffle=values['D_over_shuffle'][i],
                                                  G_identity_abs_error=None,reused_from=str(PARENT_EVAL/'distance/results.json')))
                else:
                    if not geo_path:
                        directory=dest/'MD_geometry' if draw is None else Path(draw['npz']).parent
                        directory.mkdir(exist_ok=True)
                        geo_path=str(directory/(entry['id']+'.json' if draw is None else 'local_geometry.json'))
                        geo=scorer.evaluate(data['positions']);write_json(Path(geo_path),geo)
                    else:geo=read(geo_path)
                    metrics={'raw':{}}
                    if draw is not None:
                        score_dir=Path(draw['npz']).parent
                        metrics_path=score_dir/'metrics.json'
                        if metrics_path.exists():metrics=read(metrics_path)
                        else:
                            rawdraw=dict(coordinates=data['positions'],time_ps=data['time_ps'],history_frames=entry['history_frames'],seed=seed,
                                         reference_indices=np.arange(entry['query_frames']),identity_source='validated original cache record',
                                         cache=draw.get('raw',draw['npz']))
                            metrics=score_draw(prepared,rawdraw,arm,entry['seeds'].index(seed),score_dir,read(RUN/'eval_inputs/manifest.json')['analysis'])
                        draw['metrics_json']=str(metrics_path)
                    if len(data['positions'])==50:
                        motion,_,ff=shared.supplemental(data,geo,context,ident)
                    else:motion,ff=variable_motion(data,reference,context,ident)
                    extra,sf=old.seam_metrics(data,context,ident) if entry['family']=='standard_q50' else ({},[])
                    row=dict(ident,status='ok',**shared.existing_metrics(metrics,geo,data['time_ps']),**motion,**extra)
                    row.update(ligand_motion(data,reference,context))
                    order=score_path(data['positions'],reference['observed'][-1],np.asarray(rep['align_indices']),np.asarray(rep['ca_indices']),(1,2,3,4,5,6),(1,2,4,8))
                    distances.extend(dict(ident,**d) for d in distance_curve(data['positions'],reference['observed'][-1],rep,entry['dt_ps'],order))
                if draw is None:case['reference_local_geometry_json']=str(geo_path);mdorder=order
                else:draw['local_geometry_json']=str(geo_path)
                row.update(npz=case['reference_npz'] if draw is None else draw['npz'],local_geometry_json=str(geo_path))
                row.update({'temporal.'+k:v for k,v in order.items()})
                row.update({'temporal.'+k+'_absolute_MD_error':None if v is None or mdorder.get(k) is None else abs(v-mdorder[k]) for k,v in order.items() if k.endswith('order_gain')})
                rows.append(row);frames.extend(ff);seams.extend(sf)
                orders.append(dict(ident,status='ok',metrics=order))
                slices.extend(old.geometry_slices(geo,ident,count=entry['query_frames'],scope=entry['family']))
            except Exception as exc:
                error=dict(ident,status='NA',reason=f'{type(exc).__name__}: {exc}');errors.append(error);rows.append(error)
        for model in case['models']:
            present={d['seed'] for d in model['draws']}
            for seed in model.get('requested_seeds',[]):
                if seed not in present:
                    reason=next((d['reason'] for d in model.get('missing_draws',[]) if d['seed']==seed),'No completed valid draw')
                    rows.append(dict(identity(entry,model['name'],seed),status='NA',reason=reason))
    md_rows={r['case']:r for r in rows if r['arm']=='MD' and r['status']=='ok'}
    joint_rows={(r['base_case'],r['arm'],r['seed']):r for r in rows if r.get('family')=='standard_q50' and r.get('protocol')=='joint' and r['status']=='ok'}
    for row in rows:
        if row['status']!='ok':continue
        md=md_rows.get(row['case'],{});joint=joint_rows.get((row['base_case'],row['arm'],row['seed']),{})
        for key in [k for k in row if k.startswith('seam.') and not k.endswith(('count','fraction','positions'))]:
            if shared.finite(row[key]):
                row[key+'_minus_MD_same_positions']=row[key]-md[key] if shared.finite(md.get(key)) else None
                row[key+'_minus_joint_same_positions']=row[key]-joint[key] if shared.finite(joint.get(key)) else None
    dmd={(d['case'],d['observable'],d['lag_frames']):d for d in distances if d['arm']=='MD'}
    for d in distances:
        md=dmd.get((d['case'],d['observable'],d['lag_frames']),{})
        for key in ('D_A2','D_over_shuffle'):
            d[key+'_absolute_MD_error']=abs(d[key]-md[key]) if shared.finite(d.get(key)) and shared.finite(md.get(key)) else None
    for name,items in [('draw_metrics',rows),('motion_frames',frames),('seam_frames',seams),('distance',distances)]:shared.save_csv(dest/(name+'.csv'),items)
    write_json(dest/'geometry_slices.json',slices);write_json(dest/'temporal_order.json',orders);write_json(dest/'distance.json',distances)
    write_json(dest/'panel_index.json',panel)
    means=averages(rows,['case','base_case','arm',*KEYS]);domain=averages(means,['arm',*KEYS],'metrics')
    distance_means=averages([dict(d,status='ok',**{'D.'+k:d[k] for k in ('D_A2','shuffle_platform_A2','D_over_shuffle','D_A2_absolute_MD_error','D_over_shuffle_absolute_MD_error')}) for d in distances],
                           ['case','base_case','arm',*KEYS,'observable','lag_frames','lag_ps'])
    result=dict(status='completed' if not errors else 'partial',errors=errors,draw_rows=rows,per_case=means,per_dataset=domain,
                distance_per_case=distance_means,distance_per_dataset=averages(distance_means,['arm',*KEYS,'observable','lag_frames','lag_ps'],'metrics'),
                G_identity_max_error=max([r['G_identity_abs_error'] for r in distances if r.get('G_identity_abs_error') is not None]+[0]),
                reused_parent_draws=sum(r.get('reused_from') is not None and r['arm']!='MD' for r in rows),
                new_trajectory_draws=sum(r['status']=='ok' and r['arm'] not in ('MD','parent_L4096') for r in rows),
                geometry='Fixed original same-case MD50 bands; no new/generated calibration',
                aggregation='Seed means then equal cases within dataset, actual dt/history/Q/protocol/reference origin; no mixing clocks or protocols',
                short_window_note='Four-step compatibility chains have 3/2/1 pairs at lag1/2/3; unsupported higher lags NA. Independent endpoint samples excluded.',
                frame_note='displacement_from_first means first future frame, not last observation; no independent block fitting')
    result['ligand_motion_definition']='Existing L4096 definition: heavy-atom self-Kabsch RMSF and internal pair-distance fluctuation separate from ligand-minus-fixed-pocket arithmetic-centroid displacement; pocket from matching MD first future frame at5A. Neither internal deformation nor larger amplitude alone means physical success.'
    write_json(dest/'summary.json',result)
    return result


def compatibility(out):
    endpoint=read(out/'endpoints/panel_index.json');trajectory=read(out/'trajectories/panel_index.json')
    candidates={}
    for panel in (endpoint,trajectory):
        for case in panel['cases']:
            e=case['entry']
            if e['last_observed_full64_index']!=12 or e['base_case'] not in ('atlas_6e5y_A','4J8T'):continue
            kind='direct' if e['family']=='independent_endpoint' else e.get('mode')
            if kind not in ('direct','compat_joint','compat_steps'):continue
            ref=npz(case['reference_npz']);rep=read(e['representation'])
            for model in case['models']:
                for lag in ((e['lag_multiplier'],) if kind=='direct' else (2,4)):
                    if lag not in (2,4):continue
                    vectors={'CA':[],'internal':[]};seeds=[]
                    for draw in model['draws']:
                        if not draw.get('local_geometry_json'):continue
                        data=npz(draw['npz']);index=0 if kind=='direct' else lag-1
                        phi=features(data['positions'][index:index+1],ref['observed'][-1],rep)
                        for name in vectors:vectors[name].append(phi[name][0])
                        seeds.append(draw['seed'])
                    candidates[e['base_case'],model['name'],e['history_frames'],lag,kind]=(e,ref,rep,vectors,seeds)
    rows=[]
    for key,item in candidates.items():
        case,arm,h,lag,kind=key
        if kind!='direct':continue
        e,ref,rep,base,seeds=item;y=features(ref['positions'],ref['observed'][-1],rep)
        for other in ('compat_steps','compat_joint'):
            match=candidates.get((case,arm,h,lag,other))
            for obs in ('CA','internal'):
                value=dict(base_case=case,dataset=e['dataset'],arm=arm,history_frames=h,lag_multiplier=lag,lag_ps=lag*e['base_dt_ps'],observable=obs,contrast=other+' vs direct')
                if not match or not base[obs] or not match[3][obs]:value.update(status='NA',reason='Missing same-origin direct/composed ensemble')
                else:
                    value.update(distribution_distance(base[obs],match[3][obs],seeds,match[4]),direct_seeds=seeds,other_seeds=match[4],
                                 direct_ES=energy_score(np.asarray(base[obs]),y[obs][0]),other_ES=energy_score(np.asarray(match[3][obs]),y[obs][0]))
                rows.append(value)
    dest=out/'compatibility';dest.mkdir(exist_ok=False)
    write_json(dest/'summary.json',dict(rows=rows,errors=[],interpretation='Same true t0=12 and H1/H4 for direct and composed/joint distributions. Genuine generated history in four-step chain. Not a hard CK equality test and not paired-coordinate error.'))
    return rows


def summarize(out,export_viewer=False):
    imports=read(out/'imported/import_details.json')
    endpoint=read(out/'endpoints/summary.json') if (out/'endpoints/summary.json').exists() else {}
    traj=read(out/'trajectories/summary.json') if (out/'trajectories/summary.json').exists() else {}
    errors=imports['errors']+endpoint.get('errors',[])+traj.get('errors',[])
    if not endpoint:errors.append(dict(stage='endpoints',reason='Missing endpoint summary'))
    if not traj:errors.append(dict(stage='trajectories',reason='Missing trajectory summary'))
    if not (out/'compatibility/summary.json').exists():errors.append(dict(stage='compatibility',reason='Missing same-origin compatibility summary'))
    valid=endpoint.get('available_new_draws',0)+traj.get('new_trajectory_draws',0)
    complete=valid==imports['expected_new_draws'] and not errors
    dest=out/'comparison';dest.mkdir(exist_ok=False)
    endpoint_cases=endpoint.get('per_case',[]);trajectory_cases=traj.get('per_case',[])
    lookup={(r.get('case'),r.get('base_case'),r.get('history_frames'),r.get('dt_ps'),r['arm']):r for r in trajectory_cases}
    contrasts=[]
    for row in trajectory_cases:
        if row['arm'] in ('MD','parent_L4096'):continue
        for comparator in ('control','control256','parent_L4096'):
            if comparator==row['arm']:continue
            key=(row.get('case'),row.get('base_case'),row.get('history_frames'),row.get('dt_ps'),comparator)
            control=lookup.get(key)
            if control:
                contrasts.append(dict(case=row['case'],base_case=row['base_case'],arm=row['arm'],reference_arm=comparator,
                                      **{k:row[k] for k in KEYS},metrics={k:v-control['metrics'][k] for k,v in row['metrics'].items()
                                      if shared.finite(v) and shared.finite(control['metrics'].get(k))}))
    result=dict(status='completed' if complete else 'partial',complete=complete,
                full_study_complete=complete and imports['expected_new_draws']==imports['full_plan_new_draws'],
                expected_new_draws=imports['expected_new_draws'],available_new_draws=valid,
                imported_new_draws=imports['available_new_draws'],reused_parent_draws=imports['reused_parent_draws'],
                missing=imports['missing'],errors=errors,endpoint_per_case=endpoint_cases,endpoint_per_dataset=endpoint.get('per_dataset',[]),
                trajectory_per_case=trajectory_cases,trajectory_per_dataset=traj.get('per_dataset',[]),matched_control_contrasts=contrasts,
                endpoint_summary=str(out/'endpoints/summary.json'),trajectory_summary=str(out/'trajectories/summary.json'),
                compatibility_summary=str(out/'compatibility/summary.json'),G_identity_max_error=traj.get('G_identity_max_error'),
                aggregation='No cross-dataset/clock/history/Q/protocol averaging; distinct endpoint origins averaged only after ensemble scoring. Missing arms/draws NA; no padding or endpoint concatenation.',
                parent_reuse='Only old L4096 standard48 same-condition scores/raw/G/D are copied, no old model generation or rescoring')
    if export_viewer and (out/'trajectories/panel_index.json').exists():
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        panel=read(out/'trajectories/panel_index.json')
        for c in panel['cases']:
            for m in c['models']:
                m['draws']=[d for d in m['draws'] if d.get('local_geometry_json')]
                if not m['draws']:m.update(status='NA',reason='No valid completed trajectory')
        write_json(dest/'viewer_panel_index.json',panel)
        export_panel_viewer(dest/'viewer_panel_index.json',out/'viewer',viewer_js=MAIN_RUN/'viewer_assets/3Dmol-official.js')
        result['viewer']=str(out/'viewer/index.html')
    write_json(dest/'comparison_summary.json',result)
    print(__import__('json').dumps({k:result[k] for k in ('status','available_new_draws','expected_new_draws','reused_parent_draws')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=RUN/'evaluation')
    p.add_argument('--stage',choices=['endpoints','trajectories','compatibility','summarize'],required=True)
    p.add_argument('--export-viewer',action='store_true');a=p.parse_args();cpu_only();load_engine(ENGINE)
    if a.stage=='summarize':summarize(a.output_root,a.export_viewer)
    else:globals()[a.stage](a.output_root)


if __name__=='__main__':main()
