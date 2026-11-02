"""Existing Q50 scientific metrics on the new common-origin H/Q references."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import prepare_evaluation as scope
sys.path.insert(0,str(scope.KERNEL/'tasks/flow-kernel-20261027'))
common=scope.import_file('hq_existing_metrics',scope.KERNEL/'tasks/flow-kernel-20261027/compare_results.py')
shared=common.shared
KEYS=('dataset','phase','setting','history_frames','block_size','dt_ps','query_frames','last_observed_full64_index')

def identity(entry,phase,seed=None):
    return dict(case=entry['id'],base_case=entry['base_case'],arm=phase,seed=seed,phase=phase,setting=entry['setting'],
        dataset=entry['dataset'],history_frames=entry['history_frames'],block_size=entry['block_size'],dt_ps=entry['dt_ps'],
        query_frames=50,last_observed_full64_index=11,reference_window=entry['query_full64_indices'],raw_query_frames=entry['raw_query_frames'])

def positions_slices(count=50):
    result=[('early10',list(range(min(10,count))),True),('late10',list(range(max(0,count-10),count)),True),
        ('anchor',list(range(4,count,5)),True),('interior',[i for i in range(count) if i%5!=4],True)]
    for size in (4,8,12):
        boundary=set(range(size,count,size))
        result.extend([(f'q{size}_boundary_adjacent',sorted({i for b in boundary for i in (b-1,b)}),True),
            (f'q{size}_nonboundary',[i for i in range(count) if i not in {j for b in boundary for j in (b-1,b)}],True)])
        for start in range(0,count,size):
            stop=min(start+size,count)
            result.append((f'q{size}_block{start//size+1}',list(range(start,stop)),stop-start==size))
    return result


def seam_metrics(trajectory,context,identity):
    """Same global fit for all frames; identical physical edge masks on every arm/MD."""
    import numpy as np
    xyz=shared.align(trajectory['positions'],context['reference']['positions'][0],context['mask'])
    ca=xyz[:,context['groups'][:,1]];centered=ca-ca.mean(0)
    dt=np.diff(trajectory['time_ps'])
    if len(ca)!=50 or not np.allclose(dt,dt[0]) or dt[0]<=0:raise ValueError('Expected native uniform Q50 physical time')
    velocity=np.diff(ca,axis=0)/dt[:,None,None]
    step=np.linalg.norm(np.diff(ca,axis=0),axis=-1).mean(1)
    speed=np.linalg.norm(velocity,axis=-1).mean(1)
    denominator=np.mean(np.sum(centered**2,axis=(1,2)))
    acf=np.sum(centered[:-1]*centered[1:],axis=(1,2))/denominator if denominator>1e-16 else np.full(49,np.nan)
    change=np.linalg.norm(np.diff(velocity,axis=0),axis=-1).mean(1)
    acceleration=change/dt[0]
    product=np.linalg.norm(velocity[:-1],axis=-1)*np.linalg.norm(velocity[1:],axis=-1)
    valid=product>1e-16
    cosine=np.divide(np.sum(velocity[:-1]*velocity[1:],axis=-1),product,out=np.full(product.shape,np.nan),where=valid)
    turn=np.arccos(np.clip(cosine,-1,1))*180/np.pi
    turn_mean=np.array([float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else np.nan for x in turn])
    destinations=np.arange(1,50);centers=np.arange(1,49);anchors=set(range(4,50,5))
    edge_masks={}
    for size in (4,8,12):
        mask=destinations%size==0
        edge_masks[f'q{size}_boundary']=mask;edge_masks[f'q{size}_within']=~mask
    touch=np.array([i in anchors or i-1 in anchors for i in destinations])
    edge_masks.update(touching_anchor=touch,other_pairs=~touch)
    result={}
    for label,mask in edge_masks.items():
        key='seam.'+label+'.'
        result[key+'pair_count']=int(mask.sum())
        for name,data in (('mean_step_A',step),('mean_speed_A_per_ps',speed),('lag1_acf',acf)):
            result[key+name]=shared.mean(data[mask].tolist())
        # A velocity change uses the two edges meeting at each center. Boundary
        # slices include either adjacent edge crossing the declared boundary.
        central=mask[:-1]|mask[1:] if label.endswith('boundary') or label=='touching_anchor' else mask[:-1]&mask[1:]
        result[key+'velocity_change_center_count']=int(central.sum())
        for name,data in (('velocity_change_A_per_ps',change),('second_difference_A_per_ps2',acceleration),('turn_degrees',turn_mean)):
            result[key+name]=shared.mean(data[central].tolist())
        result[key+'turn_valid_residue_fraction']=float(valid[central].mean()) if central.any() else None
    frames=[]
    for i,destination in enumerate(destinations):
        row=dict(identity,frame=int(destination),time_ps=float(trajectory['time_ps'][destination]),
            step_A=float(step[i]),speed_A_per_ps=float(speed[i]),lag1_acf=float(acf[i]) if np.isfinite(acf[i]) else None,
            q4_boundary=bool(destination%4==0),q8_boundary=bool(destination%8==0),q12_boundary=bool(destination%12==0),touching_anchor=bool(touch[i]))
        if destination<49:row.update(velocity_change_A_per_ps=float(change[i]),second_difference_A_per_ps2=float(acceleration[i]),
            turn_degrees=float(turn_mean[i]) if np.isfinite(turn_mean[i]) else None,turn_valid_residue_fraction=float(valid[i].mean()))
        frames.append(row)
    return result,frames


def geometry_slices(geometry,identity,count=50,scope='main_Q50'):
    result=[]
    for kind,parts in geometry['metrics'].items():
        for component,m in parts.items():
            if not m.get('available'):continue
            bad=m.get('bad_atoms_by_frame',[])
            if len(bad)!=count:raise ValueError('Geometry frame coverage mismatch')
            for label,indices,complete in positions_slices(count):
                counts=[len(bad[i]) for i in indices]
                result.append(dict(identity,scope=scope,check=kind,component=component,slice=label,frames=len(counts),
                    complete_block=complete,mean_bad_atom_count=shared.mean(counts),bad_frame_fraction=shared.mean([float(n>0) for n in counts])))
    return result




def reuse_phase1(previous,out,panel,rows,frames,seams,slices,orders,distances):
    """Read-only merge of this study's already scored phase1, not oldt0=3."""
    source=previous/'comparison';old=scope.read(source/'comparison_summary.json')
    oldcases={c['id']:c for c in scope.read(source/'viewer_panel_index.json')['cases']}
    selected={c['id'] for c in panel['cases']}
    if any(r['arm']=='phase1' for r in rows):raise ValueError('Reuse is only for a phase2-only CPU run')
    for case in panel['cases']:
        if case['id'] not in oldcases:continue
        prior=oldcases[case['id']]
        current_ref=common.npz(case['reference_npz']);prior_ref=common.npz(prior['reference_npz'])
        for key in ('positions','observed','time_ps','atom_identity'):
            if not np.array_equal(current_ref[key],prior_ref[key]):raise ValueError('Phase1 reuse reference/H/clock mismatch')
        for model in prior['models']:
            if model['name']=='phase1':case['models'].append(copy.deepcopy(model))
    def take(values):
        return [dict(r,reused_from=str(previous)) for r in values if r['case'] in selected and r['arm']=='phase1']
    extra=take(old['draw_rows']);rows.extend(extra)
    for name,values in [('geometry_slices',slices),('temporal_order',orders),('distance',distances)]:values.extend(take(scope.read(source/(name+'.json'))))
    frames.extend(take(common.old.read_table(source/'motion_frames.csv')))
    seams.extend(take(common.old.read_table(source/'seam_frames.csv')))
    return sum(r['status']=='ok' for r in extra),[r for r in old['errors'] if r.get('case') in selected],[r for r in old['missing'] if r.get('case') in selected]


def compare(out,export_viewer,reuse_evaluation=None):
    imports=scope.read(out/'imported/import_details.json');panel=scope.read(out/'imported/panel_index.json')
    dest=out/'comparison';dest.mkdir(exist_ok=False);common.load_engine(scope.ENGINE)
    rows=[];frames=[];seams=[];slices=[];orders=[];distances=[];errors=list(imports['errors']);references={}
    for case in panel['cases']:
        entry=case['entry'];reference=common.npz(case['reference_npz']);prepared=scope.prepare_case(entry,scope.RUN/'eval_inputs')
        rep=scope.read(entry['representation'])
        # Normal bands and flexible regions are recalibrated to exactly this
        # common t0=11 future50. Neither oldt0=3 nor generated coordinates enter.
        calibration_entry=dict(entry,record=entry['geometry_calibration_record'],history_frames=entry['geometry_calibration_history_frames'])
        calibration=scope.prepare_case(calibration_entry,scope.RUN/'eval_inputs')
        if not np.array_equal(calibration.reference,reference['positions']):raise ValueError('Geometry calibration differs from the new shared MD50 window')
        key=entry['base_case']
        if key not in references:
            scorer=common.prepare_local_geometry(calibration.reference,calibration.topology)
            context=shared.reference_context(reference)
            geo=scorer.evaluate(calibration.reference);path=dest/(key+'_MD_local_geometry.json');scope.write_json(path,geo)
            references[key]=(scorer,context,geo,path,reference['positions'].copy(),prepared.observed[-1].copy())
        scorer,context,mdgeo,mdgeo_path,mdxyz,observed=references[key]
        if not np.array_equal(mdxyz,prepared.reference) or not np.array_equal(observed,prepared.observed[-1]):raise ValueError('H/Q settings do not share physical origin/reference')
        mdorder=None
        work=[('MD',None,None)]+[(m['name'],d['seed'],d) for m in case['models'] for d in m['draws']]
        for phase,seed,draw in work:
            ident=identity(entry,phase,seed)
            try:
                data=reference if draw is None else common.npz(draw['npz']);shared.matching(reference,data)
                metrics={'raw':{}}
                if draw is None:geo=mdgeo;geo_path=mdgeo_path
                else:
                    directory=Path(draw['npz']).parent;geo_path=directory/'local_geometry.json'
                    geo=scorer.evaluate(data['positions']);scope.write_json(geo_path,geo)
                    rawdraw=dict(coordinates=data['positions'],time_ps=data['time_ps'],history_frames=entry['history_frames'],seed=seed,
                        reference_indices=np.arange(50),identity_source='validated original H/Q cache record',cache=draw['raw'])
                    metrics=common.score_draw(prepared,rawdraw,phase,entry['seeds'].index(seed),directory,scope.read(scope.RUN/'eval_inputs/manifest.json')['analysis'])
                    draw.update(local_geometry_json=str(geo_path),metrics_json=str(directory/'metrics.json'))
                motion,_,ff=shared.supplemental(data,geo,context,ident)
                extra,sf=seam_metrics(data,context,ident)
                row=dict(ident,status='ok',**shared.existing_metrics(metrics,geo,data['time_ps']),**motion,**extra)
                row.update(common.ligand_motion(data,reference,context))
                order=common.score_path(data['positions'],reference['observed'][-1],np.asarray(rep['align_indices']),np.asarray(rep['ca_indices']),(1,2,3,4,5,6),(1,2,4,8))
                distances.extend(dict(ident,**d) for d in common.distance_curve(data['positions'],reference['observed'][-1],rep,entry['dt_ps'],order))
                if draw is None:case['reference_local_geometry_json']=str(geo_path);mdorder=order
                row.update(npz=case['reference_npz'] if draw is None else draw['npz'],local_geometry_json=str(geo_path))
                row.update({'temporal.'+k:v for k,v in order.items()})
                row.update({'temporal.'+k+'_absolute_MD_error':None if v is None or mdorder.get(k) is None else abs(v-mdorder[k]) for k,v in order.items() if k.endswith('order_gain')})
                rows.append(row);frames.extend(ff);seams.extend(sf);orders.append(dict(ident,status='ok',metrics=order))
                slices.extend(geometry_slices(geo,ident,count=50,scope='first50_of_full_blocks'))
                print(json.dumps(dict(case=entry['id'],phase=phase,seed=seed,status='ok')),flush=True)
            except Exception as exc:
                bad=dict(ident,status='NA',reason=f'{type(exc).__name__}: {exc}');rows.append(bad);errors.append(bad)
        for model in case['models']:
            for missing in model.get('missing_draws',[]):rows.append(dict(identity(entry,model['name'],missing['seed']),status='NA',reason=missing['reason']))
    reused=0;prior_missing=[]
    if reuse_evaluation is not None:
        reused,prior_errors,prior_missing=reuse_phase1(reuse_evaluation,out,panel,rows,frames,seams,slices,orders,distances);errors.extend(prior_errors)
    mdrows={r['case']:r for r in rows if r['arm']=='MD' and r['status']=='ok'}
    for row in rows:
        if row['status']!='ok':continue
        md=mdrows.get(row['case'],{})
        for key in [k for k in row if k.startswith('seam.') and not k.endswith(('count','fraction','positions','_minus_MD_same_positions'))]:
            if shared.finite(row[key]):row[key+'_minus_MD_same_positions']=row[key]-md[key] if shared.finite(md.get(key)) else None
    dmd={(r['case'],r['observable'],r['lag_frames']):r for r in distances if r['arm']=='MD'}
    for row in distances:
        md=dmd.get((row['case'],row['observable'],row['lag_frames']),{})
        for key in ('D_A2','D_over_shuffle'):row[key+'_absolute_MD_error']=abs(row[key]-md[key]) if shared.finite(row.get(key)) and shared.finite(md.get(key)) else None
    means=common.averages(rows,['case','base_case','arm',*KEYS]);domains=common.averages(means,['arm',*KEYS],'metrics')
    for d in domains:
        selected=[r for r in means if all(r[k]==d[k] for k in ('arm',*KEYS))]
        d.update(actual_cases=[r['base_case'] for r in selected],actual_case_count=len(selected),case_draw_counts={r['base_case']:r['samples'] for r in selected})
    lookup={(r['base_case'],r['phase'],r['setting']):r for r in means};contrasts=[]
    settings_pairs=[('h4q8','h4q4'),('h8q4','h4q4'),('h8q8','h4q8'),('h8q8','h8q4'),('h12q12','h4q4'),('h12q12','h8q8')]
    pairs=[('phase2',s,'phase1',s) for s in scope.SETTINGS]
    pairs += [(p,a,p,b) for p in imports['phases'] for a,b in settings_pairs]
    for base_case in imports['selected_cases']:
        for phase,a,control_phase,b in pairs:
            x=lookup.get((base_case,phase,a));y=lookup.get((base_case,control_phase,b))
            if not x or not y:continue
            contrasts.append(dict(base_case=base_case,dataset=x['dataset'],dt_ps=x['dt_ps'],phase=phase,setting=a,reference_phase=control_phase,reference_setting=b,
                actual_draws=x['samples'],reference_draws=y['samples'],metrics={k:v-y['metrics'][k] for k,v in x['metrics'].items() if shared.finite(v) and shared.finite(y['metrics'].get(k))}))
    dmeans=common.averages([dict(d,status='ok',**{'D.'+k:d[k] for k in ('D_A2','shuffle_platform_A2','D_over_shuffle','D_A2_absolute_MD_error','D_over_shuffle_absolute_MD_error')}) for d in distances],['case','base_case','arm',*KEYS,'observable','lag_frames','lag_ps'])
    for name,items in [('draw_metrics',rows),('motion_frames',frames),('seam_frames',seams),('distance',distances)]:shared.save_csv(dest/(name+'.csv'),items)
    for name,items in [('geometry_slices',slices),('temporal_order',orders),('distance',distances)]:scope.write_json(dest/(name+'.json'),items)
    covered=sum(r['status']=='ok' and r['arm']!='MD' for r in rows);valid=covered-reused;complete=valid==imports['expected_new_draws'] and not errors
    result=dict(status='completed' if complete else 'partial',complete=complete,full_study_complete=complete and covered==120,reused_prior_phase_draws=reused,total_covered_draws=covered,
        available_new_draws=valid,expected_new_draws=imports['expected_new_draws'],missing=imports['missing']+prior_missing,errors=errors,
        draw_rows=rows,per_case=means,per_dataset=domains,contrasts=contrasts,distance_per_case=dmeans,
        distance_per_dataset=common.averages(dmeans,['arm',*KEYS,'observable','lag_frames','lag_ps'],'metrics'),
        G_identity_max_error=max([d['G_identity_abs_error'] for d in distances if d.get('G_identity_abs_error') is not None]+[0]),
        geometry_calibration='Exact common t0=11 true future12:62, all settings/phases; original geometry definitions and trusted chemical topology',
        geometry_slices_definition='Mean bad-atom counts and fraction of frames containing a bad atom, not feature anomaly rates; actual complete blocks flagged',
        alignment='One global reference frame for entire path, never independent per-block fitting',
        aggregation='Seed means within physical case then equal cases by phase/H/Q/dataset/dt; actual n and membership explicit; no old t0=3 reuse',
        raw_tail='Q4raw52/Q8raw56/Q12raw60 retained; only first50 scored, last scored2 not full block',
        protocol_clock='ATLAS100ps/MISATO80ps; G and D use continuous actual50future, replicas never concatenated',
        frame_note='displacement_from_first is first future frame, not last observation. RMSD growth alone is not error accumulation.')
    scope.write_json(dest/'viewer_panel_index.json',panel)
    if export_viewer:
        from molvid.evaluation.trajectory_viewer import export_panel_viewer
        export_panel_viewer(dest/'viewer_panel_index.json',out/'viewer',viewer_js=scope.MAIN_RUN/'viewer_assets/3Dmol-official.js')
        result['viewer']=str(out/'viewer/index.html')
    scope.write_json(dest/'comparison_summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=scope.RUN/'evaluation');p.add_argument('--export-viewer',action='store_true');p.add_argument('--reuse-evaluation',type=Path);a=p.parse_args();scope.cpu_only()
    r=compare(a.output_root.resolve(),a.export_viewer,a.reuse_evaluation);print(json.dumps({k:r[k] for k in ('status','available_new_draws','expected_new_draws','G_identity_max_error')}),flush=True)

if __name__=='__main__':main()
