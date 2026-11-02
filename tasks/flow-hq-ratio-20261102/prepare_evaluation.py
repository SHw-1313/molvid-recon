"""Real same-origin DEV inputs for five H/Q settings; no model calls."""
import argparse,copy,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/flow_hq_ratio_20261102'
KERNEL=Path('/workspace/MOLVID-FLOW-KERNEL-20261027')
SOURCE=KERNEL/'runs/flow_kernel_20261027/eval_inputs'
spec=importlib.util.spec_from_file_location('hq_existing_inputs',KERNEL/'tasks/flow-kernel-20261027/prepare_evaluation.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SETTINGS={'h4q4':(4,4),'h4q8':(4,8),'h8q4':(8,4),'h8q8':(8,8),'h12q12':(12,12)}

def prepare(output):
    base.cpu_only();torch.set_num_threads(2);start,cpu=time.monotonic(),time.process_time()
    out=output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Preserve fixed DEV input manifest')
    old=base.read(SOURCE/'manifest.json');templates={e['base_case']:e for e in old['standard_entries'] if e['mode']=='joint'}
    contact=base.import_file('hq_existing_contact','/workspace/molvid-misato-contact-filter-v1/molvid/data/contact_filter.py')
    entries=[];checks=[]
    for case in base.CASES:
        source=SOURCE/case/'reference64.pt';full=torch.load(source,map_location='cpu',weights_only=False)['record']
        dt=100. if case.startswith('atlas_') else 80.
        if len(full['x'])!=64 or full['split']!='valid' or not np.isfinite(full['x']).all():raise ValueError('Invalid true64 DEV reference '+case)
        if not np.array_equal(np.diff(full['time_ps']),np.full(63,dt)):raise ValueError('Physical clock mismatch '+case)
        original=torch.load(templates[case]['record_path'],map_location='cpu',weights_only=False)['record']
        fields=('atom_identity','atom_source_index','atype','btype','block_id','component_id','edge_mask','loss_mask','align_mask','bond_index')
        if any(not np.array_equal(full[k],original[k]) for k in fields):raise ValueError('Original DEV topology changed '+case)
        distances=None if case.startswith('atlas_') else contact.minimum_ligand_protein_distance(full)
        if distances is not None and (not np.isfinite(distances).all() or np.any(distances>5.)):raise ValueError(case+' violates existing all-real-frame5A selection')
        check=dict(case=case,source_reference64=str(source),real_frames=64,atoms=len(full['atype']),dt_ps=dt,
                   topology_equal=True,last_observed_full64_index=11,reference_full64_indices=list(range(12,62)),
                   minimum_contact_A_by_frame=None if distances is None else distances.tolist(),
                   contact_max_A=None if distances is None else float(distances.max()))
        checks.append(check)
        for setting,(h,q) in SETTINGS.items():
            key=case+'__'+setting;folder=out/key;folder.mkdir()
            history=list(range(12-h,12));future=list(range(12,62));rawq=((50+q-1)//q)*q
            record=base.sliced_record(full,history,future,key);path=folder/'record.pt';torch.save({'record':record},path)
            condition=copy.deepcopy(record)
            for field in ('x','bpos','frame_mask','time_ps'):condition[field]=np.asarray(record[field])[:h].copy()
            condition['delta_time_ps']=np.diff(condition['time_ps']);condition['native_frame_indices']=record['native_frame_indices'][:h]
            for field in ('rotation','translation'):condition['alignment'][field]=record['alignment'][field][:h]
            condition.update(future_frames=0,selected_full64_indices=history)
            condition_path=folder/'condition_record.pt';torch.save({'record':condition},condition_path)
            item=copy.deepcopy(templates[case]);item.update(id=key,sample_id=key,case=case,base_case=case,setting=setting,
                family='hq_rollout_q50',protocol=setting,mode=setting,record=str(path),record_path=str(path),condition_record_path=str(condition_path),
                history_frames=h,block_query_frames=q,block_size=q,query_frames=50,raw_query_frames=rawq,block_lengths=[q]*(rawq//q),
                planned_vector_field_evaluations=35*(rawq//q),models=[],seeds=[17,29],last_observed_full64_index=11,
                history_full64_indices=history,query_full64_indices=future,base_dt_ps=dt,dt_ps=dt,
                query_time_ps=(np.arange(1,51)*dt).tolist(),raw_query_time_ps=(np.arange(1,rawq+1)*dt).tolist(),
                observed_time_ps=((np.arange(h)-h+1)*dt).tolist(),source_time_ps=np.asarray(full['time_ps'])[history+future].tolist(),
                absolute_native_time_ps=(np.asarray(full['native_frame_indices'])[history+future]*float(full['native_delta_time_ps'])).tolist(),
                native_frame_indices=np.asarray(full['native_frame_indices'])[history+future].tolist(),
                reference_npz=str(folder/'reference.npz'),condition_npz=str(folder/'condition.npz'),source_reference64=str(source),
                source_manifest=str(SOURCE/'manifest.json'),extra_future_reference='No GT beyond64; known physical clock alone extrapolated for full final generated block',
                history_policy='Fixed true H ending at source11, then latest H generated/previous observed coordinates; no future access',
                geometry_calibration_record=str(path),geometry_calibration_history_frames=h,
                geometry_calibration_note='Same true source12:62 MD50 geometry normal bands for all H/Q and phases; no old t0=3 bands',
                raw_tail_note='All generated blocks complete; score only first50; truncated final scored segment is not a complete block',
                actual_generation_requested=False)
            item.pop('raw60_reference_record',None);item.pop('raw60_query_full64_indices',None)
            prepared=base.prepare_case(item,out)
            if not np.array_equal(prepared.reference,np.asarray(full['x'])[12:62]):raise ValueError('Reference coordinates altered')
            if not np.array_equal(prepared.observed,np.asarray(full['x'])[history]):raise ValueError('Condition coordinates altered')
            if not np.array_equal(prepared.time_ps,item['query_time_ps']):raise ValueError('Relative query clock altered')
            base.export_npz(folder/'reference.npz',prepared.reference,prepared.time_ps,prepared,history_frames=h)
            np.savez_compressed(folder/'condition.npz',observed=prepared.observed.astype(np.float32),observed_time_ps=item['observed_time_ps'],
                query_time_ps=item['query_time_ps'],raw_query_time_ps=item['raw_query_time_ps'],atom_identity=prepared.topology['atom_identity'],coordinate_unit='angstrom',sample_id=key)
            entries.append(item)
        print(json.dumps(check),flush=True)
    result=dict(schema='hq_ratio_same_origin_DEV_v1',status='completed',entries=entries,cases=list(base.CASES),settings=SETTINGS,
        seeds=[17,29],last_observed_full64_index=11,scored_future_full64_indices=list(range(12,62)),source_checks=checks,errors=[],
        phase1_checkpoint='/workspace/MOLVID-FLOW-Q4-ALIGN-20261030/runs/flow_q4_align_20261030/routed_q4_q1/coordinate_final.pt',
        phase1_draws=60,phase2_draws=60,phase2='Five matched trained arms, each evaluated only at its own H/Q; no25-way cross',
        aggregation='Seed means within case, then equal systems by phase/H/Q/dataset/actualclock; do not reuse old t0=3 scores',
        CPU_wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,GPU_seconds=0,model_calls=0)
    result['analysis']=old.get('analysis',{})
    base.write_json(out/'manifest.json',result)
    jobs=[]
    for phase in ('phase1','phase2'):
        for e in entries:
            checkpoint=result['phase1_checkpoint'] if phase=='phase1' else str(RUN/e['setting']/'coordinate_final.pt')
            jobs.append(dict(arm=phase,phase=phase,setting=e['setting'],entry_id=e['id'],checkpoint=checkpoint,seeds=[17,29],
                             output=str(RUN/'generation'/phase/e['id']),priority=1 if phase=='phase1' else 2))
    base.write_json(out/'genplan.json',dict(schema='hq_ratio_two_phase_finite_plan_v1',jobs=jobs,expected_new_draws=120,
        counts_by_phase={'phase1':60,'phase2':60},budget_seconds=115200,source_epsilon_contract='Pre-draw60,N,3 and take each physical-time prefix; generated extra tail retained without GT',
        order='Phase1 then phase2, fixed case then setting order; no result-based selection'))
    print(json.dumps(dict(status='completed',entries=len(entries),manifest=str(out/'manifest.json'))),flush=True)


MAIN,MAIN_RUN,ENGINE,PARENT_EVAL=base.MAIN,base.MAIN_RUN,base.ENGINE,base.PARENT_EVAL
ENDPOINT_ORIGINS=()
read,write_json,cpu_only,prepare_case,import_file=base.read,base.write_json,base.cpu_only,base.prepare_case,base.import_file

def import_generated(args):
    from molvid.evaluation.common_panel import load_draws
    base.cpu_only();start,cpu=time.monotonic(),time.process_time()
    run=args.run_root.resolve();out=args.output_root.resolve();dest=out/'imported';dest.mkdir(parents=True,exist_ok=False)
    manifest=base.read(run/'eval_inputs/manifest.json');entries={e['id']:e for e in manifest['entries']}
    plan=base.read(run/'eval_inputs/genplan.json')
    jobs=[j for j in plan['jobs'] if j['phase'] in args.phases and entries[j['entry_id']]['base_case'] in args.cases and (not args.settings or j['setting'] in args.settings)]
    groups={}
    for job in jobs:groups.setdefault(job['entry_id'],[]).append(job)
    cases=[];missing=[];errors=[];sampling=[];available=0;expected=0
    for key,selected in groups.items():
        entry=entries[key];prepared=base.prepare_case(entry,run/'eval_inputs');folder=dest/key;folder.mkdir()
        reference=folder/'reference.npz';base.export_npz(reference,prepared.reference,prepared.time_ps,prepared,history_frames=entry['history_frames'])
        case=dict(id=key,entry=entry,reference_npz=str(reference),models=[],status='ok')
        for job in selected:
            phase=job['phase'];model=dict(name=phase,requested_seeds=job['seeds'],draws=[],missing_draws=[],status='NA')
            for seed in job['seeds']:
                expected+=1;source=Path(job['output']);raw=source/f'coordinates_seed{seed}.pt'
                identity=dict(case=key,base_case=entry['base_case'],arm=phase,phase=phase,setting=entry['setting'],seed=seed,raw=str(raw))
                try:
                    p=base.read(source/'protocol.json');receipt=p.get('draws',{}).get(str(seed))
                    if not receipt or seed not in p.get('seeds',[]):raise FileNotFoundError('No completed per-draw receipt')
                    if p.get('arm')!=phase or p.get('id')!=key:raise ValueError('Protocol phase/entry identity differs')
                    if Path(p['checkpoint']).resolve()!=Path(job['checkpoint']).resolve():raise ValueError('Wrong phase checkpoint')
                    if p.get('checkpoint_step')!=(512 if phase=='phase1' else 256):raise ValueError('Wrong scheduled checkpoint slot')
                    for field in ('setting','history_frames','block_size','query_time_ps','history_full64_indices','query_full64_indices'):
                        if p.get(field)!=entry[field]:raise ValueError('Protocol condition/clock mismatch: '+field)
                    info=receipt.get('sampling_info',{});nfe=entry['planned_vector_field_evaluations']
                    if receipt.get('actual_vector_field_evaluations')!=nfe or info.get('total_NFE')!=nfe:raise ValueError('Actual NFE mismatch')
                    if info.get('block_lengths')!=entry['block_lengths']:raise ValueError('Complete generated block lengths differ')
                    draw=load_draws(dict(name=phase,cache=str(raw),seeds=[seed],history_frames=entry['history_frames'],coordinate_unit='angstrom'),prepared,run)[0]
                    if not np.isfinite(draw['coordinates']).all():raise ValueError('Raw nonfinite output retained; no replacement')
                    artifact=source/f'artifacts_seed{seed}.pt';saved=torch.load(artifact,map_location='cpu',weights_only=False)
                    full=np.asarray(saved['full_coordinates_A']);times=np.asarray(saved['full_query_time_ps']).reshape(-1)
                    if full.shape!=(entry['raw_query_frames'],*draw['coordinates'].shape[1:]) or not np.isfinite(full).all() or not np.array_equal(full[:50],draw['coordinates']):raise ValueError('Raw full-block output/prefix mismatch')
                    expected_time=np.asarray(entry['raw_query_time_ps'])
                    if not np.allclose(times,expected_time,atol=1e-5,rtol=0):raise ValueError('Raw full query times must be relative to the last observation')
                    target=folder/phase/f'seed_{seed}';target.mkdir(parents=True)
                    trajectory=target/'trajectory.npz';base.export_npz(trajectory,draw['coordinates'],draw['time_ps'],prepared,history_frames=entry['history_frames'],seed=seed)
                    model['draws'].append(dict(seed=seed,npz=str(trajectory),raw=str(raw),protocol=str(source/'protocol.json'),artifact=str(artifact),sampling=receipt,status='ok'))
                    sampling.append(dict(identity,actual_NFE=nfe,sampling=receipt));available+=1
                except Exception as exc:
                    bad=dict(identity,status='NA',reason=f'{type(exc).__name__}: {exc}');missing.append(bad);model['missing_draws'].append(bad)
                    if not isinstance(exc,(FileNotFoundError,json.JSONDecodeError)):errors.append(bad)
            model.update(status='ok' if model['draws'] else 'NA',actual_draw_count=len(model['draws']),reason=None if model['draws'] else 'No completed valid draw')
            case['models'].append(model)
        cases.append(case)
    details=dict(status='completed' if available==expected and not errors else 'partial',expected_new_draws=expected,available_new_draws=available,
        full_plan_new_draws=120,phases=args.phases,selected_cases=args.cases,selected_settings=args.settings,missing=missing,errors=errors,sampling=sampling,
        reused_parent_draws=0,CPU_wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,GPU_seconds=0,
        reference_scope='Only common t0=11 and true future12:62; no t0=3 score or geometry-band reuse')
    base.write_json(dest/'panel_index.json',dict(cases=cases,errors=errors));base.write_json(dest/'import_details.json',details)
    print(json.dumps({k:details[k] for k in ('status','expected_new_draws','available_new_draws')}),flush=True)
    if not args.allow_partial and (available!=expected or errors):raise RuntimeError('Full selected generated scope incomplete; partial evidence retained')


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=RUN/'eval_inputs')
    p.add_argument('--import-generated',action='store_true');p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path)
    p.add_argument('--phases',nargs='+',choices=('phase1','phase2'),default=['phase1','phase2']);p.add_argument('--cases',nargs='+',choices=base.CASES,default=list(base.CASES))
    p.add_argument('--settings',nargs='+',choices=SETTINGS);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.import_generated:
        a.output_root=a.output_root or a.run_root/'evaluation';import_generated(a)
    else:prepare(a.output)

if __name__=='__main__':cli()
