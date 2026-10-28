"""Prepare existing six DEV references for independent Q1 endpoints/real rollouts.

No model calls, data search, interpolation, independent block alignment, or scores.
Prepared inputs are capabilities, not authorization to generate every combination.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flow_kernel_20261027'
MAIN = Path('/workspace/MOLVID-MAIN-20261016')
MAIN_RUN = MAIN / 'runs/main_integration_20261016'
SOURCE_MANIFEST = Path('/workspace/JEPA-ANYTHING-MD-20261023/runs/jepa_md_20261023/eval_inputs/manifest.json')
RECONSTRUCT = Path('/workspace/MOLVID-FLOW-TIMEBASIS-20261023/tasks/flow-timebasis-20261023/prepare_inputs.py')
TEMPORAL = Path('/workspace/molvid_temporal_eval_v1')
sys.path[:0] = [str(MAIN), str(TEMPORAL)]
from molvid.evaluation.common_panel import prepare_case, export_npz, write_json
from temporal_eval.metrics import fixed_pairs

CASES = ('atlas_1whz_A', 'atlas_3vnb_A', 'atlas_6e5y_A', '4J8T', '2XSB', '5TV1')
ENDPOINT_ORIGINS = (12, 28, 44)
LAGS = (1, 2, 4)
ENDPOINT_SEEDS = (17, 29, 43, 59)
ROLLOUT_SEEDS = (17, 29)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sliced_record(full, history, query, entry_id):
    indices = np.asarray([*history, *query], dtype=np.int64)
    record = copy.deepcopy(full)
    for key in ('x', 'bpos', 'frame_mask', 'time_ps'):
        record[key] = np.asarray(full[key])[indices].copy()
    # Relative input time origin is first actual observation, not a new alignment.
    record['time_ps'] -= record['time_ps'][0]
    record['delta_time_ps'] = np.diff(record['time_ps'])
    record['native_frame_indices'] = np.asarray(full['native_frame_indices'])[indices].tolist()
    for key in ('rotation', 'translation'):
        record['alignment'][key] = np.asarray(full['alignment'][key])[indices].tolist()
    dt = float(np.diff(np.asarray(full['time_ps']))[0])
    diffs = np.diff(record['native_frame_indices'])
    uniform = bool(np.all(diffs == diffs[0]))
    query_dt = float((query[0] - history[-1]) * dt)
    record.update(sample_id=entry_id, history_frames=len(history), future_frames=len(query),
                  source_stride=int(diffs[0]) if uniform else None,
                  sampled_delta_time_ps=float(record['delta_time_ps'][0]) if uniform else None,
                  history_delta_time_ps=dt if len(history) > 1 else None,
                  query_delta_time_ps=query_dt, time_bucket_id=f'query_{query_dt:g}ps',
                  anchor=int(np.asarray(full['native_frame_indices'])[history[-1]]),
                  lag_ps=query_dt, selected_full64_indices=indices.tolist(),
                  timestamp_provenance='Exact original DEV64 indices, no fit/interpolation or replica concatenation')
    return record


def representation(full):
    ids = np.asarray(full['atom_identity'], str)
    blocks = np.asarray(full['block_id'])
    component = np.asarray(full['component_id'])
    names = np.asarray([x.rsplit(':', 2)[-2].strip().upper() for x in ids])
    ca = []
    for block in np.unique(blocks):
        found = np.flatnonzero((blocks == block) & (component == 0) & (names == 'CA'))
        if len(found) > 1:
            raise ValueError('Ambiguous CA identity')
        if len(found) == 1:
            ca.append(int(found[0]))
    align = np.flatnonzero(np.asarray(full['align_mask'], bool))
    if len(ca) < 2 or len(align) < 3 or np.any(component[align] != 0):
        raise ValueError('Insufficient trusted protein CA/alignment atoms')
    pairs = np.stack(fixed_pairs(len(ca), 4096), axis=1)
    return dict(align_indices=align.tolist(), ca_indices=ca, ca_atom_identity=ids[ca].tolist(),
                internal_pairs_ca_indices=pairs.tolist(), ca_count=len(ca), pair_count=len(pairs),
                alignment='Each complete endpoint/frame to the same true last observation with original record.align_mask; no pairwise/block-specific alignment',
                ca_representation='aligned CA coordinates / sqrt(N_CA), endpoint vector norm in Angstrom',
                internal_representation='fixed CA-pair distances / sqrt(P), vector norm in Angstrom',
                energy_score='mean_i ||phi(X_i)-phi(y)|| - sum_i!=j ||phi(X_i)-phi(X_j)|| / (2*m*(m-1))',
                energy_score_note='Unbiased iid-draw estimate of population energy score; nonself draw pairs; one heldout MD endpoint, not known conditional distribution. Finite estimates can be negative.',
                ca_selection='Original portable exporter: sorted block_id, component_id=0, one CA per residue')


def prepare(output):
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use enter-container + torch-ito with CUDA hidden')
    torch.set_num_threads(2)
    start, cpu = time.monotonic(), time.process_time()
    output = output.resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError('Preserve the existing input manifest')
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_MANIFEST.read_text())
    sources = {(e['case'], e['initid']): e for e in source['entries']}
    reconstruct = import_file('kernel_verified_true64', RECONSTRUCT).reconstruct
    contact = import_file('kernel_original_contact_filter', '/workspace/molvid-misato-contact-filter-v1/molvid/data/contact_filter.py')
    old_path = MAIN_RUN / 'short50_generated_20261017/manifest_ready.json'
    old = json.loads(old_path.read_text())
    templates = {e['id']: e for e in old['cases']}
    cases, endpoints, rollouts, standard, checks = [], [], [], [], []
    for case in CASES:
        full, check = reconstruct(case, sources)
        full["history_frames"] = 4
        full["future_frames"] = 60
        dt = check['base_dt_ps']
        distances = None if case.startswith('atlas_') else contact.minimum_ligand_protein_distance(full)
        if distances is not None and (not np.isfinite(distances).all() or np.any(distances > 5)):
            raise ValueError(f'{case}: original every-frame 5A reference selection failed')
        folder = output / case
        folder.mkdir(exist_ok=True)
        full_path = folder / 'reference64.pt'
        torch.save({'record': full}, full_path)
        rep = representation(full)
        write_json(folder / 'representation.json', rep)
        check.update(atom_count=len(full['atype']), contact_max_A=None if distances is None else float(distances.max()),
                     contact_min_heavy_A_by_frame=None if distances is None else distances.tolist(),
                     reference64=str(full_path), representation=str(folder / 'representation.json'))
        checks.append(check)
        template = {k: copy.deepcopy(v) for k, v in templates[case].items() if k != 'models'}
        if isinstance(template.get('topology_annotations'), str):
            ann = Path(template['topology_annotations'])
            template['topology_annotations'] = str(ann if ann.is_absolute() else (old_path.parent / ann).resolve())

        def add(kind, history, query, label, target, seeds, **metadata):
            entry_id = f'{case}__{label}'
            sub = folder / label
            sub.mkdir(exist_ok=True)
            record = sliced_record(full, history, query, entry_id)
            record_path = sub / 'record.pt'
            torch.save({'record': record}, record_path)
            item = dict(template, id=entry_id, sample_id=entry_id, base_case=case, case=case,
                        dataset='ATLAS' if case.startswith('atlas_') else 'MISATO',
                        family=kind, protocol=label, record=str(record_path), record_path=str(record_path),
                        history_frames=len(history), query_frames=len(query), models=[], seeds=list(seeds),
                        source_reference64=str(full_path), source_manifest=str(SOURCE_MANIFEST),
                        history_full64_indices=list(history), query_full64_indices=list(query),
                        last_observed_full64_index=int(history[-1]), base_dt_ps=dt,
                        dt_ps=float((query[0] - history[-1]) * dt),
                        observed_time_ps=((np.asarray(history) - history[-1]) * dt).tolist(),
                        query_time_ps=((np.asarray(query) - history[-1]) * dt).tolist(),
                        source_time_ps=np.asarray(full["time_ps"])[list(history) + list(query)].tolist(),
                        absolute_native_time_ps=(np.asarray(full["native_frame_indices"])[list(history) + list(query)] * float(full["native_delta_time_ps"])).tolist(),
                        native_frame_indices=np.asarray(full['native_frame_indices'])[list(history) + list(query)].tolist(),
                        reference_npz=str(sub / 'reference.npz'), condition_npz=str(sub / 'condition.npz'),
                        representation=str(folder / 'representation.json'),
                        geometry_calibration_record=templates[case]['record'], geometry_calibration_history_frames=4,
                        geometry_calibration_note='Original fixed same-case MD50 normal bands; never recalibrate from Q1 or generated samples',
                        actual_generation_requested=False, **metadata)
            prepared = prepare_case(item, output)
            export_npz(sub / 'reference.npz', prepared.reference, prepared.time_ps, prepared, history_frames=len(history))
            np.savez_compressed(sub / 'condition.npz', observed=prepared.observed.astype(np.float32),
                                observed_time_ps=np.asarray(item['observed_time_ps']), query_time_ps=prepared.time_ps,
                                atom_identity=prepared.topology['atom_identity'], coordinate_unit='angstrom',
                                sample_id=entry_id, history_frames=len(history), query_frames=len(query))
            target.append(item)
            return item

        for origin in ENDPOINT_ORIGINS:
            for h in (1, 4):
                history = list(range(origin-h+1, origin+1))
                for lag in LAGS:
                    add('independent_endpoint', history, [origin+lag], f'endpoint_t{origin}_h{h}_lag{lag}',
                        endpoints, ENDPOINT_SEEDS, lag_multiplier=lag, history_policy='fixed dense true history; H1 retains only final point',
                        trajectory_statistics='NA: independent single endpoints cannot form a trajectory')
        for h in (1, 4):
            for lag in LAGS:
                n = 50 // lag
                add('free_q1_rollout', list(range(4-h, 4)), [3+lag*i for i in range(1, n+1)],
                    f'rollout_h{h}_lag{lag}', rollouts, ROLLOUT_SEEDS, lag_multiplier=lag,
                    requested_horizon_base_steps=50, covered_horizon_base_steps=n*lag,
                    uncovered_tail_base_steps=50-n*lag, history_policy='latest H actual frames; initial true history only, then generated history with true timestamps',
                    total_Q1_model_calls=n, planned_vector_field_evaluations=35*n)
        for mode in ('joint', 'q12', 'q4', 'coarse_fine'):
            item = add('standard_q50', [0, 1, 2, 3], list(range(4, 54)), mode, standard, ROLLOUT_SEEDS,
                       mode=mode, history_policy='original H4', raw_query_frames=60 if mode == 'q12' else 50,
                       coarse_fine_definition='historical joint10-anchor + pure clamp Q50; NOT scheme E' if mode == 'coarse_fine' else None)
            if mode == 'q12':
                item['raw60_reference_record'] = str(full_path)
                item['raw60_query_full64_indices'] = list(range(4, 64))
        print(json.dumps({'case': case, 'real_frames': 64, 'atoms': len(full['atype']), 'base_dt_ps': dt,
                          'contact_max_A': check['contact_max_A'], 'CA': rep['ca_count'], 'pairs': rep['pair_count']}), flush=True)
    result = dict(schema='lag_history_kernel_eval_inputs_v1', status='completed', errors=[],
                  source_manifest=str(SOURCE_MANIFEST), source_reconstruction_checks=checks,
                  cases=list(CASES), endpoint_entries=endpoints, rollout_entries=rollouts, standard_entries=standard,
                  prepared_counts=dict(endpoint_inputs=len(endpoints), rollout_inputs=len(rollouts), standard_inputs=len(standard)),
                  endpoint_origins=list(ENDPOINT_ORIGINS), endpoint_seeds=list(ENDPOINT_SEEDS), rollout_seeds=list(ROLLOUT_SEEDS),
                  generation_matrix_status='await engineer frozen arm/protocol matrix; preparation is not a generation request',
                  units=dict(coordinates='angstrom', time='ps'), analysis=old.get('analysis', {}),
                  aggregation='Endpoint draws within case/origin/history/lag, origins within case, then equal cases within dataset/actual clock/protocol. Never pool independent endpoints into trajectories.',
                  comparison_scope='Separate endpoint distribution, free recurrence, and direct/composed/joint endpoint compatibility; finite-history diagnostic, no mandatory coordinate-marginal CK.',
                  primary_endpoint_minimum_draws=4, missing_policy='Explicit NA with actual n; no duplicate/padded draws',
                  wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, GPU_seconds=0)
    write_json(output / 'manifest.json', result)
    print(json.dumps({'manifest': str(output / 'manifest.json'), 'counts': result['prepared_counts'], 'errors': [],
                      'wall_seconds': result['wall_seconds'], 'cpu_seconds': result['cpu_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'eval_inputs')
    args = parser.parse_args()
    prepare(args.output)



ARMS = ('control', 'h1', 'history_mix', 'dense_history', 'lag1')
ENGINE = Path('/workspace/complexmd_eval_logicmd55_misato89_v0.3.1_20261014/engine')
PARENT_EVAL = Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/evaluation')


def read(path):
    return json.loads(Path(path).read_text())


def cpu_only():
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use enter-container + torch-ito with CUDA hidden')
    torch.set_num_threads(2)


def add_compatibility(output):
    """Append only eight explicitly authorized same-origin inputs; retain old PTs."""
    cpu_only()
    path = output / 'manifest.json'
    manifest = read(path)
    if not manifest.get('compatibility_entries'):
        entries = []
        for case in ('atlas_6e5y_A', '4J8T'):
            raw = torch.load(output / case / 'reference64.pt', map_location='cpu', weights_only=False)
            full = raw['record']
            for h in (1, 4):
                template = next(e for e in manifest['endpoint_entries'] if e['base_case'] == case and e['history_frames'] == h and e['last_observed_full64_index'] == 12 and e['lag_multiplier'] == 1)
                for mode, count in [('compat_joint', 12), ('compat_steps', 4)]:
                    label = f'{mode}_t12_h{h}'
                    identity = case + '__' + label
                    folder = output / case / label
                    folder.mkdir(exist_ok=False)
                    history = list(range(13-h, 13)); query = list(range(13, 13+count))
                    record = sliced_record(full, history, query, identity)
                    record_path = folder / 'record.pt'
                    torch.save({'record': record}, record_path)
                    item = copy.deepcopy(template)
                    item.update(id=identity, sample_id=identity, family='compatibility', protocol=label, mode=mode,
                                record=str(record_path), record_path=str(record_path), query_frames=count,
                                query_full64_indices=query, query_time_ps=[template['base_dt_ps']*i for i in range(1, count+1)],
                                source_time_ps=np.asarray(full['time_ps'])[history+query].tolist(),
                                absolute_native_time_ps=(np.asarray(full['native_frame_indices'])[history+query]*float(full['native_delta_time_ps'])).tolist(),
                                native_frame_indices=np.asarray(full['native_frame_indices'])[history+query].tolist(),
                                reference_npz=str(folder/'reference.npz'), condition_npz=str(folder/'condition.npz'),
                                trajectory_statistics='Actual same-origin joint trajectory or actual four-step generated-history chain; not independent endpoints',
                                comparison_endpoint_offsets=[2,4], initial_origin_identical_to_endpoint=True)
                    prepared = prepare_case(item, output)
                    export_npz(folder/'reference.npz', prepared.reference, prepared.time_ps, prepared, history_frames=h)
                    np.savez_compressed(folder/'condition.npz', observed=prepared.observed.astype(np.float32),
                                        observed_time_ps=np.asarray(item['observed_time_ps']), query_time_ps=prepared.time_ps,
                                        atom_identity=prepared.topology['atom_identity'], coordinate_unit='angstrom', sample_id=identity,
                                        history_frames=h, query_frames=count)
                    entries.append(item)
        manifest['compatibility_entries'] = entries
        manifest['prepared_counts']['compatibility_inputs'] = len(entries)
        write_json(path, manifest)
    return manifest


def generation_plan(run):
    manifest = add_compatibility(run/'eval_inputs')
    jobs = []
    def add(arm, entry, priority, seeds=None, reason=''):
        checkpoint = run/'control/checkpoint_step256.pt' if arm == 'control256' else run/arm/'coordinate_final.pt'
        jobs.append(dict(arm=arm, entry_id=entry['id'], checkpoint=str(checkpoint),
                         checkpoint_step=256 if arm == 'control256' else 512,
                         seeds=list(seeds or entry['seeds']), priority=priority, reason=reason,
                         output=str(run/'generation'/arm/entry['id'])))
    for arm in ARMS:
        for e in manifest['endpoint_entries']:
            add(arm,e,1 if e['last_observed_full64_index']==12 else 5,reason='same-origin independent endpoint distribution')
        for e in manifest['standard_entries']:
            add(arm,e,2 if e['mode']=='joint' else 7,reason='unchanged idx3 standard protocol')
        for e in manifest['rollout_entries']:
            native = 1 if arm=='h1' else 4
            if e['history_frames']==native:
                add(arm,e,3 if e['lag_multiplier']==1 else 6,reason='native-history free recurrence')
            elif arm in ('h1','history_mix') and e['base_case'] in ('atlas_6e5y_A','4J8T') and e['lag_multiplier']==1:
                add(arm,e,4,reason='preselected cross-history diagnostic')
        for e in manifest['compatibility_entries']:
            add(arm,e,4,reason='same-origin joint / four-step vs direct endpoint distribution')
    for e in manifest['standard_entries']:
        if e['mode']=='joint': add('control256',e,2,reason='256-update control dose; parent_step4096 is separate')
    jobs.sort(key=lambda x:(x['priority'],ARMS.index(x['arm']) if x['arm'] in ARMS else len(ARMS),x['entry_id']))
    keys=[(j['arm'],j['entry_id']) for j in jobs]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate generation conditions')
    result=dict(schema='kernel_finite_generation_plan_v1',jobs=jobs,expected_new_draws=sum(len(j['seeds']) for j in jobs),
                counts_by_priority={str(k):sum(len(j['seeds']) for j in jobs if j['priority']==k) for k in range(1,8)},
                parent_standard_reused_draws=48,parent_evaluation=str(PARENT_EVAL),
                phase_steps='New optimizer phase0..512; control256 step256; shared initialization parent_step4096',
                budget_policy='Fixed ordered plan, generation budget controlled solely by engineer generation_execution; partial preserved, no score-based selection')
    write_json(run/'eval_inputs/genplan.json',result)
    print(json.dumps(dict(genplan=str(run/'eval_inputs/genplan.json'),jobs=len(jobs),new_draws=result['expected_new_draws'],priorities=result['counts_by_priority'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN/'eval_inputs')
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--extend-plan',action='store_true')
    args=parser.parse_args()
    if args.extend_plan:generation_plan(args.run_root.resolve())
    else:prepare(args.output)




def all_entries(manifest):
    return {e['id']:e for key in ('endpoint_entries','rollout_entries','standard_entries','compatibility_entries') for e in manifest.get(key,[])}


def requested_nfe(entry):
    if entry['family']=='independent_endpoint':return 35
    if entry['family']=='free_q1_rollout' or entry.get('mode')=='compat_steps':return 35*entry['query_frames']
    return dict(joint=35,q12=175,q4=455,coarse_fine=70,compat_joint=35)[entry['mode']]


def import_generated(args):
    from molvid.evaluation.common_panel import load_draws
    cpu_only();start=time.monotonic();cpu=time.process_time()
    run=args.run_root.resolve();out=args.output_root.resolve();dest=out/'imported'
    dest.mkdir(parents=True,exist_ok=False)
    inputs=read(run/'eval_inputs/manifest.json');entries=all_entries(inputs);plan=read(run/'eval_inputs/genplan.json')
    jobs=[j for j in plan['jobs'] if (not args.arms or j['arm'] in args.arms) and entries[j['entry_id']]['base_case'] in args.cases
          and (not args.priorities or j['priority'] in args.priorities)]
    groups={}
    for j in jobs:groups.setdefault(j['entry_id'],[]).append(j)
    endpoint_panel=[];trajectory_panel=[];missing=[];errors=[];sampling=[];available=0;expected=0
    parent_index={c['id']:c for c in read(PARENT_EVAL/'comparison/viewer_panel_index.json')['cases']}
    parent_count=0
    for entry_id,selected in groups.items():
        entry=entries[entry_id];prepared=prepare_case(entry,run/'eval_inputs')
        folder=dest/entry_id;folder.mkdir()
        refpath=folder/'reference.npz';export_npz(refpath,prepared.reference,prepared.time_ps,prepared,history_frames=entry['history_frames'])
        case=dict(id=entry_id,entry=entry,reference_npz=str(refpath),models=[],status='ok')
        for job in selected:
            arm=job['arm'];model=dict(name=arm,requested_seeds=job['seeds'],draws=[],missing_draws=[],status='NA')
            for seed in job['seeds']:
                expected+=1;source=run/'generation'/arm/entry_id;raw=source/f'coordinates_seed{seed}.pt'
                identity=dict(case=entry_id,base_case=entry['base_case'],arm=arm,seed=seed,raw=str(raw))
                try:
                    p=read(source/'protocol.json');receipt=p.get('draws',{}).get(str(seed))
                    if not receipt or seed not in p.get('seeds',[]):raise FileNotFoundError('No completed per-draw protocol receipt')
                    if p.get('arm')!=arm or p.get('id')!=entry_id:raise ValueError('Protocol arm/input identity differs')
                    if p.get('checkpoint_step')!=job['checkpoint_step'] or Path(p['checkpoint']).resolve()!=Path(job['checkpoint']).resolve():raise ValueError('Checkpoint identity/phase step differs')
                    for key in ('history_frames','query_frames','query_time_ps','history_full64_indices','query_full64_indices'):
                        if p.get(key)!=entry[key]:raise ValueError('Protocol source/clock mismatch: '+key)
                    nfe=requested_nfe(entry)
                    if receipt.get('actual_vector_field_evaluations')!=nfe or receipt.get('sampling_info',{}).get('total_NFE')!=nfe:
                        raise ValueError('Actual field count differs from approved protocol')
                    # Kernel protocols use a top-level string protocol label; the
                    # common loader's optional protocol wrapper expects a dict.
                    # Identity, H/Q, clock, checkpoint and NFE are checked above;
                    # retain the original receipt path in the exported draw.
                    draw=load_draws(dict(name=arm,cache=str(raw),seeds=[seed],history_frames=entry['history_frames'],coordinate_unit='angstrom'),prepared,run)[0]
                    if not np.isfinite(draw['coordinates']).all():raise ValueError('Raw nonfinite output retained as failed; not padded or repaired')
                    artifact=None
                    if entry.get('mode') in ('q12','coarse_fine'):
                        artifact=source/f'artifacts_seed{seed}.pt'
                        data=torch.load(artifact,map_location='cpu',weights_only=False)
                        if entry['mode']=='q12':
                            full=np.asarray(data['full_coordinates_A']);times=np.asarray(data['full_query_time_ps']).reshape(-1)
                            if full.shape!=(60,*draw['coordinates'].shape[1:]) or not np.isfinite(full).all() or not np.array_equal(full[:50],draw['coordinates']):raise ValueError('Raw60 prefix/finiteness differs')
                            if not np.allclose(times,np.arange(1,61)*entry['base_dt_ps']+prepared.record['time_ps'][entry['history_frames']-1],atol=1e-5,rtol=0):raise ValueError('Raw60 clock differs')
                        else:
                            ix=np.asarray(data['anchor_indices']).reshape(-1)
                            if not np.array_equal(ix,np.arange(4,50,5)) or not np.array_equal(np.asarray(data['anchor_coordinates_A']),draw['coordinates'][ix]):raise ValueError('Original CF anchors not preserved')
                    target=folder/arm/f'seed_{seed}';target.mkdir(parents=True)
                    npz=target/'trajectory.npz';export_npz(npz,draw['coordinates'],draw['time_ps'],prepared,history_frames=entry['history_frames'],seed=seed)
                    model['draws'].append(dict(seed=seed,npz=str(npz),raw=str(raw),protocol=str(source/'protocol.json'),status='ok',
                                              artifact=None if artifact is None else str(artifact),sampling=receipt))
                    sampling.append(dict(identity,actual_NFE=nfe,details=receipt));available+=1
                except Exception as exc:
                    failed=dict(identity,status='NA',reason=f'{type(exc).__name__}: {exc}')
                    missing.append(failed);model['missing_draws'].append(failed)
                    if not isinstance(exc,(FileNotFoundError,json.JSONDecodeError)):errors.append(failed)
            model.update(status='ok' if model['draws'] else 'NA',actual_draw_count=len(model['draws']),reason=None if model['draws'] else 'No completed valid draw')
            case['models'].append(model)
        if entry['family']=='standard_q50':
            previous=parent_index[entry['base_case']]
            with np.load(previous['reference_npz'],allow_pickle=False) as data:
                if not np.array_equal(data['positions'],prepared.reference) or not np.array_equal(data['observed'],prepared.observed):raise ValueError('Parent reuse requires exact original idx3 coordinates')
                if not np.array_equal(data['atom_identity'],prepared.topology['atom_identity']) or not np.array_equal(data['time_ps'],prepared.time_ps):raise ValueError('Parent topology/clock mismatch')
            source_name='ligand_'+entry['mode']
            old=next(m for m in previous['models'] if m['name']==source_name)
            reused=dict(name='parent_L4096',draws=[],status='ok',requested_seeds=[17,29],reused_from=str(PARENT_EVAL))
            for d in old['draws']:
                target=folder/'parent_L4096'/f'seed_{d["seed"]}';target.mkdir(parents=True)
                with np.load(d['npz'],allow_pickle=False) as data:xyz=data['positions']
                npz=target/'trajectory.npz';export_npz(npz,xyz,prepared.time_ps,prepared,history_frames=4,seed=d['seed'])
                reused['draws'].append(dict(d,npz=str(npz),source_npz=d['npz'],source_arm=source_name,reused_from=str(PARENT_EVAL)))
                parent_count+=1
            case['reference_local_geometry_json']=previous['reference_local_geometry_json']
            case['models'].append(reused)
        (endpoint_panel if entry['family']=='independent_endpoint' else trajectory_panel).append(case)
    details=dict(status='completed' if available==expected and not errors else 'partial',expected_new_draws=expected,available_new_draws=available,
                 reused_parent_draws=parent_count,full_plan_new_draws=plan['expected_new_draws'],selected_arms=args.arms,
                 selected_cases=args.cases,selected_priorities=args.priorities,missing=missing,errors=errors,sampling=sampling,
                 wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,GPU_seconds=0)
    write_json(dest/'endpoint_panel.json',dict(cases=endpoint_panel,errors=[]))
    write_json(dest/'trajectory_panel.json',dict(cases=trajectory_panel,errors=[]))
    write_json(dest/'import_details.json',details)
    print(json.dumps({k:details[k] for k in ('status','expected_new_draws','available_new_draws','reused_parent_draws')}),flush=True)
    if details['status']!='completed' and not args.allow_partial:raise ValueError('Partial import retained; use --allow-partial')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN/'eval_inputs')
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--extend-plan',action='store_true')
    parser.add_argument('--import-generated',action='store_true')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--arms',nargs='+',choices=[*ARMS,'control256'])
    parser.add_argument('--cases',nargs='+',default=list(CASES),choices=CASES)
    parser.add_argument('--priorities',nargs='+',type=int,choices=range(1,8))
    args=parser.parse_args()
    if args.import_generated:
        args.output_root=args.output_root or args.run_root/'evaluation'
        import_generated(args)
    elif args.extend_plan:generation_plan(args.run_root.resolve())
    else:prepare(args.output)


if __name__ == '__main__':
    main()
