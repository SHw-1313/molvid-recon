"""Bounded CPU preparation of original first128 TRAIN windows with12+24 true frames."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from train_window36 import read_contiguous_hq_record,DEV_SYSTEMS
from molvid.data.kernel_windows import plan_digest
from molvid.data.store import ClipMMapDataset
from molvid.data.batch import collate_clip_records
SOURCE=Path('/workspace/MOLVID-FLOW-KERNEL-20261027/runs/flow_kernel_20261027/data/data_manifest.json')
CONTACT=Path('/workspace/molvid-misato-contact-filter-v1/molvid/data/contact_filter.py')
_STORES={}
_CONTACT=None
CONTACT_EXCLUSION_REASON='ValueError: One or more real36 reference frames fails existing5A rule; no silent replacement'


def classify_contact_exclusion(entry):
    """Only the proved, finite 36-frame 5A rule can exclude a candidate."""
    if entry.get('status')=='excluded':
        return entry
    distances=entry.get('minimum_contact_A_by_frame',[])
    proven=(entry.get('status')=='NA' and entry.get('reason')==CONTACT_EXCLUSION_REASON
            and entry.get('domain')=='complex' and entry.get('all_atom_fields_equal') is True
            and len(entry.get('native_frame_indices',[]))==36 and len(distances)==36
            and all(np.isfinite(d) for d in distances) and any(d>5. for d in distances))
    if proven:
        entry=dict(entry,original_preparation_status='NA',status='excluded',
                   exclusion_rule='existing5A_all_36_real_frames',
                   violating_window_frames=[i for i,d in enumerate(distances) if d>5.])
    return entry


def prepare_one(row, out, allow_earlier_origin):
    """Independent process owns its raw file and reader/library locks."""
    global _CONTACT
    torch.set_num_threads(1)
    entry=dict(row,joint_ordinal=row['plan_index'],aux_group_history=(4,8)[row['plan_index']%2])
    began,cpu=time.monotonic(),time.process_time()
    try:
        if row['store'] not in _STORES:_STORES[row['store']]=ClipMMapDataset(row['store'])
        original=_STORES[row['store']][row['store_index']]
        entry['load_record_seconds']=time.monotonic()-began
        if original['sample_id']!=row['sample_id'] or original['topology_fingerprint']!=row['topology_fingerprint']:raise ValueError('Original fixed TRAIN identity changed')
        record,evidence=read_contiguous_hq_record(original,row['source_topology'],row['source_trajectory'],allow_earlier_origin=allow_earlier_origin)
        entry.update(evidence)
        contact_start=time.monotonic()
        if row['domain']=='complex':
            if _CONTACT is None:
                spec=importlib.util.spec_from_file_location('memory_existing_5A',CONTACT);_CONTACT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_CONTACT)
            d=_CONTACT.minimum_ligand_protein_distance(record);entry.update(minimum_contact_A_by_frame=d.tolist(),maximum_frame_minimum_contact_A=float(d.max()))
            if np.any(d>5.):raise ValueError('One or more real36 reference frames fails existing5A rule; no silent replacement')
        entry['contact_seconds']=time.monotonic()-contact_start
        shapes=[list(collate_clip_records([record]).x.shape)]
        if shapes[0]!=[36,row['atom_count'],3]:raise ValueError('Wrong true36 native record')
        path=Path(out)/f"window_{row['plan_index']:03d}.pt"
        torch.save(dict(record=record),path)
        entry.update(status='ok',record=str(path),record_path=str(path),record_shape=shapes[0],history_indices=list(range(12)),target_indices=list(range(12,36)))
    except Exception as exc:
        entry.update(status='NA',reason=f'{type(exc).__name__}: {exc}')
    entry.update(preparation_wall_seconds=time.monotonic()-began,preparation_cpu_seconds=time.process_time()-cpu)
    return classify_contact_exclusion(entry)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-manifest',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=ROOT/'runs/flow_hq_ratio_20261102/data')
    p.add_argument('--all',action='store_true');p.add_argument('--allow-earlier-origin',action='store_true')
    p.add_argument('--workers',type=int,choices=(1,2),default=1)
    p.add_argument('--resume',action='store_true',help='Keep all completed rows from this same plan; continue only unseen indices')
    a=p.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use existing torch-ito, CUDA hidden')
    torch.set_num_threads(1);start,cpu=time.monotonic(),time.process_time()
    source=json.loads(a.source_manifest.read_text())
    if source['status']!='complete':raise ValueError('Existing TRAIN plan incomplete')
    candidates=source['rows'][:128]
    if [r['plan_index'] for r in candidates]!=list(range(128)):raise ValueError('Need unchanged original first128 joint plan')
    selected=candidates if a.all else sorted([r for domain in ('protein','complex') for r in [x for x in candidates if x['domain']==domain][:2]],key=lambda x:x['plan_index'])
    if any(r['system'] in DEV_SYSTEMS or r['split']!='train' for r in candidates):raise ValueError('TRAIN/DEV isolation violation in fixed source plan')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Preserve original completed preparation receipt')
    report=dict(schema='molvid.hq_true_contiguous36.v1',status='preparing',source_plan=str(a.source_manifest),
                source_plan_digest=source['plan_digest'],seed=source['seed'],data_config=source['data_config'],
                source_candidates=len(candidates),candidate_count=len(selected),source_domain_counts=dict(Counter(r['domain'] for r in candidates)),
                source_max_atoms=max(r['atom_count'] for r in candidates),selected_plan_indices=[r['plan_index'] for r in selected],
                history_frames_max=12,common_query_frames=24,total_frames=36,io_workers=a.workers,io_executor='independent_processes_spawn',
                origin_rule='min(original origin, last native frame -35*stride)' if a.allow_earlier_origin else 'original cached origin, insufficient length explicit NA',
                contact_rule='All actual36 MISATO reference frames <=5A, existing heavy-atom filter; no silent replacement',
                training_history='Initial12 past frames and common24 targets. Each TRAIN H/Q chunk only sees preceding true history; inference uses generated history. Excluded joint plan index also skips its paired Q1 slot without compressing the original256-slot LR schedule.',
                alignment='Same raw replica, shared first-frame coordinate convention across all36; no clip concatenation or independently fitted subblocks',
                rows=[],exclusions=[],failures=[],model_calls=0,GPU_seconds=0)
    if a.resume:
        old=json.loads((out/'preparation_partial.json').read_text())
        for key in ('source_plan_digest','selected_plan_indices','origin_rule','seed','data_config'):
            if old[key]!=report[key]:raise ValueError('Cannot resume a different preparation plan: '+key)
        old_nonvalid=[classify_contact_exclusion(r) for r in old.get('failures',[])+old.get('exclusions',[])]
        report.update(rows=old['rows'],exclusions=[r for r in old_nonvalid if r['status']=='excluded'],
                      failures=[r for r in old_nonvalid if r['status']!='excluded'])
        for row in report['rows']:
            if not Path(row['record']).is_file():raise FileNotFoundError('A completed row lost its prepared record')
        report['preserved_rows_at_resume']=len(report['rows'])
        report['prior_execution_evidence']=[str(out/'preparation_threaded_snapshot.json'),str(out/'preparation_before_contact_exclusion.json')]
    elif (out/'preparation_partial.json').exists():raise FileExistsError('Use explicit same-plan resume')
    rows,exclusions,failures=report['rows'],report['exclusions'],report['failures'];seen={r['plan_index'] for r in rows+exclusions+failures}
    pending=[r for r in selected if r['plan_index'] not in seen]
    def save(name='preparation_partial.json'):
        report['rows']=sorted(rows,key=lambda r:r['plan_index']);report['failures']=sorted(failures,key=lambda r:r['plan_index'])
        report['exclusions']=sorted(exclusions,key=lambda r:r['plan_index'])
        destination=out/name;temporary=out/(name+'.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(destination)
    save()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare_one,row,str(out),a.allow_earlier_origin) for row in pending]):
            entry=classify_contact_exclusion(future.result());(rows if entry['status']=='ok' else exclusions if entry['status']=='excluded' else failures).append(entry)
            save();print(json.dumps(dict(plan_index=entry['plan_index'],system=entry['system'],available=len(rows),excluded=len(exclusions),failed=len(failures),row_seconds=entry['preparation_wall_seconds'],execution_seconds=time.monotonic()-start)),flush=True)
    rows.sort(key=lambda r:r['plan_index']);failures.sort(key=lambda r:r['plan_index'])
    scanned_indices={r['plan_index'] for r in rows+exclusions+failures}
    scanned_all=scanned_indices=={r['plan_index'] for r in selected}
    report.update(status='complete' if scanned_all and not failures else 'partial',plan_digest=plan_digest(rows),
                  full_fixed_plan_prepared=bool(a.all and scanned_all and not failures),
                  full_candidate_plan_scanned=bool(a.all and scanned_all),all_candidates_processed=bool(a.all and scanned_all),scanned_candidate_count=len(scanned_indices),
                  exclusion_count=len(exclusions),valid_joint_windows=len(rows),training_updates=2*len(rows),target_frames=25*len(rows),planned_schedule_slots=256,
                  valid_pool_rule='Original first128 predetermined candidates scanned once; only proved5A exclusions removed, paired joint/Q1 slots both skipped, no replacement/repetition or LR compression.',
                  requested_windows=len(selected),available_windows=len(rows),available_domain_counts=dict(Counter(r['domain'] for r in rows)),
                  max_prepared_atoms=max([r['atom_count'] for r in rows]+[0]),
                  CPU_wall_seconds_this_execution=time.monotonic()-start,CPU_parent_seconds_this_execution=time.process_time()-cpu,
                  CPU_worker_seconds_this_execution=sum(r.get('preparation_cpu_seconds',0) for r in rows+exclusions+failures if r['plan_index'] not in seen))
    save('manifest.json');print(json.dumps({k:report[k] for k in ('status','requested_windows','available_windows','max_prepared_atoms','CPU_wall_seconds_this_execution')}),flush=True)


if __name__=='__main__':main()
