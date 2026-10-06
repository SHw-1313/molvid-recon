"""Bind the completed512 audit, exercise the real trainer input path on CPU.

Produces INPUT-ONLY recipes, deliberately lacking the completed CUDA gate pins.
Never submits training; no miniature corpus substitution if full512 is pending.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
import numpy as np
from molvid.cli.train_frame_joint import _open_data,_sampler
from molvid.data.batch import collate_clip_records
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file
from molvid.training.data_scale_input import expanded_training_input,read
from molvid.training.execution import execution_schedule


def process(pid):
    try:
        root=Path('/proc')/str(pid)
        return (root.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
                root.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:
        return None


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU audit requires hidden CUDA')
    tracked=process(args.owner_pid)
    state=read(args.qualified/'status.json')
    if state['status']!='completed' and (tracked is None or 'tools.wait_full_data_qualification' not in tracked[1]):
        raise ValueError('identified live full512 qualifier required while waiting')
    root=Path(__file__).resolve().parents[1]
    identity=freeze_run_identity(args.output,inputs=[args.qualified/'run_identity.json',
        args.recipes/'query4_71.json',args.recipes/'query4_72.json'],
        purpose='Real TRAIN512 trainer-input audit without model/CUDA calls',
        protocol=dict(owner_pid=args.owner_pid,process_identity=tracked,model_calls=0,
            training_updates=0,formal_training_submitted=False,protected_payloads_read=0))
    def unchanged():
        for path,digest in identity['input_sha256'].items():
            if sha256_file(path)!=digest:raise ValueError('registered input changed')
        for path,digest in identity['source_files_sha256'].items():
            if sha256_file(root/path)!=digest:raise ValueError('trainer input source changed')
    try:
        started=time.monotonic()
        while state['status']!='completed':
            unchanged()
            if state['status']=='failed':raise RuntimeError('full512 prerequisite failed')
            if process(args.owner_pid)!=tracked:raise RuntimeError('identified qualifier lost or PID reused')
            if time.monotonic()-started>86400:raise TimeoutError('24h prerequisite deadline')
            atomic_write_json(args.output/'status.json',dict(status='waiting',prerequisite=state))
            time.sleep(30);state=read(args.qualified/'status.json')
        if state.get('full512_cpu_qualified') is not True:
            raise ValueError('full512 concat AND calendar qualification required')
        audit=args.qualified/'concat';calendar=args.qualified/'calendar'
        settings=dict(audit=str(audit),calendar=str(calendar))
        for path,key in [(audit/'summary.json','audit_summary_sha256'),
            (audit/'run_identity.json','audit_identity_sha256'),
            (audit/'closed_shards_snapshot.json','snapshot_sha256'),
            (calendar/'summary.json','calendar_summary_sha256'),
            (calendar/'run_identity.json','calendar_identity_sha256')]:
            settings[key]=sha256_file(path)
        reports={}
        for seed in (71,72):
            unchanged()
            recipe=args.recipes/f'query4_{seed}.json';raw=deepcopy(read(recipe))
            raw['coordinate_data_scale']=dict(settings,reference_recipe=str(recipe),reference_recipe_sha256=sha256_file(recipe),
                matched_manifest_sha256=sha256_file(calendar/f'matched_{seed}.json'))
            atomic_write_json(args.output/f'candidate_input_only_{seed}.json',raw)
            base,_,base_hash,_,_,close=_open_data(raw['data'])
            try:
                sampler=_sampler(base,raw['training'],seed=seed,rank=0,world=1)
                reference=execution_schedule(sampler.global_batches,sampler.specs,histories=(4,8),seed=seed,epoch=0)[2048:10240]
                atomic_write_json(args.output/'status.json',dict(status='running',stage='verify_and_load_full512',seed=seed))
                with expanded_training_input(base,base_hash,raw,seed,reference) as result:
                    data,specs,entries,manifest,digest,contract,_=result
                    lookup={s.index:s for s in specs}
                    selected={0,len(entries)-1,max(range(len(entries)),key=lambda j:sum(
                        lookup[i].atoms*(lookup[i].frames-entries[j].history_frames) for i in entries[j].indices))}
                    for key in {(e.view,e.time_bucket_id,e.history_frames) for e in entries}:
                        selected.add(next(j for j,e in enumerate(entries) if (e.view,e.time_bucket_id,e.history_frames)==key))
                    probes=[]
                    for j in sorted(selected):
                        entry=entries[j];records=[data[i] for i in entry.indices]
                        batch=collate_clip_records(records)
                        for i,record in zip(entry.indices,records,strict=True):
                            if (record['split']!='train' or record['sample_id']!=lookup[i].sample_id
                                    or record['x'].shape[:2]!=(16,lookup[i].atoms)
                                    or not np.isfinite(record['x']).all()):
                                raise ValueError('actual scheduled record identity/shape/train/finite gate failed')
                        probes.append(dict(cursor=j,indices=entry.indices,history=entry.history_frames,
                            samples=list(batch.sample_id),atoms=batch.x.shape[1],frames=batch.x.shape[0],
                            time_ps=batch.time_ps.tolist(),frame_mask=batch.frame_mask.tolist()))
                    reports[str(seed)]=dict(schedule_sha256=manifest['sha256'],data_hash=digest,
                        systems=512,records=len(data),selected_updates=len(entries),real_batches_decoded=len(probes),
                        probes=probes,exposure=contract['exposure'])
            finally:
                close()
        unchanged()
        atomic_write_json(args.output/'summary.json',dict(status='completed',seeds=reports,
            model_calls=0,training_updates=0,protected_payloads_read=0,training_qualified=False,
            recipe_status='input_only_missing_CUDA_gate; not submitted',seconds=time.monotonic()-started))
        atomic_write_json(args.output/'status.json',dict(status='completed',training_qualified=False))
    except BaseException as error:
        atomic_write_json(args.output/'status.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('qualified','recipes','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    run(args)
