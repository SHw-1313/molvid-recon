"""Finite paired static restoration: native Q1 source sampling and separate teacher diagnostic."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from prepare_static_probe import ROOT, MAIN, MAIN_RUN, RUN, STAGES, RHO, CORR, standalone


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def generate(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import molvid
    from molvid.data.batch import collate_clip_records
    from molvid.flow.coordinate_state import CoordinateStateDenoiser
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or not torch.cuda.is_available():
        raise RuntimeError('Authorized CUDA through existing torch-ito required; no CPU model fallback')
    if Path(molvid.__file__).resolve().parent != ROOT/'molvid':
        raise RuntimeError('Wrong worktree model import')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(args.allocator_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory,1.),0)
    manifest=read(args.manifest); stage=manifest['stages'][args.stage]
    if abs(manifest['corruption_sigma_A']-CORR)>1e-12 or abs(manifest['source_rho_A']-RHO)>1e-12:
        raise ValueError('This study separates 0.2 observation corruption from fixed source rho')
    selected=[r for r in manifest['snapshots'] if not args.snapshots or r['id'] in args.snapshots]
    if not selected or (args.snapshots and len(selected)!=len(set(args.snapshots))):
        raise ValueError('Unknown or duplicate requested snapshot')
    output=(args.output or RUN/'static_probe/generated'/args.stage).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    protocol=dict(schema='molvid.static_ref.probe.generation.v1', status='started', stage=args.stage,
        checkpoint=stage['checkpoint'], requested_snapshots=[r['id'] for r in selected], snapshots={},
        manifest=str(args.manifest.resolve()), coordinate_unit='angstrom', task='static',
        corruption_seed=manifest['corruption_seed'], source_seed=manifest['source_seed'],
        corruption_sigma_A=manifest['corruption_sigma_A'], source_rho_A=manifest['source_rho_A'],
        query_frames=1, sampler_intervals=18, full_NFE=35, teacher_NFE=1, teacher_progress=.5,
        scope=manifest['scope'], CUDA_device=torch.cuda.get_device_name(0))
    def status(**changes):
        protocol.update(changes, wall_seconds=time.monotonic()-start,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write(output/'protocol.json', protocol)
    status()
    try:
        saved=torch.load(stage['checkpoint'],map_location='cpu',weights_only=False)
        cfg=saved['settings']
        if saved['step']!=stage['expected_step'] or cfg['scheme']!='flow':
            raise ValueError('Stage checkpoint step or RF scheme differs')
        if cfg.get('temporal_experiment',{}).get('mode','control')!='control':
            raise ValueError('Only native control RF static path supported')
        if cfg['sampler_intervals']!=18 or abs(cfg['scales']['flow_rho_A']-RHO)>1e-12:
            raise ValueError('Common native source/35NFE recipe differs')
        model=CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False)
        loaded=model.load_state_dict(saved['model'],strict=True)
        protocol.update(checkpoint_step=saved['step'],settings=cfg,trained_corruption_sigma_A=cfg['static_corruption_A'],
            load=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys))
        del saved
        for snapshot in selected:
            begin=time.monotonic();payload=torch.load(snapshot['input'],map_location='cpu',weights_only=False)
            clean=collate_clip_records([payload['record']]).to('cuda:0')
            corrupted=payload['corrupted_A'].to('cuda:0')
            if payload['corruption_sigma_A']!=CORR or payload['source_rho_A']!=RHO:
                raise ValueError('Corruption/source protocol does not match prepared payload')
            if corrupted.shape!=clean.x.shape or clean.frames!=1:
                raise ValueError('Expected one fixed corrupted static snapshot')
            _,group=torch.unique(torch.stack((clean.abid,clean.block_id),-1),dim=0,return_inverse=True)
            centers=corrupted.new_zeros(1,int(group.max())+1,3).index_add_(1,group,corrupted)
            centers=centers/torch.bincount(group)[None,:,None]
            observed=replace(clean,x=corrupted,bpos=centers[:,group])
            with torch.no_grad():
                condition=model.prepare_condition(observed,task='static')
                query=clean.time_ps
                # Separate identically seeded generator reconstructs the exact
                # native source for diagnostics; it never consumes sampler RNG.
                noise=torch.randn(clean.x.shape,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(29))
                source=condition.reference_A[None]+model.rho_A*noise
                before=model.forward_calls
                full=model.sample(condition,query,steps=18,generator=torch.Generator(device='cuda:0').manual_seed(29)).float()
                full_nfe=model.forward_calls-before
                teacher=.5*source+.5*clean.x
                before=model.forward_calls
                endpoint=teacher+.5*model(teacher,condition=condition,query_time_ps=query,progress=teacher.new_full((1,),.5))
                teacher_nfe=model.forward_calls-before
            torch.cuda.synchronize()
            raw=dict(clean_A=clean.x.cpu(),corrupted_A=corrupted.cpu(),source_A=source.cpu(),
                full_A=full.cpu(),teacher_input_A=teacher.cpu(),teacher_endpoint_A=endpoint.cpu(),
                atom_identity=payload['record']['atom_identity'],input=snapshot['input'],coordinate_unit='angstrom',
                full_NFE=full_nfe,teacher_NFE=teacher_nfe)
            raw_path=output/(snapshot['id']+'.pt');torch.save(raw,raw_path)
            if full_nfe!=35 or teacher_nfe!=1 or any(not bool(torch.isfinite(v).all()) for k,v in raw.items() if k.endswith('_A')):
                raise ValueError('Raw retained: nonfinite output or native NFE mismatch')
            protocol['snapshots'][snapshot['id']]=dict(status='completed',raw=str(raw_path),
                full_NFE=full_nfe,teacher_NFE=teacher_nfe,wall_seconds=time.monotonic()-begin)
            status()
            print(json.dumps(dict(stage=args.stage,snapshot=snapshot['id'],**protocol['snapshots'][snapshot['id']])),flush=True)
        status(status='completed')
    except Exception as exc:
        status(status='failed',error=repr(exc));raise


def score(args):
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Score only in existing torch-ito with CUDA hidden')
    import numpy as np
    import torch
    sys.path.insert(0,str(MAIN));sys.path.insert(0,str(MAIN/'tasks/main-integration-20261016'))
    import compare_geometry_continuation as shared
    from molvid.evaluation.common_panel import prepare_case,write_json
    from molvid.evaluation.local_geometry import prepare_local_geometry
    shared.load_engine(shared.ENGINE)
    start,cpu=time.monotonic(),time.process_time()
    manifest=read(args.manifest);output=args.output or RUN/'static_probe/scores'
    output.mkdir(parents=True,exist_ok=False)
    contexts={};paired_sources={};rows=[];errors=[]
    for stage in args.stages:
        folder=args.generation_root/stage
        try:protocol=read(folder/'protocol.json')
        except (OSError,json.JSONDecodeError):protocol={}
        for snapshot in manifest['snapshots']:
            identity=dict(stage=stage,snapshot=snapshot['id'],case=snapshot['case'],dataset=snapshot['dataset'],reference_future_index=snapshot['reference_future_index'])
            try:
                info=protocol.get('snapshots',{}).get(snapshot['id'],{})
                if info.get('status')!='completed':raise ValueError('Snapshot generation not completed')
                if Path(protocol['checkpoint']).resolve()!=Path(manifest['stages'][stage]['checkpoint']).resolve():raise ValueError('Checkpoint source mismatch')
                raw=torch.load(info['raw'],map_location='cpu',weights_only=False)
                payload=torch.load(snapshot['input'],map_location='cpu',weights_only=False)
                if not np.array_equal(raw['atom_identity'],payload['record']['atom_identity']):raise ValueError('Atom identity mismatch')
                target=np.asarray(raw['clean_A'],float);corrupted=np.asarray(raw['corrupted_A'],float);source=np.asarray(raw['source_A'],float)
                if not np.array_equal(target,np.asarray(payload['record']['x'])) or not np.array_equal(corrupted,np.asarray(payload['corrupted_A'])):raise ValueError('Fixed clean/corruption mismatch')
                if snapshot['id'] in paired_sources and not np.array_equal(source,paired_sources[snapshot['id']]):raise ValueError('Native source not identical across stages')
                paired_sources[snapshot['id']]=source
                if snapshot['case'] not in contexts:
                    prepared=prepare_case(snapshot['reference_case'],MAIN_RUN/'short50_generated_20261017')
                    topo=prepared.topology
                    protein=~np.asarray(topo['ligand_mask'],bool)
                    heavy=~np.isin(np.char.upper(topo['elements'].astype(str)),['H','D'])
                    bb=protein & heavy & np.isin(topo['atom_names'],['N','CA','C','O','OXT'])
                    contexts[snapshot['case']]=(prepare_local_geometry(prepared.reference,topo),bb,
                        dict(all=np.ones(len(bb),bool),protein=protein,protein_backbone=bb,
                             protein_sidechain=protein & heavy & ~bb,ligand=~protein))
                geometry,fit,parts=contexts[snapshot['case']]
                for variant,key in [('clean','clean_A'),('corrupted','corrupted_A'),('source','source_A'),('full','full_A'),('teacher_input','teacher_input_A'),('teacher_endpoint','teacher_endpoint_A')]:
                    xyz=np.asarray(raw[key],float)
                    if xyz.shape!=target.shape or not np.isfinite(xyz).all():raise ValueError('Invalid preserved coordinates')
                    aligned=shared.align(xyz,target[0],fit)
                    metrics={}
                    for name,mask in parts.items():
                        metrics[name+'.atoms']=int(mask.sum())
                        for space,coords in [('raw',xyz),('backbone_aligned',aligned)]:
                            metrics[f'{name}.{space}.RMS_error_A']=float(np.sqrt(np.square(coords[:,mask]-target[:,mask]).sum(-1).mean())) if mask.any() else None
                    report=geometry.evaluate(xyz)
                    geo_path=output/f'{stage}_{snapshot["id"]}_{variant}_geometry.json';write_json(geo_path,report)
                    for kind,components in report['metrics'].items():
                        for part,data in components.items():
                            for name,value in data.items():
                                if isinstance(value,(int,float,bool)) or value is None:metrics[f'geometry.{part}.{kind}.{name}']=value
                    rows.append(dict(identity,variant=variant,status='ok',raw=info['raw'],geometry=str(geo_path),**metrics))
            except Exception as exc:
                failure=dict(identity,status='NA',reason=f'{type(exc).__name__}: {exc}');errors.append(failure);rows.append(failure)
    shared.save_csv(output/'rows.csv',rows)
    aggregates=[]
    for stage in args.stages:
        for variant in ('clean','corrupted','source','full','teacher_input','teacher_endpoint'):
            for domain in ('ATLAS','MISATO'):
                selected=[r for r in rows if r['stage']==stage and r.get('variant')==variant and r['dataset']==domain and r['status']=='ok']
                numeric={k for r in selected for k,v in r.items() if shared.finite(v) and k!='reference_future_index'}
                cases=sorted({r['case'] for r in selected})
                aggregates.append(dict(stage=stage,variant=variant,dataset=domain,cases=len(cases),snapshots=len(selected),
                    metrics={k:shared.mean([shared.mean([r.get(k) for r in selected if r['case']==c]) for c in cases]) for k in numeric}))
    summary=dict(status='completed' if not errors else 'incomplete',errors=errors,rows=rows,per_dataset=aggregates,
        requested_snapshots=len(manifest['snapshots'])*len(args.stages),completed_snapshots=sum(r.get('variant')=='full' for r in rows),
        scope=manifest['scope'],paired_sources_checked=True,
        definitions=dict(full='Native static Q1 source, 18 intervals /35 field calls',teacher='Clean-target interpolated s=.5; one field endpoint, not true-source restoration',
            alignment='One whole-protein backbone fit per structure; ligand shares same transform, never independently fit',
            geometry=manifest['geometry_reference'],aggregation='First two fixed snapshots within each case, then equal cases within dataset'),
        execution=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,GPU_seconds=0,model_calls=0))
    write_json(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','errors','requested_snapshots','completed_snapshots','execution')}))
    if errors and not args.allow_partial:raise SystemExit('Incomplete static probe retained; explicit partial output required')


def merge(args):
    """Join existing scores and verify paired inputs; perform no metric evaluation."""
    if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Merge only in existing torch-ito with CUDA hidden')
    import csv
    import numpy as np
    import torch
    start,cpu=time.monotonic(),time.process_time()
    manifest=read(args.manifest);output=args.output or RUN/'static_probe/scores'
    output.mkdir(parents=True,exist_ok=False)
    rows=[];aggregates=[];errors=[];sources=[];identities=set();definitions=None
    for folder in args.sources:
        path=folder/'summary.json'
        try:
            summary=read(path)
            if summary['scope']!=manifest['scope']:raise ValueError('Snapshot scope differs')
            if definitions is None:definitions=summary['definitions']
            elif summary['definitions']!=definitions:raise ValueError('Score definitions differ')
            stages={r['stage'] for r in summary['rows']}
            if not stages<=set(STAGES):raise ValueError('Unknown stage in source summary')
            for r in summary['rows']:
                identity=(r['stage'],r['snapshot'],r.get('variant'))
                if identity in identities:raise ValueError('Duplicate source row: '+str(identity))
                identities.add(identity)
            rows.extend(summary['rows']);aggregates.extend(summary['per_dataset'])
            errors.extend(dict(e,source_summary=str(path)) for e in summary['errors'])
            sources.append(dict(summary=str(path.resolve()),status=summary['status'],stages=sorted(stages),
                execution=summary['execution'],requested_snapshots=summary['requested_snapshots'],
                completed_snapshots=summary['completed_snapshots']))
        except Exception as exc:
            errors.append(dict(source_summary=str(path),reason=f'{type(exc).__name__}: {exc}'))
    full={(r['stage'],r['snapshot']):r for r in rows if r.get('variant')=='full' and r.get('status')=='ok'}
    pairing=[];missing=[]
    def same_bits(a,b):
        a=torch.as_tensor(a).detach().cpu().contiguous();b=torch.as_tensor(b).detach().cpu().contiguous()
        return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    for snapshot in manifest['snapshots']:
        sid=snapshot['id'];baseline=None;baseline_stage=None
        try:payload=torch.load(snapshot['input'],map_location='cpu',weights_only=False)
        except Exception as exc:
            errors.append(dict(snapshot=sid,reason=f'Fixed input cannot be read: {exc}'));continue
        for stage in STAGES:
            row=full.get((stage,sid))
            if row is None:
                missing.append(dict(stage=stage,snapshot=sid,reason='No completed full-source score row'))
                continue
            item=dict(stage=stage,snapshot=sid,raw=row['raw'],baseline_stage=baseline_stage)
            try:
                raw=torch.load(row['raw'],map_location='cpu',weights_only=False)
                if not np.array_equal(raw['atom_identity'],payload['record']['atom_identity']):raise ValueError('Atom identity differs')
                if not same_bits(raw['clean_A'],payload['record']['x']):raise ValueError('Clean structure differs from fixed input')
                if not same_bits(raw['corrupted_A'],payload['corrupted_A']):raise ValueError('Corruption differs from fixed input')
                if (raw.get('full_NFE'),raw.get('teacher_NFE'))!=(35,1):raise ValueError('Native full/teacher NFE differs')
                if baseline is None:
                    baseline={k:raw[k] for k in ('clean_A','corrupted_A','source_A')};baseline_stage=stage
                equal={k:same_bits(raw[k],baseline[k]) for k in baseline}
                item.update(baseline_stage=baseline_stage,bitwise_equal=equal,status='ok' if all(equal.values()) else 'failed')
                if not all(equal.values()):raise ValueError('Cross-stage clean/corruption/source bit pattern differs')
            except Exception as exc:
                item.update(status='failed',reason=f'{type(exc).__name__}: {exc}');errors.append(item.copy())
            pairing.append(item)
    complete=not errors and not missing and len(full)==len(STAGES)*len(manifest['snapshots'])
    # Source numeric rows/aggregates are preserved verbatim. Any failed pairing
    # makes the merged result explicitly incomplete instead of changing scores.
    fieldnames=sorted({k for r in rows for k in r})
    with (output/'rows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    execution=dict(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,GPU_seconds=0,
        model_calls=0,geometry_evaluations=0,kind='Read existing scores and compare preserved input tensors only')
    total=dict(wall_seconds=execution['wall_seconds']+sum(s['execution']['wall_seconds'] for s in sources),
        cpu_seconds=execution['cpu_seconds']+sum(s['execution']['cpu_seconds'] for s in sources),GPU_seconds=0,
        meaning='Sum of old/new scoring stages and this merge; not elapsed calendar duration')
    summary=dict(status='completed' if complete else 'incomplete',errors=errors,missing_snapshots=missing,rows=rows,
        per_dataset=aggregates,requested_snapshots=len(STAGES)*len(manifest['snapshots']),completed_snapshots=len(full),
        scope=manifest['scope'],definitions=definitions,paired_sources_checked=complete,
        paired_input_checks=pairing,source_summaries=sources,execution=execution,total_scoring_and_merge=total,
        original_geometry_scores_recomputed=False)
    write(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','requested_snapshots','completed_snapshots','execution','total_scoring_and_merge')}))
    if not complete and not args.allow_partial:raise SystemExit('Incomplete merged probe retained; explicit --allow-partial required')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--stage',choices=STAGES,required=True)
    g.add_argument('--manifest',type=Path,default=RUN/'static_probe/inputs/manifest.json')
    g.add_argument('--snapshots',nargs='+');g.add_argument('--output',type=Path)
    g.add_argument('--allocator-GiB',type=float,default=32.)
    s=sub.add_parser('score');s.add_argument('--stages',choices=STAGES,nargs='+',default=list(STAGES))
    s.add_argument('--manifest',type=Path,default=RUN/'static_probe/inputs/manifest.json')
    s.add_argument('--generation-root',type=Path,default=RUN/'static_probe/generated')
    s.add_argument('--output',type=Path);s.add_argument('--allow-partial',action='store_true')
    m=sub.add_parser('merge',help='Reuse old3/new2 scores; verify all paired raw inputs without scoring again')
    m.add_argument('--manifest',type=Path,default=RUN/'static_probe/inputs/manifest.json')
    m.add_argument('--sources',type=Path,nargs='+',default=[RUN/'static_probe/scores_old',RUN/'static_probe/scores_new'])
    m.add_argument('--output',type=Path);m.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    if args.command=='generate':generate(args)
    elif args.command=='score':score(args)
    else:merge(args)
