"""N1 frozen-A1 native-grid decoder/solver diagnostics; targets are scorer-only."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..data.io import read_topology
from ..data.preprocess import topology_metadata, make_clip_record
from ..data.batch import collate_clip_records
from ..training.predictive_data import load_parent, VIEWS
from ..training.batches import prepare_frame_joint_batch
from ..flow.sampling import euler_sample_frames
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from ..provenance import freeze_run_identity
from ..runtime import configure_device, atomic_write_json, sha256_file
from .static_response import validation_plan
from .geometry import bond_errors
from .physical import sampled_clash_metrics
from .temporal import temporal_metrics_v3


def native_records(output):
    """Read exact XTC frames, not interpolation. Fixed observed span=300 ps."""
    import mdtraj as md
    manifest=json.loads((VIEWS/'paired_view_manifest.json').read_text())
    raw_root=Path(manifest['raw_root']);systems=validation_plan(systems=4)['systems']
    records=[];sources=[]
    for system in systems:
        name=system.removeprefix('atlas_');pdb=raw_root/name/f'{name}.pdb';xtc=raw_root/name/f'{name}_prod_R1_fit.xtc'
        if not pdb.exists() or not xtc.exists():
            raise FileNotFoundError(f'native reference missing, cannot invent time labels: {pdb}, {xtc}')
        metadata=topology_metadata(read_topology(pdb),complex_topology=False)
        specs=[(anchor,100) for anchor in (450,2450)]+[(450,lag) for lag in (80,200,400,1000)]
        indices={}
        for anchor,lag in specs:
            indices[anchor,lag]=[anchor-30,anchor-20,anchor-10,anchor]+[anchor+i*(lag//10) for i in range(1,13)]
        wanted={i for values in indices.values() for i in values};selected={};offset=0;previous=None
        for chunk in md.iterload(str(xtc),top=str(pdb),chunk=256):
            for j,timestamp in enumerate(chunk.time):
                index=offset+j
                if previous is not None and abs(float(timestamp)-previous-10.)>1e-3:
                    raise ValueError('native XTC is not the declared 10 ps grid')
                previous=float(timestamp)
                if index in wanted: selected[index]=(chunk.xyz[j].copy()*10.,float(timestamp))
            offset+=len(chunk)
            if offset>max(wanted): break
        if set(selected)!=wanted: raise ValueError('native trajectory too short for declared non-overlapping anchors')
        sources.append(dict(system=system,pdb=str(pdb),pdb_sha256=sha256_file(pdb),xtc=str(xtc),xtc_sha256=sha256_file(xtc)))
        for anchor,lag in specs:
            ids=indices[anchor,lag];frames=np.stack([selected[i][0] for i in ids]);times=np.array([selected[i][1] for i in ids])
            record=make_clip_record(frames,times,metadata,sample_id=f'{system}_R1_native{anchor}_q{lag}',source='atlas',
                system_id=system,replica='R1',split='valid',source_stride=1,native_dt_ps=10.,timestamp_provenance='native_XTC_exact_frame_selection')
            record['time_bucket_id']=f'fixed_H4_100ps_query_{lag}ps';record['lag_ps']=lag
            record['native_frame_indices']=ids;record['anchor']=anchor
            records.append(record)
        # Observed coordinates and times must remain bit-identical across grids.
        same=[r for r in records if r['system_id']==system and r['anchor']==450]
        for r in same[1:]:
            for k in ('x','bpos','time_ps'):
                assert np.array_equal(r[k][:4],same[0][k][:4]), (system,k)
        print(json.dumps({'phase':'native_data','system':system,'records':len(records)}),flush=True)
    atomic_write_json(output/'native_sources.json',dict(sources=sources,test_opened=False,
        native_dt_ps=10.,selection='first four declared valid systems, R1, anchors450/2450; not a new holdout'))
    return records


def chemistry_indices(bonds,n):
    adjacency=defaultdict(set)
    for a,b in bonds.detach().cpu().T.tolist():
        if a!=b: adjacency[a].add(b);adjacency[b].add(a)
    angles=[];volumes=[];torsions=[]
    for center,neighbors in sorted(adjacency.items()):
        neighbors=sorted(neighbors)
        angles.extend((a,center,b) for a,b in combinations(neighbors,2))
        if len(neighbors)>=3: volumes.append((center,*neighbors[:3]))
    for b,neighbors in sorted(adjacency.items()):
        for c in sorted(neighbors):
            if b>=c: continue
            for a in sorted(adjacency[b]-{c}):
                for d in sorted(adjacency[c]-{b,a}): torsions.append((a,b,c,d))
    # Bounded deterministic diagnostic selections, no selection by future error.
    return angles[:4096],volumes[:4096],torsions[:2048]


def internal_geometry(pred,truth,bonds):
    angles,volumes,torsions=chemistry_indices(bonds,pred.shape[1]);result={}
    if angles:
        a,b,c=torch.tensor(angles,device=pred.device).T
        def cosine(x):
            u=torch.nn.functional.normalize(x[:,a]-x[:,b],dim=-1);v=torch.nn.functional.normalize(x[:,c]-x[:,b],dim=-1)
            return (u*v).sum(-1)
        result['angle_cosine_rmse']=float((cosine(pred)-cosine(truth)).square().mean().sqrt())
    if volumes:
        o,a,b,c=torch.tensor(volumes,device=pred.device).T
        def volume(x): return (torch.cross(x[:,a]-x[:,o],x[:,b]-x[:,o],dim=-1)*(x[:,c]-x[:,o])).sum(-1)
        p,t=volume(pred),volume(truth);valid=t.abs()>1e-3
        result['oriented_volume_reference_count']=int(valid.sum())
        result['oriented_volume_sign_mismatch']=float(((p*t<0)&valid).sum()/valid.sum().clamp_min(1))
        # This compares ordered local handedness; it is not an R/S assignment.
    if torsions:
        a,b,c,d=torch.tensor(torsions,device=pred.device).T
        def torsion(x):
            axis=torch.nn.functional.normalize(x[:,c]-x[:,b],dim=-1)
            first=x[:,a]-x[:,b];last=x[:,d]-x[:,c]
            first=first-(first*axis).sum(-1,keepdim=True)*axis
            last=last-(last*axis).sum(-1,keepdim=True)*axis
            return torch.atan2((torch.cross(axis,first,dim=-1)*last).sum(-1),(first*last).sum(-1))
        p,t=torsion(pred),torsion(truth)
        result['torsion_circular_error']=float((1-torch.cos(p-t)).mean())
        result['torsion_increment_circular_error']=float((1-torch.cos((p[1:]-p[:-1])-(t[1:]-t[:-1]))).mean())
    return result


@torch.no_grad()
def run_diagnostic(output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True)
    model,raw,release=load_parent(device)
    protocol=dict(schema='molvid.review.N1.v1',systems=4,anchors=[450,2450],history_frames=4,observed_dt_ps=100,
        query_frames=12,query_lags_ps=[80,100,200,400,1000],primary_solvers=[8,16],primary_seeds=[0,1],
        extra_grid_solver=16,extra_grid_seed=0,draw_cap=48,gpu_wall_cap_seconds=7200,
        oracle='true future latent for scoring only, never generation condition',test_opened=False,
        scope='bounded development diagnostic; grid extension has one draw per system, no equivalence claims')
    freeze_run_identity(output,inputs=[release['selected_checkpoint']['path'],release['configuration']['path'],
        VIEWS/'paired_view_manifest.json'],purpose='N1 frozen decoder solver time',protocol=protocol)
    records=native_records(output)
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    assert sha256_file(recipe)==release['auxiliary_recipe']['sha256']
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    rows=[];draws=0;started=time.monotonic();endpoint_features={}
    for record in records:
        cpu=collate_clip_records([record])
        batch=prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
        truth=batch.coordinate_batch.x.float();context=batch.observed_context;query=batch.query
        arms=[('oracle',0,-1),('persistence',0,-1)]
        if record['lag_ps']==100: arms += [('generated',steps,seed) for steps in (8,16) for seed in (0,1)]
        else: arms += [('generated',16,0)]
        # Fixed atom pairs carry invariant common-endpoint descriptors.
        pair_a=torch.linspace(0,len(cpu.atype)-1,min(64,len(cpu.atype)),device=device).long()
        pair_b=pair_a.roll(17)
        for variant,steps,seed in arms:
            if time.monotonic()-started>7200:
                atomic_write_json(output/'summary.json',dict(status='budget_exhausted',draws=draws,test_opened=False));return
            if variant=='oracle': latent=batch.target_future;future=model.decoder(context,latent,query).coordinates.float()
            elif variant=='persistence': latent=None;future=context.coordinates[-1:].expand(12,-1,-1).float()
            else:
                assert draws<48
                latent,_=euler_sample_frames(model,context=context,query=query,steps=steps,seed=seed)
                future=model.decoder(context,latent,query).coordinates.float();draws+=1
            if not bool(torch.isfinite(future).all()): raise FloatingPointError('nonfinite N1 generated coordinates')
            prediction=torch.cat((context.coordinates.float(),future))
            temporal=temporal_metrics_v3(prediction,truth,batch.coordinate_batch,history_frames=4)
            clash=sampled_clash_metrics(future,batch,physical)
            metrics={**temporal.metrics,**bond_errors(prediction,truth,batch.coordinate_batch,list(range(4,16))),
                     **{k:v for k,v in clash.items() if k!='samples'},**internal_geometry(future,truth[4:],cpu.bond_index)}
            if latent is not None:
                normalized=model.normalize(latent)
                metrics['latent_h_mse']=float((normalized.h-batch.normalized_target.h).square().mean())
                metrics['latent_v_mse']=float((normalized.v-batch.normalized_target.v).square().mean())
            horizons=(query.time_ps[0]-context.latent.time_ps[0,-1]).cpu().tolist()
            predicted_pair=torch.linalg.vector_norm(future[:,pair_a]-future[:,pair_b],dim=-1)
            true_pair=torch.linalg.vector_norm(truth[4:,pair_a]-truth[4:,pair_b],dim=-1)
            reference_pair=torch.linalg.vector_norm(context.coordinates[-1,pair_a]-context.coordinates[-1,pair_b],dim=-1)
            curve=[dict(horizon_ps=h,internal_pair_mse=float((p-t).square().mean()),
                predicted_internal_change_A2=float((p-reference_pair).square().mean()),
                true_internal_change_A2=float((t-reference_pair).square().mean()))
                for h,p,t in zip(horizons,predicted_pair,true_pair)]
            if 400. in horizons and record['anchor']==450 and variant=='generated' and steps==16 and seed==0:
                i=horizons.index(400.);endpoint_features[f"{record['system_id']}_{record['lag_ps']}"]=dict(
                    prediction=predicted_pair[i].cpu(),target=true_pair[i].cpu())
            rows.append(dict(system=record['system_id'],sample_id=record['sample_id'],anchor=record['anchor'],
                lag_ps=record['lag_ps'],variant=variant,steps=steps,seed=seed,metrics=metrics,response_curve=curve))
            atomic_write_json(output/'rows.json',rows)
        print(json.dumps({'phase':'N1','sample':record['sample_id'],'draws':draws,'seconds':time.monotonic()-started}),flush=True)
    comparisons={}
    for system in sorted({r['system_id'] for r in records}):
        ref=endpoint_features[f'{system}_100'];table={}
        for lag in (80,200,400):
            other=endpoint_features[f'{system}_{lag}']
            assert torch.equal(other['target'],ref['target'])
            table[str(lag)]=float((other['prediction']-ref['prediction']).square().mean())
        comparisons[system]=table
    torch.save(endpoint_features,output/'common_endpoint_features.pt')
    atomic_write_json(output/'summary.json',dict(status='completed',draws=draws,rows=len(rows),seconds=time.monotonic()-started,
        common_endpoint_single_draw_pair_mse=comparisons,test_opened=False,
        limitations=['Endpoint discrepancies are paired single-draw sensitivity, NOT distribution equivalence tests.',
                    'One training checkpoint; no training-seed CI.','Oriented volume is not absolute stereochemical R/S validation.']))
