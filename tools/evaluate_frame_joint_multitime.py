"""CUDA multi-time Frame Joint evaluation with paired lag and clock diagnostics."""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from molvid.checkpoints import load_frame_joint_inference
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.geometry import (
    _align_mask, _align_trajectory_to_first, _aligned_prediction, _drmsd, _frame_atom_mask, _kabsch_align,
    _mask, _nonbond_count, _nonbond_pairs, _pairs, _bond_rmse, _frame_view,
)
from molvid.generation import sample_frame_joint
from molvid.runtime import atomic_write_json, configure_device, sha256_file
from molvid.training.batches import prepare_frame_joint_batch

PATHS=("persistence","clean_latent_decoder_oracle","generated")
LAGS=(100,200,300,400)
HISTORIES=(4,8)

def scalar(v):
    if v is None: return None
    if isinstance(v,(float,int,str,bool)): return v
    if isinstance(v,torch.Tensor): return float(v.detach().cpu())
    return float(v)

def jsonable(v):
    if isinstance(v,Mapping): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(np.integer,np.floating,np.bool_)): return v.item()
    if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
    return v

def summary_metrics(metrics: Mapping[str,Any])->dict[str,Any]:
    f=metrics["future"]; r=metrics["rmsf"]["future"]; d=metrics["dynamic"]["future"]
    return {"aligned_rmsd":scalar(f.get("aligned_rmsd")),"drmsd":scalar(f.get("drmsd")),
      "bond_rmse":scalar(f.get("bond_rmse")),"contact_f1":scalar(f.get("contact_f1")),
      "contact_occupancy_mae":scalar(f.get("contact_occupancy_mae")),
      "rmsf_prediction":scalar(r.get("prediction")),"rmsf_target":scalar(r.get("target")),
      "rmsf_absolute_error":scalar(r.get("absolute_error")),
      "rmsf_correlation":scalar(r.get("correlation")) if r.get("correlation_available",False) else None,
      "rmsf_spearman":scalar(r.get("spearman")) if r.get("spearman") is not None else None,
      "rmsf_flattening_ratio":(scalar(r.get("prediction"))/scalar(r.get("target")) if r.get("prediction") is not None and r.get("target") not in (None,0) else None),
      "velocity_rmse":scalar(metrics.get("velocity_rmse")),
      "dynamic_correlation":scalar(d.get("dynamic_correlation")) if d.get("available",False) else None,
      "dynamic_acf_prediction":scalar(d.get("prediction")) if d.get("acf_available",False) else None,
      "dynamic_acf_target":scalar(d.get("target")) if d.get("acf_available",False) else None}

def pearson(a,b):
    ok=torch.isfinite(a)&torch.isfinite(b)
    a=a[ok]; b=b[ok]
    if a.numel()<2: return None,"insufficient_atoms"
    aa=a-a.mean(); bb=b-b.mean(); den=torch.linalg.vector_norm(aa)*torch.linalg.vector_norm(bb)
    if float(den)<=1e-12: return None,"zero_variance"
    return float((aa*bb).sum().detach().cpu()/den.detach().cpu()),None

def spearman(a,b):
    ok=torch.isfinite(a)&torch.isfinite(b); a=a[ok]; b=b[ok]
    if a.numel()<2: return None,"insufficient_atoms"
    # Stable rank ordering is sufficient for the diagnostic; exact ties are marked unavailable.
    if torch.unique(a).numel()<2 or torch.unique(b).numel()<2: return None,"zero_variance"
    ra=torch.argsort(torch.argsort(a)).to(torch.float32); rb=torch.argsort(torch.argsort(b)).to(torch.float32)
    return pearson(ra,rb)

def rmsf_arrays(pred,target,batch,history):
    frames=int(pred.shape[0]); mask=_mask(batch,frames,pred.device); future_idx=list(range(int(history),frames)); idx=torch.as_tensor(future_idx,device=pred.device,dtype=torch.long)
    view=_frame_view(batch,future_idx); pred_future=pred.index_select(0,idx); target_future=target.index_select(0,idx); mask_future=mask.index_select(0,idx)
    frame_valid=_frame_atom_mask(view,len(future_idx),pred.device); align=_align_mask(view,len(future_idx),pred.device)
    ap=_align_trajectory_to_first(pred_future,view,frame_valid,align); at=_align_trajectory_to_first(target_future,view,frame_valid,align)
    weights=mask_future.to(ap.dtype); counts=weights.sum(0); safe=counts.clamp_min(1.0)
    pm=(ap*weights.unsqueeze(-1)).sum(0)/safe.unsqueeze(-1); tm=(at*weights.unsqueeze(-1)).sum(0)/safe.unsqueeze(-1)
    p=torch.sqrt((((ap-pm.unsqueeze(0)).square().sum(-1))*weights).sum(0)/safe); t=torch.sqrt((((at-tm.unsqueeze(0)).square().sum(-1))*weights).sum(0)/safe)
    valid=counts>0; p=torch.where(valid,p,torch.full_like(p,float('nan'))); t=torch.where(valid,t,torch.full_like(t,float('nan')))
    mae=float((p[valid]-t[valid]).abs().mean().detach().cpu()) if bool(valid.any()) else None
    pr,pr_reason=pearson(p,t); sp,sp_reason=spearman(p,t)
    return p,t,{"available":bool(valid.any()),"valid_atoms":int(valid.sum()),"mae":mae,"pearson":pr,"pearson_reason":pr_reason,"spearman":sp,"spearman_reason":sp_reason,"prediction_mean":float(p[valid].mean().detach().cpu()) if bool(valid.any()) else None,"target_mean":float(t[valid].mean().detach().cpu()) if bool(valid.any()) else None,"alignment":"per-frame-Kabsch-to-first-future-frame","mask":"loss_mask"}

def align_to_anchor(coords,batch,anchor):
    frames,n,_=coords.shape; out=coords.clone(); am=_align_mask(batch,frames,coords.device)
    ref=coords[anchor]; ref_mask=am[anchor]
    for i in range(frames):
        common=am[i]&ref_mask
        if not bool(common.any()): continue
        src=coords[i,common]; rp=ref[common]; aligned,mode=_kabsch_align(src,rp)
        if mode=="none": continue
        sc=src.mean(0); rc=rp.mean(0)
        if mode=="translation_only": out[i]=coords[i]-sc+rc
        else:
            cov=(src-sc).transpose(0,1)@(rp-rc); left,_,rt=torch.linalg.svd(cov,full_matrices=False); rot=left@rt
            if torch.linalg.det(rot)<0: left=left.clone(); left[:,-1]*=-1; rot=left@rt
            out[i]=(coords[i]-sc)@rot+rc
    return out

def lag_msd(pred,target,batch,history):
    frames,n,_=pred.shape; mask=_mask(batch,frames,pred.device); anchor=int(history)-1
    ap=align_to_anchor(pred,batch,anchor); at=align_to_anchor(target,batch,anchor)
    times=torch.as_tensor(batch.time_ps,device=pred.device,dtype=torch.float32)[0]
    values=[]
    for delta in (1,2,3):
        per=[]
        for i in range(history,frames-delta):
            j=i+delta; m=mask[i]&mask[j]
            if bool(m.any()):
                dp=ap[j,m]-ap[i,m]; dt=at[j,m]-at[i,m]
                per.append((dp.square().sum(-1).mean(),dt.square().sum(-1).mean()))
        if per:
            pv=torch.stack([x[0] for x in per]).mean(); tv=torch.stack([x[1] for x in per]).mean()
            values.append({"delta_frames":delta,"delta_ps":float((times[history+delta]-times[history]).detach().cpu()),
              "prediction":float(pv.detach().cpu()),"target":float(tv.detach().cpu()),
              "absolute_error":float((pv-tv).abs().detach().cpu()),"pair_count":len(per)})
        else: values.append({"delta_frames":delta,"delta_ps":None,"prediction":None,"target":None,"absolute_error":None,"pair_count":0})
    curve=[]
    for j in range(history,frames):
        m=mask[anchor]&mask[j]
        if bool(m.any()):
            dp=ap[j,m]-ap[anchor,m]; dt=at[j,m]-at[anchor,m]
            pv=dp.square().sum(-1).mean(); tv=dt.square().sum(-1).mean()
            curve.append({"frame_index":j,"lag_ps":float((times[j]-times[anchor]).detach().cpu()),
              "prediction":float(pv.detach().cpu()),"target":float(tv.detach().cpu()),
              "absolute_error":float((pv-tv).abs().detach().cpu()),"atom_count":int(m.sum())})
    return {"lag_msd":values,"t0_curve":curve,"reference":"last observed frame; per-frame Kabsch to common t0"}

def _contact_fast(pred,target,batch,mask,frames,cutoff=4.5):
    source,destination=_nonbond_pairs(batch,int(pred.shape[1]),pred.device)
    values=[]
    if source.numel()==0: return {"contact_f1":1.0,"contact_occupancy_mae":0.0}
    for i in frames:
        pm=mask[i,source]&mask[i,destination]; total=_nonbond_count(batch,mask[i])
        if not bool(pm.any()): values.append((1.0 if total==0 else 0.0,0.0)); continue
        src=source[pm]; dst=destination[pm]
        pd=torch.linalg.vector_norm(pred[i,src]-pred[i,dst],dim=-1)<cutoff
        td=torch.linalg.vector_norm(target[i,src]-target[i,dst],dim=-1)<cutoff
        tp=int((pd&td).sum()); fp=int((pd&~td).sum()); fn=int((~pd&td).sum()); tc=int(td.sum())
        union=tp+fp+fn; f1=(2*tp/(2*tp+fp+fn)) if union else 1.0
        occ=(fp+fn)/float(max(total,1)); values.append((f1,occ))
    if not values: return {"contact_f1":0.0,"contact_occupancy_mae":0.0}
    return {"contact_f1":sum(v[0] for v in values)/len(values),"contact_occupancy_mae":sum(v[1] for v in values)/len(values)}

def per_frame(pred,target,batch,history,aligned=None):
    mask=_mask(batch,int(pred.shape[0]),pred.device); frames=list(range(int(history),int(target.shape[0])))
    if aligned is None: aligned,_=_aligned_prediction(pred,target,batch,frames=frames)
    times=torch.as_tensor(batch.time_ps,device=pred.device,dtype=torch.float32)[0]; base=times[history-1]; out=[]
    pairs=_pairs(batch,pred.device); source,dest=(pairs[0],pairs[1]) if pairs is not None else (None,None)
    for i in frames:
        m=mask[i]; d=(aligned[i,m]-target[i,m]); ar=float(d.square().sum(-1).mean().sqrt().detach().cpu()) if bool(m.any()) else 0.0
        idx=torch.nonzero(m,as_tuple=False).flatten(); dr=float((torch.pdist(pred[i,idx])-torch.pdist(target[i,idx])).square().mean().sqrt().detach().cpu()) if idx.numel()>=2 else 0.0
        if source is None: br=0.0
        else:
            pm=m[source]&m[dest]
            if bool(pm.any()):
                pd=torch.linalg.vector_norm(pred[i,source]-pred[i,dest],dim=-1); td=torch.linalg.vector_norm(target[i,source]-target[i,dest],dim=-1); br=float((pd[pm]-td[pm]).square().mean().sqrt().detach().cpu())
            else: br=0.0
        out.append({"frame_index":i,"distance_from_last_observation_ps":float((times[i]-base).detach().cpu()),"aligned_rmsd":ar,"drmsd":dr,"bond_rmse":br})
    return out

def _dynamic_fast(pred,target,batch,history,aligned):
    frames=list(range(int(history),int(pred.shape[0]))); mask=_mask(batch,int(pred.shape[0]),pred.device)
    if len(frames)<3: return {"available":False,"reason":"insufficient_frames","dynamic_correlation":None,"acf_available":False,"prediction":None,"target":None}
    idx=torch.as_tensor(frames,device=pred.device); pm=mask.index_select(0,idx); ap=aligned.index_select(0,idx); at=_align_trajectory_to_first(target,batch,_frame_atom_mask(batch,int(target.shape[0]),target.device),_align_mask(batch,int(target.shape[0]),target.device)).index_select(0,idx)
    stable=pm.all(0)
    if not bool(stable.any()): return {"available":False,"reason":"no_stable_atoms","dynamic_correlation":None,"acf_available":False,"prediction":None,"target":None}
    pv=ap[1:,stable]-ap[:-1,stable]; tv=at[1:,stable]-at[:-1,stable]; pv=pv-pv.mean(0,keepdim=True); tv=tv-tv.mean(0,keepdim=True)
    corr,_=pearson(pv.reshape(-1),tv.reshape(-1)); pacf,_=pearson(pv[:-1].reshape(-1),pv[1:].reshape(-1)) if pv.shape[0]>=2 else (None,None); tacf,_=pearson(tv[:-1].reshape(-1),tv[1:].reshape(-1)) if tv.shape[0]>=2 else (None,None)
    return {"available":corr is not None,"reason":None if corr is not None else "zero_variance","dynamic_correlation":corr,"acf_available":pacf is not None and tacf is not None,"prediction":pacf,"target":tacf}

def fast_metrics(pred,target,batch,history):
    frames=list(range(int(history),int(target.shape[0]))); mask=_mask(batch,int(pred.shape[0]),pred.device); aligned,_=_aligned_prediction(pred,target,batch,frames=frames)
    vals=[]; dr=[]
    for i in frames:
        m=mask[i]
        if bool(m.any()): vals.append((aligned[i,m]-target[i,m]).square().sum(-1).mean())
        idx=torch.nonzero(m,as_tuple=False).flatten()
        if idx.numel()>=2: dr.append((torch.pdist(pred[i,idx])-torch.pdist(target[i,idx])).square().mean())
    br=_bond_rmse(pred,target,batch,mask,frames)
    c=_contact_fast(pred,target,batch,mask,frames)
    pa,ta,rs=rmsf_arrays(pred,target,batch,history)
    lags=lag_msd(pred,target,batch,history); pf=per_frame(pred,target,batch,history,aligned)
    dynamic=_dynamic_fast(pred,target,batch,history,aligned)
    vel=[]
    for i in range(max(history,1),int(pred.shape[0])):
        m=mask[i]&mask[i-1]
        if bool(m.any()): vel.append((pred[i,m]-pred[i-1,m]-target[i,m]+target[i-1,m]).square().sum(-1).mean())
    future={"aligned_rmsd":float(torch.stack(vals).mean().sqrt().detach().cpu()) if vals else 0.0,"drmsd":float(torch.stack(dr).mean().sqrt().detach().cpu()) if dr else 0.0,"bond_rmse":float(br),**c}
    m={"future":future,"rmsf":{"future":{"prediction":rs["prediction_mean"],"target":rs["target_mean"],"absolute_error":rs["mae"],"correlation":rs["pearson"],"correlation_available":rs["pearson"] is not None,"spearman":rs["spearman"],"spearman_available":rs["spearman"] is not None}},"dynamic":{"future":dynamic},"velocity_rmse":float(torch.stack(vel).mean().sqrt().detach().cpu()) if vel else 0.0}
    return m,rs,pa,ta,lags,pf

def metrics_for(pred,target,batch,history):
    return fast_metrics(pred,target,batch,history)

def record_key(meta,path,seed=None,clock="true"):
    return f"{meta['sample_id']}|H{meta['history_frames']}|{path}|seed={seed}|clock={clock}"

def meta_for(record):
    sid=str(record["sample_id"]); pre=sid.rsplit("_R",1); system=pre[0]; replica="R"+pre[1].split("_",1)[0]
    return {"sample_id":sid,"system":system,"replica":replica,"lag_ps":int(record["lag_ps"]),"history_frames":int(record["history_frames"]),
      "anchor_id":str(record["anchor_id"]),"anchor_window_300":int(record["anchor_window_300"]),"anchor_raw_frame_index":int(record["anchor_raw_frame_index"]),
      "anchor_time_ps":float(record["anchor_time_ps"]),"source_sample_id_300":str(record["source_sample_id_300"]),
      "source_raw_frame_indices":jsonable(record["source_raw_frame_indices"]),"source_absolute_time_ps":jsonable(record["source_absolute_time_ps"])}

def wrong_clock_record(record,dt=100):
    out=dict(record); t=np.arange(len(record["time_ps"]),dtype=np.float32)*float(dt); out["time_ps"]=t; out["delta_time_ps"]=np.full(len(t)-1,float(dt),dtype=np.float32); out["time_bucket_id"]=f"dt_{dt}ps_wrong_clock"; return out

def hash_tensor(t): return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def leakage_check(model,record,device,steps):
    batch=collate_clip_records([record]); h=int(record["history_frames"]); base=batch.to(device)
    p1,_=sample_frame_joint(model,template=batch,prefix_coordinates=batch.x[:h],history_frames=h,steps=steps,seed=0)
    altered=dict(record); altered["x"]=np.array(record["x"],copy=True); altered["bpos"]=np.array(record["bpos"],copy=True)
    altered["x"][h:]+=123.456; altered["bpos"][h:]-=77.0
    batch2=collate_clip_records([altered]); p2,_=sample_frame_joint(model,template=batch2,prefix_coordinates=batch2.x[:h],history_frames=h,steps=steps,seed=0)
    diff=float((p1-p2).abs().max().detach().cpu()); return {"sample_id":record["sample_id"],"history_frames":h,"steps":steps,"max_abs_difference":diff,"passed":diff==0.0}

def expected_evaluation_rows(records, seeds):
    """Return the exact protocol row count for the selected view family."""
    seed_count=len(tuple(seeds))
    wrong_clock_records=sum(
        int(record["history_frames"])==4 and int(record["lag_ps"]) in (200,300,400)
        for record in records
    )
    return 2*len(records)+seed_count*(len(records)+wrong_clock_records)

def args_parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--codec",type=Path,required=True); p.add_argument("--codec-sha256",required=True)
    p.add_argument("--paired-store",type=Path,required=True); p.add_argument("--paired-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--steps",type=int,default=16)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2]); p.add_argument("--wrong-clock-dt",type=int,default=100)
    p.add_argument("--check-only",action="store_true"); p.add_argument("--limit",type=int); p.add_argument("--sample-id",action="append"); p.add_argument("--resume",action="store_true")
    return p.parse_args()

def main():
    args=args_parse(); device=configure_device(args.device,deterministic=True)
    if device.type!="cuda": raise RuntimeError("CUDA is required for model evaluation")
    out=args.output; out.mkdir(parents=True,exist_ok=True); (out/"coordinates").mkdir(exist_ok=True); (out/"checks").mkdir(exist_ok=True)
    loaded=load_frame_joint_inference(args.checkpoint,expected_sha256=args.checkpoint_sha256,codec_path=args.codec,codec_sha256=args.codec_sha256,device=device)
    model=loaded["model"]; model.eval(); ds=ClipMMapDataset(args.paired_store)
    by_id={str(row[0]):i for i,row in enumerate(ds._index)}
    manifest=json.loads(args.paired_manifest.read_text())
    ids=list(manifest["sample_ids"] if "sample_ids" in manifest else by_id)
    if args.sample_id: ids=[x for x in ids if x in set(args.sample_id)]
    ids=[x for x in ids if x in by_id]
    if args.limit: ids=ids[:int(args.limit)]
    if not ids: raise RuntimeError("no paired records selected")
    try:
      checks={"device":str(device),"cuda_device_name":torch.cuda.get_device_name(device),"checkpoint_sha256":args.checkpoint_sha256,"codec_sha256":args.codec_sha256,"steps":args.steps}
      existing_checks=out/"checks"/"checks.json"
      if args.resume and existing_checks.exists():
        checks=json.loads(existing_checks.read_text())
      else:
        preferred_check_records=[]; fallback_check_records=[]; changed_clock_records=[]
        for sid in ids:
          r=ds[by_id[sid]]
          if int(r["history_frames"]) in HISTORIES:
            fallback_check_records.append(r)
            if int(r["lag_ps"])!=int(args.wrong_clock_dt): changed_clock_records.append(r)
            if int(r["lag_ps"])==300: preferred_check_records.append(r)
        check_records=(preferred_check_records or changed_clock_records or fallback_check_records)[:2]
        if not check_records: raise RuntimeError("no supported H4/H8 records available for CUDA check")
        finite=[]; prefix=[]; times=[]
        for r in check_records:
          h=int(r["history_frames"]); b=collate_clip_records([r]); pred,g=sample_frame_joint(model,template=b,prefix_coordinates=b.x[:h],history_frames=h,steps=args.steps,seed=0)
          finite.append(bool(torch.isfinite(pred).all())); prefix.append(bool(torch.equal(pred[:h].cpu(),b.x[:h]))); times.append({"sample_id":r["sample_id"],"h":h,"lag_ps":r["lag_ps"],"time_ps":jsonable(r["time_ps"]),"delta_ps":jsonable(r["delta_time_ps"]),"shape":list(pred.shape),"finite":finite[-1],"prefix_exact":prefix[-1],"metadata":jsonable(g)})
        checks["sample_checks"]=times; checks["finite_passed"]=all(finite); checks["prefix_passed"]=all(prefix)
        checks["future_leakage"]=leakage_check(model,check_records[0],device,args.steps)
        checks["future_leakage_passed"]=checks["future_leakage"]["passed"]
        true_dt=int(check_records[0]["lag_ps"])
        checks["wrong_clock_time_change"]={"true_dt":true_dt,"wrong_dt":args.wrong_clock_dt,"changes_input":true_dt!=args.wrong_clock_dt,"scoring_clock":"true physical clock"}
        checks["passed"]=bool(checks["finite_passed"] and checks["prefix_passed"] and checks["future_leakage_passed"] and checks["wrong_clock_time_change"]["changes_input"])
        atomic_write_json(existing_checks,checks)
      print(json.dumps({"check_only":True,"passed":checks["passed"],"output":str(out/"checks"/"checks.json")},sort_keys=True),flush=True)
      if args.check_only:
        return 0 if checks["passed"] else 2
      if not checks["passed"]: raise RuntimeError("minimal CUDA checks failed")
      rows_path=out/"rows.jsonl"; completed=set()
      if args.resume and rows_path.exists():
        with rows_path.open() as f:
          for line in f:
            try: completed.add(str(json.loads(line)["row_key"]))
            except Exception: pass
      def emit(row):
        with rows_path.open("a",encoding="utf-8") as f: f.write(json.dumps(jsonable(row),sort_keys=True)+"\n")
      total=0; started_all=time.time(); records=[ds[by_id[sid]] for sid in ids]
      # Seed 0 is intentionally completed for every lag/H bucket before seeds 1/2.
      for seed in list(args.seeds):
        for ri,r in enumerate(records):
          h=int(r["history_frames"]); meta=meta_for(r); true_batch=collate_clip_records([r]); prepared=prepare_frame_joint_batch(model.target_teacher,true_batch,device=device,normalizer=model,history_frames=h); target=prepared.coordinate_batch.x.float(); cb=prepared.coordinate_batch; q=int(true_batch.frames-h)
          persistence=torch.cat((target[:h],target[h-1:h].expand(q,-1,-1)),dim=0)
          oracle_future=model.decoder(prepared.observed_context,prepared.target_future,prepared.query).coordinates.float(); oracle=torch.cat((target[:h],oracle_future),dim=0)
          base_preds=(("persistence",persistence),("clean_latent_decoder_oracle",oracle)) if seed==int(args.seeds[0]) else ()
          if seed==int(args.seeds[0]):
            for path,pred in base_preds:
              key=record_key(meta,path,None,"true")
              if key in completed: continue
              m,rs,pa,ta,lags,pf=metrics_for(pred,target,cb,h); arr_name=hashlib.sha1(key.encode()).hexdigest()+".npz"; np.savez_compressed(out/"coordinates"/arr_name,prediction=pred.detach().cpu().numpy(),rmsf_prediction=pa.detach().cpu().numpy(),rmsf_target=ta.detach().cpu().numpy())
              emit({"row_key":key,"path":path,"clock":"true","seed":None,"meta":meta,"time_ps":jsonable(r["time_ps"]),"source_absolute_time_ps":jsonable(r["source_absolute_time_ps"]),"mask_info":{"coordinate_unit":r.get("coordinate_unit"),"loss_mask_atoms":int(cb.loss_mask.sum()),"align_mask_atoms":int(cb.align_mask.sum()),"align_and_loss_atoms":int((cb.loss_mask&cb.align_mask).sum()),"frame_valid_atoms":int(cb.frame_mask.sum()),"total_atoms":int(cb.atom_count),"total_frames":int(cb.frames)},"summary":summary_metrics(m),"rmsf":rs,"lag_msd":lags,"per_frame":pf,"coordinate_file":arr_name,"generation":None})
              total+=1
          true_pred,gen=sample_frame_joint(model,template=true_batch,prefix_coordinates=true_batch.x[:h],history_frames=h,steps=args.steps,seed=int(seed)); m,rs,pa,ta,lags,pf=metrics_for(true_pred,target,cb,h)
          key=record_key(meta,"generated",seed,"true")
          if key not in completed:
            arr_name=hashlib.sha1(key.encode()).hexdigest()+".npz"; np.savez_compressed(out/"coordinates"/arr_name,prediction=true_pred.detach().cpu().numpy(),rmsf_prediction=pa.detach().cpu().numpy(),rmsf_target=ta.detach().cpu().numpy())
            emit({"row_key":key,"path":"generated","clock":"true","seed":int(seed),"meta":meta,"time_ps":jsonable(r["time_ps"]),"source_absolute_time_ps":jsonable(r["source_absolute_time_ps"]),"mask_info":{"coordinate_unit":r.get("coordinate_unit"),"loss_mask_atoms":int(cb.loss_mask.sum()),"align_mask_atoms":int(cb.align_mask.sum()),"align_and_loss_atoms":int((cb.loss_mask&cb.align_mask).sum()),"frame_valid_atoms":int(cb.frame_mask.sum()),"total_atoms":int(cb.atom_count),"total_frames":int(cb.frames)},"summary":summary_metrics(m),"rmsf":rs,"lag_msd":lags,"per_frame":pf,"coordinate_file":arr_name,"generation":jsonable(gen)})
            total+=1
          # B is H4 only, and always scored with the true target/time batch.
          if h==4 and int(r["lag_ps"]) in (200,300,400):
            wr=wrong_clock_record(r,args.wrong_clock_dt); wb=collate_clip_records([wr]); wrong_pred,wgen=sample_frame_joint(model,template=wb,prefix_coordinates=wb.x[:h],history_frames=h,steps=args.steps,seed=int(seed)); m,rs,pa,ta,lags,pf=metrics_for(wrong_pred,target,cb,h)
            key=record_key(meta,"generated",seed,f"wrong_{args.wrong_clock_dt}ps")
            if key not in completed:
              arr_name=hashlib.sha1(key.encode()).hexdigest()+".npz"; np.savez_compressed(out/"coordinates"/arr_name,prediction=wrong_pred.detach().cpu().numpy(),rmsf_prediction=pa.detach().cpu().numpy(),rmsf_target=ta.detach().cpu().numpy())
              emit({"row_key":key,"path":"generated","clock":f"wrong_{args.wrong_clock_dt}ps","seed":int(seed),"meta":meta,"time_ps":jsonable(r["time_ps"]),"input_time_ps":jsonable(wr["time_ps"]),"source_absolute_time_ps":jsonable(r["source_absolute_time_ps"]),"mask_info":{"coordinate_unit":r.get("coordinate_unit"),"loss_mask_atoms":int(cb.loss_mask.sum()),"align_mask_atoms":int(cb.align_mask.sum()),"align_and_loss_atoms":int((cb.loss_mask&cb.align_mask).sum()),"frame_valid_atoms":int(cb.frame_mask.sum()),"total_atoms":int(cb.atom_count),"total_frames":int(cb.frames)},"summary":summary_metrics(m),"rmsf":rs,"lag_msd":lags,"per_frame":pf,"coordinate_file":arr_name,"generation":jsonable(wgen),"scoring_clock":"true physical clock"})
              total+=1
          if (ri+1)%8==0: print(json.dumps({"seed":seed,"records_done":ri+1,"records_total":len(records),"rows_written":total,"elapsed_s":round(time.time()-started_all,1)},sort_keys=True),flush=True)
      protocol={"schema":"molvid.frame_joint.multitime_eval.v1","checkpoint":{"path":str(args.checkpoint),"sha256":args.checkpoint_sha256},"codec":{"path":str(args.codec),"sha256":args.codec_sha256},"paired_store":{"path":str(args.paired_store),"index_sha256":sha256_file(args.paired_store/"index.txt")},"paired_manifest":{"path":str(args.paired_manifest),"sha256":sha256_file(args.paired_manifest)},"device":str(device),"cuda_device_name":torch.cuda.get_device_name(device),"steps":args.steps,"seeds":list(args.seeds),"wrong_clock_dt":args.wrong_clock_dt,"ids":ids,"expected_rows":expected_evaluation_rows(records,args.seeds),"rows_written":total,"paths":list(PATHS),"physical_time_axis":"distance from last observed frame; source_absolute_time_ps retained","aggregation":"window mean -> replica mean -> equal system mean","generated_conditioning":"observed prefix, topology, query time only","wrong_clock_scoring":"true target/time batch; only H4, 200/300/400ps"}
      atomic_write_json(out/"protocol.json",protocol); print(json.dumps({"complete":True,"rows_written":total,"output":str(out),"elapsed_s":time.time()-started_all},sort_keys=True),flush=True)
    finally: ds.close()
    return 0
if __name__=="__main__": raise SystemExit(main())
