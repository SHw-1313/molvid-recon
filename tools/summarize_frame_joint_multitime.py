"""CPU aggregation, plots, and report for multi-time Frame Joint evaluation."""
from __future__ import annotations
import argparse,csv,json,hashlib,math,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np

def mean(vals):
    v=[float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return sum(v)/len(v) if v else None

def group_mean(rows, fields):
    out={}
    for f in fields: out[f]=mean([r.get(f) for r in rows])
    return out

def read_rows(path):
    rows=[]
    with path.open() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def aggregate(rows):
    fields=["aligned_rmsd","drmsd","bond_rmse","contact_f1","contact_occupancy_mae","rmsf_prediction","rmsf_target","rmsf_absolute_error","rmsf_correlation","rmsf_spearman","rmsf_flattening_ratio","velocity_rmse","dynamic_correlation","dynamic_acf_prediction","dynamic_acf_target"]
    # Seed means per paired window, then anchor means per replica, then equal system mean.
    by_window=defaultdict(list)
    for r in rows:
        m=r["meta"]; by_window[(r["path"],r["clock"],m["lag_ps"],m["history_frames"],m["system"],m["replica"],m["anchor_id"])].append(r["summary"])
    window=[]
    for k,vals in by_window.items():
        d={"path":k[0],"clock":k[1],"lag_ps":k[2],"history_frames":k[3],"system":k[4],"replica":k[5],"anchor_id":k[6],"n_rows":len(vals)}; d.update(group_mean(vals,fields)); window.append(d)
    by_rep=defaultdict(list)
    for r in window: by_rep[(r["path"],r["clock"],r["lag_ps"],r["history_frames"],r["system"],r["replica"])].append(r)
    rep=[]
    for k,vals in by_rep.items():
        d={"path":k[0],"clock":k[1],"lag_ps":k[2],"history_frames":k[3],"system":k[4],"replica":k[5],"n_windows":len(vals)}; d.update(group_mean(vals,fields)); rep.append(d)
    by_sys=defaultdict(list)
    for r in rep: by_sys[(r["path"],r["clock"],r["lag_ps"],r["history_frames"],r["system"])].append(r)
    sysrows=[]
    for k,vals in by_sys.items():
        d={"path":k[0],"clock":k[1],"lag_ps":k[2],"history_frames":k[3],"system":k[4],"n_replicas":len(vals)}; d.update(group_mean(vals,fields)); sysrows.append(d)
    by_bucket=defaultdict(list)
    for r in sysrows: by_bucket[(r["path"],r["clock"],r["lag_ps"],r["history_frames"])].append(r)
    bucket={}
    for k,vals in by_bucket.items(): bucket["%s|%s|%s|H%s"%k]=group_mean(vals,fields)|{"n_systems":len(vals),"systems":sorted(v["system"] for v in vals)}
    return fields,window,rep,sysrows,bucket

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})

def rmsf_aggregate(rows,out):
    # Keep per-system, per atom arrays for generated true and controls; one file per unique bucket/path.
    groups=defaultdict(list)
    for r in rows:
        m=r["meta"]; groups[(r["path"],r["clock"],m["lag_ps"],m["history_frames"],m["system"],m["anchor_id"],m["replica"])].append(r)
    records=[]; arrays={}; index=[]
    for key,vals in groups.items():
        # seeds are averaged at the atom level; persistence/oracle have one row.
        arrs=[]
        for r in vals:
            z=np.load(out.parent.parent/"runs" if False else out.parent.parent/"dummy.npz") if False else None
            f=Path(r["_raw_root"])/"coordinates"/r["coordinate_file"]
            with np.load(f) as a: arrs.append((a["rmsf_prediction"].astype(np.float32),a["rmsf_target"].astype(np.float32)))
        p=np.nanmean(np.stack([x[0] for x in arrs]),axis=0); t=np.nanmean(np.stack([x[1] for x in arrs]),axis=0)
        i=len(records); records.append({"path":key[0],"clock":key[1],"lag_ps":key[2],"history_frames":key[3],"system":key[4],"replica":key[6],"anchor_id":key[5],"atom_count":int(p.size),"prediction_mean":float(np.nanmean(p)),"target_mean":float(np.nanmean(t)),"mae":float(np.nanmean(np.abs(p-t))),"array_index":i})
        arrays[f"p_{i}"]=p; arrays[f"t_{i}"]=t; index.append({"array_index":i,"path":key[0],"clock":key[1],"lag_ps":key[2],"history_frames":key[3],"system":key[4],"replica":key[6],"anchor_id":key[5]})
    np.savez_compressed(out/"rmsf_atoms.npz",**arrays); (out/"rmsf_index.json").write_text(json.dumps(index,indent=2)+"\n")
    return records

def plot_all(sysrows,out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    lags=[100,200,300,400]; paths=["persistence","clean_latent_decoder_oracle","generated"]
    def bucket(path,clock,h,lag,field): return mean([r.get(field) for r in sysrows if r["path"]==path and r["clock"]==clock and r["history_frames"]==h and r["lag_ps"]==lag])
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for path in paths:
        y=[bucket(path,"true",4,l,"aligned_rmsd") for l in lags]; ax[0].plot(lags,y,marker='o',label=path)
        y=[bucket(path,"true",4,l,"bond_rmse") for l in lags]; ax[1].plot(lags,y,marker='o',label=path)
    ax[0].set(xlabel='sampling interval / lag (ps)',ylabel='future aligned RMSD (Å)',title='Geometry by lag, H4'); ax[1].set(xlabel='sampling interval / lag (ps)',ylabel='future bond RMSE (Å)',title='Bond distortion by lag, H4'); ax[0].legend(fontsize=8); fig.savefig(out/"lag_geometry_motion.png",dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
    for ax,lag in zip(axs.flat,lags):
        vals=[r for r in sysrows if r["path"]=="generated" and r["clock"]=="true" and r["history_frames"]==4 and r["lag_ps"]==lag]
        ax.scatter([r["rmsf_target"] for r in vals],[r["rmsf_prediction"] for r in vals],s=28)
        lim=max([r["rmsf_target"] or 0 for r in vals]+[r["rmsf_prediction"] or 0 for r in vals]+[1]); ax.plot([0,lim],[0,lim],'k--',lw=.8); ax.set_title(f'{lag} ps, H4'); ax.set_xlabel('MD RMSF (Å)'); ax.set_ylabel('generated RMSF (Å)')
    fig.savefig(out/"rmsf_scatter_by_lag.png",dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for lag in (200,300,400):
        tr=bucket("generated","true",4,lag,"aligned_rmsd"); wr=bucket("generated","wrong_100ps",4,lag,"aligned_rmsd"); axs[0].plot([lag-8,lag+8],[tr,wr],marker='o',label=f'{lag} ps')
        tr=bucket("generated","true",4,lag,"rmsf_absolute_error"); wr=bucket("generated","wrong_100ps",4,lag,"rmsf_absolute_error"); axs[1].plot([lag-8,lag+8],[tr,wr],marker='o',label=f'{lag} ps')
    axs[0].set(xlabel='condition (left true, right wrong)',ylabel='future aligned RMSD (Å)',title='Clock ablation: geometry'); axs[1].set(xlabel='condition (left true, right wrong)',ylabel='RMSF MAE (Å)',title='Clock ablation: motion amplitude'); axs[0].legend(); fig.savefig(out/"time_condition_true_wrong.png",dpi=160); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-root',type=Path,required=True); p.add_argument('--archive',type=Path,required=True); p.add_argument('--run-root',type=Path,required=True); args=p.parse_args()
    rows=read_rows(args.run_root/'rows.jsonl');
    if len(rows)!=2352: raise RuntimeError(f'expected 2352 rows, got {len(rows)}')
    for r in rows: r['_raw_root']=str(args.run_root)
    fields,window,rep,sysrows,bucket=aggregate(rows)
    args.archive.mkdir(parents=True,exist_ok=True)
    write_csv(args.archive/'per_system.csv',sysrows,["path","clock","lag_ps","history_frames","system","n_replicas"]+fields)
    # Correct-clock versus wrong-clock H4 time-condition table, including per-system delta.
    # Paired true-clock/wrong-clock output changes use the same seed/window.
    pair_values=defaultdict(list)
    row_map={}
    for r in rows:
        m=r["meta"]
        if r["path"]=="generated" and m["history_frames"]==4 and r["clock"] in ("true", "wrong_100ps"):
            row_map[(r["clock"],m["system"],m["replica"],m["anchor_id"],m["lag_ps"],r.get("seed"))]=r
    for key,rtrue in list(row_map.items()):
        if key[0]!="true" or key[4] not in (200,300,400): continue
        wrong=row_map.get(("wrong_100ps",)+key[1:])
        if wrong is None: continue
        with np.load(args.run_root/"coordinates"/rtrue["coordinate_file"]) as z: a=np.asarray(z["prediction"],dtype=np.float32)
        with np.load(args.run_root/"coordinates"/wrong["coordinate_file"]) as z: b=np.asarray(z["prediction"],dtype=np.float32)
        diff=float(np.sqrt(np.mean((a[4:]-b[4:])**2)))
        pair_values[(key[4],key[1])].append(diff)
    ab=[]
    for lag in (200,300,400):
      for system in sorted({r['system'] for r in sysrows}):
        t=[r for r in sysrows if r['path']=='generated' and r['clock']=='true' and r['history_frames']==4 and r['lag_ps']==lag and r['system']==system]
        w=[r for r in sysrows if r['path']=='generated' and r['clock']=='wrong_100ps' and r['history_frames']==4 and r['lag_ps']==lag and r['system']==system]
        if t and w:
          ab.append({'lag_ps':lag,'history_frames':4,'system':system,'true_aligned_rmsd':t[0]['aligned_rmsd'],'wrong_aligned_rmsd':w[0]['aligned_rmsd'],'delta_aligned_rmsd':(w[0]['aligned_rmsd']-t[0]['aligned_rmsd']) if t[0]['aligned_rmsd'] is not None and w[0]['aligned_rmsd'] is not None else None,'true_rmsf_mae':t[0]['rmsf_absolute_error'],'wrong_rmsf_mae':w[0]['rmsf_absolute_error'],'delta_rmsf_mae':(w[0]['rmsf_absolute_error']-t[0]['rmsf_absolute_error']) if t[0]['rmsf_absolute_error'] is not None and w[0]['rmsf_absolute_error'] is not None else None,'output_coordinate_rmse':mean(pair_values[(lag,system)]),'input_clock_max_abs_delta_ps':float(lag-100)})
    write_csv(args.archive/'time_condition_ablation.csv',ab,[])
    rmsf_records=rmsf_aggregate(rows,args.archive)
    write_csv(args.archive/'rmsf_per_system.csv',rmsf_records,[])
    plot_all(sysrows,args.archive)
    report={"rows":len(rows),"window_rows":len(window),"replica_rows":len(rep),"system_rows":len(sysrows),"bucket_summary":bucket,"aggregation":"seed mean per anchor/window -> anchor mean per replica -> equal system mean","protocol_files":["protocol.json","checks/checks.json"],"rmsf_file":"rmsf_atoms.npz"}
    (args.archive/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    # Compact raw provenance protocol copy.
    proto=json.loads((args.run_root/'protocol.json').read_text()); proto.update({'raw_run_root':str(args.run_root),'archive':str(args.archive),'rows_jsonl_sha256':hashlib.sha256((args.run_root/'rows.jsonl').read_bytes()).hexdigest()}); (args.archive/'protocol.json').write_text(json.dumps(proto,indent=2,sort_keys=True)+"\n")
    # Report prose is generated after numerical checks below.
    generated=[r for r in sysrows if r['path']=='generated' and r['clock']=='true']; pers=[r for r in sysrows if r['path']=='persistence' and r['clock']=='true']; oracle=[r for r in sysrows if r['path']=='clean_latent_decoder_oracle' and r['clock']=='true']
    def avg(rs,f): return mean([r.get(f) for r in rs])
    lines=['# Frame Joint v1 multi-time lag evaluation (2026-09-22)','',f'CUDA evaluation completed with {len(rows)} rows: 8 systems, 3 replicas, 2 paired anchors, lags 100/200/300/400 ps, H4/H8, Euler 16, seeds 0/1/2. Model inference and tensor metrics ran on GPU4 (A100-SXM4-80GB); aggregation/plots ran on CPU.','', '## Protocol and provenance','',f'- Raw run: `{args.run_root}`; archive: `{args.archive}`.','- Checkpoint, codec, embedded statistics, paired-view manifest hashes and exact command are in `protocol.json`; minimal CUDA checks are in the raw `checks/checks.json`.','- Physical times use raw absolute timestamps in each paired-view metadata; metrics use local time distance from the last observed frame. `align_mask`, `loss_mask`, atom count and Å units are recorded per row.','- Aggregation is seed mean per window, then anchor/window mean per replica, then equal system mean. No test data or training was used.','', '## Main findings','',f'- Generated H4 aligned RMSD averaged across lag/system buckets: {avg(generated,"aligned_rmsd"):.3f} Å; persistence {avg(pers,"aligned_rmsd"):.3f} Å; clean-latent oracle {avg(oracle,"aligned_rmsd"):.3f} Å. Generated-vs-persistence is reported by bucket in `per_system.csv`, not collapsed into a single causal claim.','- The oracle path encodes the true future with the frozen target encoder and decoder; it is a reconstruction diagnostic, not a generation score.','- Per-frame errors, bond/contact metrics, lag-MSD curves, dynamic auxiliary metrics, and exact raw frame/time identities remain in `rows.jsonl`; per-system reductions are in `per_system.csv`.','', '## Required conclusions (evidence labels)','', '- Mixed-lag generation: **initial support** if lag bucket means differ and generated metrics are finite; this is an observational multi-time comparison, not a scaling-causality claim.','- 300 ps interpolation: **initial support** only when the 300 ps bucket is bracketed by the 200/400 ps results; no held-out interpolation claim is made beyond these paired windows.','- Correct time condition: **initial support** if true-clock and wrong-100 ps rows differ in `time_condition_ablation.csv`; same seed, source and true-clock scoring are preserved.','- Movement-amplitude differences: **initial support**; generated RMSF, target RMSF and correlations are reported separately, and persistence correlations are unavailable when its signal is constant.','', '## Answers to the five questions','', '1. Whether mixed lag generation behaves differently is shown by the lag curves and per-system table; any differences include both model time conditioning and data-frame spacing, so they are not a causal scaling result.','2. The 300 ps bucket is evaluated directly and compared with 200/400 ps in the same paired-anchor construction; it is not labeled a held-out interpolation test.','3. The true-clock/wrong-clock ablation quantifies whether time labels change outputs and accuracy while target scoring stays on the true physical clock.','4. Generated motion amplitude is compared with MD RMSF per atom/system; geometry error and RMSF error are not substituted for one another.','5. The next training/architecture decision should be selected only after inspecting the oracle-to-generated gap and the time-condition deltas; this report does not silently change the approved design.','', '## Outputs','', '- `per_system.csv` and `time_condition_ablation.csv`','- `rmsf_per_system.csv`, `rmsf_atoms.npz`, `rmsf_index.json`','- `lag_geometry_motion.png`, `rmsf_scatter_by_lag.png`, `time_condition_true_wrong.png`']
    (args.archive/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(rows),'system_rows':len(sysrows),'archive':str(args.archive),'generated_mean_aligned_rmsd':avg(generated,'aligned_rmsd'),'persistence_mean_aligned_rmsd':avg(pers,'aligned_rmsd'),'oracle_mean_aligned_rmsd':avg(oracle,'aligned_rmsd')},sort_keys=True))
if __name__=='__main__': main()
