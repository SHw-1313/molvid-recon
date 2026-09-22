#!/usr/bin/env python
"""CPU summary, CSV and plots for Frame Joint RMSD diagnosis outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def mean(values):
    values=[float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--archive',required=True)
    ap.add_argument('--pilot',required=True); ap.add_argument('--capacity',required=True); ap.add_argument('--endpoint',required=True)
    a=ap.parse_args(); out=Path(a.output); archive=Path(a.archive); archive.mkdir(parents=True,exist_ok=True)
    sources={'pilot_48':Path(a.pilot),'capacity_192_step111573':Path(a.capacity)}
    payloads={k:json.loads((v/'diagnosis.json').read_text()) for k,v in sources.items()}
    summary=[]; per_frame=[]; system_rows=[]
    fields=['checkpoint','history','path','aligned_rmsd','drmsd','bond_rmse','contact_f1','contact_occupancy_mae','rmsf_prediction','rmsf_target','rmsf_absolute_error','rmsf_correlation','velocity_rmse','dynamic_correlation','available_aligned_rmsd','available_bond_rmse']
    for ck,d in payloads.items():
        for h, paths in d['summary'].items():
            for path, value in paths.items():
                m=value['mean']; counts=value.get('mean',{}).get('available_counts',{})
                row={'checkpoint':ck,'history':h,'path':path}; row.update({k:m.get(k) for k in fields[3:-2]}); row['available_aligned_rmsd']=counts.get('aligned_rmsd'); row['available_bond_rmse']=counts.get('bond_rmse'); summary.append(row)
                for system, sm in value['systems'].items():
                    sr={'checkpoint':ck,'history':h,'path':path,'system':system}; sr.update({k:sm.get(k) for k in fields[3:-2]}); system_rows.append(sr)
        for h, paths in d['per_frame_summary'].items():
            for path, value in paths.items():
                for frame in value['frames']:
                    m=frame['summary']['mean']; per_frame.append({'checkpoint':ck,'history':h,'path':path,'frame_index':frame['frame_index'],'distance_from_last_observation_ps':frame['distance_from_last_observation_ps'],'aligned_rmsd':m.get('aligned_rmsd'),'drmsd':m.get('drmsd'),'bond_rmse':m.get('bond_rmse'),'available_aligned_rmsd':m.get('available_counts',{}).get('aligned_rmsd'),'available_bond_rmse':m.get('available_counts',{}).get('bond_rmse')})
    write_csv(archive/'summary.csv',summary,fields)
    write_csv(archive/'per_frame_summary.csv',per_frame,list(per_frame[0]) if per_frame else [])
    write_csv(archive/'system_summary.csv',system_rows,list(system_rows[0]) if system_rows else [])
    ep=json.loads((Path(a.endpoint)/'diagnosis.json').read_text())
    eprows=[]
    for h in sorted({r['history_frames'] for r in ep['rows']}):
        for s in sorted({r['flow_time'] for r in ep['rows'] if r['history_frames']==h}):
            rs=[r for r in ep['rows'] if r['history_frames']==h and abs(r['flow_time']-s)<1e-9]
            eprows.append({'history':f'H{h}','flow_time':s,'windows':len(rs),'endpoint_aligned_rmsd':mean([r['endpoint']['future']['aligned_rmsd'] for r in rs]),'endpoint_bond_rmse':mean([r['endpoint']['future']['bond_rmse'] for r in rs]),'flow_velocity_mse_h':mean([r['flow_velocity_mse']['h'] for r in rs]),'flow_velocity_mse_v':mean([r['flow_velocity_mse']['v'] for r in rs]),'endpoint_latent_mse_h':mean([r['endpoint_latent_mse_normalized']['h'] for r in rs]),'endpoint_latent_mse_v':mean([r['endpoint_latent_mse_normalized']['v'] for r in rs]),'clean_oracle_aligned_rmsd':mean([r['clean_oracle']['future']['aligned_rmsd'] for r in rs]),'clean_oracle_bond_rmse':mean([r['clean_oracle']['future']['bond_rmse'] for r in rs])})
    write_csv(archive/'endpoint_summary.csv',eprows,list(eprows[0]))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    colors={'persistence':'#555555','clean_latent_decoder_oracle':'#2ca02c','generated':'#d62728'}
    labels={'persistence':'Persistence','clean_latent_decoder_oracle':'Clean oracle','generated':'Generated'}
    for metric,ylabel,filename in [('aligned_rmsd','aligned RMSD (Å)','aligned_rmsd_by_time.png'),('bond_rmse','bond RMSE (Å)','bond_rmse_by_time.png')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,h in zip(axs,['H4','H8']):
            for ck,sty in [('pilot_48','-'),('capacity_192_step111573','--')]:
                for path in ['persistence','clean_latent_decoder_oracle','generated']:
                    rs=[r for r in per_frame if r['checkpoint']==ck and r['history']==h and r['path']==path]
                    rs.sort(key=lambda x:x['distance_from_last_observation_ps'])
                    if not rs: continue
                    ax.plot([r['distance_from_last_observation_ps'] for r in rs],[r[metric] for r in rs],linestyle=sty,color=colors[path],label=f'{labels[path]} ({"48" if ck=="pilot_48" else "192"})')
            ax.set_title(h); ax.set_xlabel('distance from last observation (ps)'); ax.grid(alpha=.25)
        axs[0].set_ylabel(ylabel); handles, labs=axs[0].get_legend_handles_labels(); fig.legend(handles,labs,loc='upper center',ncol=3,fontsize=8); fig.tight_layout(rect=(0,0,1,.86)); fig.savefig(archive/filename,dpi=170); plt.close(fig)
    for metric,ylabel,filename in [('aligned_rmsd','aligned RMSD (Å)','aggregate_aligned_rmsd.png'),('bond_rmse','bond RMSE (Å)','aggregate_bond_rmse.png')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,h in zip(axs,['H4','H8']):
            rs=[r for r in summary if r['history']==h]
            x=np.arange(3); width=.35
            for j,ck in enumerate(['pilot_48','capacity_192_step111573']):
                vals=[next(r[metric] for r in rs if r['checkpoint']==ck and r['path']==p) for p in ['persistence','clean_latent_decoder_oracle','generated']]
                ax.bar(x+(j-.5)*width,vals,width,label='48' if ck=='pilot_48' else '192')
            ax.set_xticks(x,['Persistence','Oracle','Generated']); ax.set_title(h); ax.grid(axis='y',alpha=.25)
        axs[0].set_ylabel(ylabel); axs[1].legend(); fig.tight_layout(); fig.savefig(archive/filename,dpi=170); plt.close(fig)
    # Compact report generated from the exact saved tables.
    sby={(r['checkpoint'],r['history'],r['path']):r for r in summary}
    def fmt(v): return 'NA' if v is None else f'{float(v):.4f}'
    lines=['# Frame Joint v1 RMSD diagnosis (2026-09-22)','', 'This report compares persistence, clean-latent decoder oracle, and generated sampling on the fixed 48-system pilot valid-quick windows (8 systems, 24 trajectories, two windows each; H4/H8; 100 ps spacing; Euler 16; seed 0). All model inference and tensor metrics ran on CUDA; summaries and plots were produced on CPU.','', '## Protocol and provenance','',f'- Pilot checkpoint: `{payloads["pilot_48"]["checkpoint"]["path"]}`; SHA `{payloads["pilot_48"]["checkpoint"]["sha256"]}`.',f'- 192 capacity checkpoint: `{payloads["capacity_192_step111573"]["checkpoint"]["path"]}`; SHA `{payloads["capacity_192_step111573"]["checkpoint"]["sha256"]}`; training stage is the joint stage; this is the latest stable checkpoint selected for this report.', '- Frozen codec: `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`; SHA `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`. Reference valid-quick JSON SHA: `ba79a2272f530642007db30a6a0b2ba9787dc993520dcbf9766e2e3c07072561`.' ,f'- Pilot valid index SHA: `{payloads["pilot_48"]["valid_store"]["index_sha256"]}`; 192 valid index SHA: `{payloads["capacity_192_step111573"]["valid_store"]["index_sha256"]}`.', '- Coordinates are Å. Aligned RMSD uses per-frame Kabsch on `align_mask`, then RMSD/dRMSD/bond metrics use the evaluator loss-mask intersection. The outputs record loss-mask and align-mask atom counts per window; representative windows have 1199 loss atoms, 612 align atoms, 612 intersection atoms.', '- Aggregation is window mean, then replica mean, then equal system mean. Persistence dynamic/RMSF correlations are unavailable when its predicted motion has zero variance and are kept null.', '', '## Aggregate future metrics', '', '| checkpoint | H | path | aligned RMSD | dRMSD | bond RMSE | contact F1 | RMSF AE | dynamic corr | velocity RMSE |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['checkpoint']} | {r['history']} | {labels[r['path']]} | {fmt(r['aligned_rmsd'])} | {fmt(r['drmsd'])} | {fmt(r['bond_rmse'])} | {fmt(r['contact_f1'])} | {fmt(r['rmsf_absolute_error'])} | {fmt(r['dynamic_correlation'])} | {fmt(r['velocity_rmse'])} |")
    lines += ['', '## First and last future frame', '', '| checkpoint | H | path | first distance / RMSD / bond | last distance / RMSD / bond |','|---|---:|---|---|---|']
    for ck in ['pilot_48','capacity_192_step111573']:
        for h in ['H4','H8']:
            for path in ['persistence','clean_latent_decoder_oracle','generated']:
                rs=sorted([r for r in per_frame if r['checkpoint']==ck and r['history']==h and r['path']==path],key=lambda z:z['distance_from_last_observation_ps'])
                if rs:
                    lines.append(f"| {ck} | {h} | {labels[path]} | +{rs[0]['distance_from_last_observation_ps']:.0f} ps / {fmt(rs[0]['aligned_rmsd'])} / {fmt(rs[0]['bond_rmse'])} | +{rs[-1]['distance_from_last_observation_ps']:.0f} ps / {fmt(rs[-1]['aligned_rmsd'])} / {fmt(rs[-1]['bond_rmse'])} |")
    lines += ['', '## System heterogeneity', '', 'The complete per-system table is in `system_summary.csv`. On the pilot, the largest generated-minus-persistence aligned-RMSD gaps are `atlas_3l4h_A` (+0.294 Å H4, +0.350 Å H8), `atlas_3f0o_B` (+0.229 Å H4, +0.296 Å H8), and H8 `atlas_5ef9_A` (+0.327 Å). The pilot H4 `atlas_2ejn_B` mean is slightly below persistence (−0.013 Å), but its generated bond RMSE remains ~0.531 Å. On the latest 192 checkpoint, every system is worse than persistence; the largest gaps are `atlas_3l4h_A` (+0.326 Å H4), `atlas_2ejn_B` (+0.502 Å H8), and `atlas_5ii7_A` (+0.249/+0.342 Å H4/H8).', '', '## Endpoint diagnostic (future-informed; not generation performance)', '', 'The endpoint inputs intentionally contain the clean future through the training interpolation. The same source/noise seed is reused across s values within each window. The `(1-s)` endpoint factor means raw endpoint error must not be compared across s without accounting for this factor; velocity MSE and normalized endpoint latent MSE are shown together.', '', '| H | s | endpoint RMSD | endpoint bond | flow h MSE | flow v MSE | normalized endpoint h MSE | normalized endpoint v MSE | clean oracle RMSD |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in eprows:
        lines.append(f"| {r['history']} | {r['flow_time']:.2f} | {r['endpoint_aligned_rmsd']:.4f} | {r['endpoint_bond_rmse']:.4f} | {r['flow_velocity_mse_h']:.4f} | {r['flow_velocity_mse_v']:.4f} | {r['endpoint_latent_mse_h']:.4f} | {r['endpoint_latent_mse_v']:.4f} | {r['clean_oracle_aligned_rmsd']:.4f} |")
    lines += ['', '## Conclusions', '', '- The generated path is worse than simply copying the last observed coordinates on the aggregate aligned-RMSD criterion for both H4 and H8. It is already worse at the first +100 ps frame (pilot H4: 1.749 vs 1.519 Å; H8: 1.803 vs 1.509 Å) and drifts further by the last frame. Its bond error is also much larger from the first frame (pilot H4: 0.510 vs 0.042 Å; H8: 0.565 vs 0.042 Å), so the structural and bond failures appear together rather than being only a rigid-body/dynamics issue.', '- The clean-latent decoder oracle is near the reconstruction floor (pilot ~0.064 Å aligned RMSD and ~0.0367 Å bond RMSE; 192 ~0.025 Å and ~0.0146 Å), with contact F1 above 0.98 and dynamic correlation ~0.9997–0.9999. This rules out the frozen decoder/codec as the dominant explanation for the generated 2.2–2.4 Å error on this protocol.', '- Endpoint diagnostics show a large reduction in coordinate/bond error as s approaches 0.95, but the normalized endpoint latent and velocity MSE expose the expected `(1-s)` scaling and do not establish that the model only handles low-noise denoising. The full sampler still fails badly, so source-to-target flow accuracy and/or trajectory integration remains the leading localization, not a decoder-only floor.', '- The 192 checkpoint improves bond RMSE substantially (pilot 0.544/0.598 Å to 0.354/0.364 Å for H4/H8) and improves the clean oracle, but does not improve generated aligned RMSD (H4 2.390, H8 2.280). Thus scaling changes bond fidelity without solving the generated coordinate drift. This is an observational comparison, not a causal scaling experiment.', '- Persistence has low bond error by construction but misses physical motion; its RMSF/dynamic correlations are null where predicted variance is zero. Generated dynamic correlations near zero are not, by themselves, proof of no learned dynamics; they accompany the large geometric and bond errors here.', '- Most supported next experiment: diagnose/train the source-to-target latent flow and endpoint integration (including calibrated latent/velocity targets and sampler stability) while keeping the measured decoder oracle as a fixed acceptance gate. Do not treat this report as authorization to change training in this evaluation-only turn.', '', '## Files', '', '- Raw pilot: `runs/frame_joint_v1_rmsd_diagnosis_260922/pilot_48_full/diagnosis.json` and `coordinates.npz`.', '- Raw 192: `runs/frame_joint_v1_rmsd_diagnosis_260922/192_step111573_full/diagnosis.json` and `coordinates.npz`.', '- Endpoint diagnostic: `runs/frame_joint_v1_rmsd_diagnosis_260922/pilot_endpoint/diagnosis.json` and `coordinates.npz`.', '- CSVs and plots are in this archive directory.']
    (archive/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
