import json,csv,hashlib
from pathlib import Path
root=Path('/workspace/molvid-recon-learned-memory-v1'); b=root/'agent/gm_motion_memory_v3/B'; runs=root/'runs/learned_memory_v1'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
pilot=runs/'cuda_pilot_r03'; prof=runs/'cuda_pilot_r03_profile'
m=read(pilot/'train_manifest.json'); m.update(budget_gpu_hours=48,pilot_budget_gpu_hours=8,cuda_smoke='6 passed',formal_M_short='not_run_waiting_A_evaluated_recipe',formal_M_long='not_run_waiting_A_evaluated_recipe',formal_M_index='not_default',A_evaluated_recipe=None,CPU_model_fallback=False,data_paired_checks=198,training_code_commit='52617d23f79f743fd585e9fe9e143ca3e2768199',execution_code_commit='3a484c88252ab8a6b953dd3616d8b130e183bd2d',execution_checks=read(runs/'execution_r01/checks.json'))
ledgers=[dict(path=str(p.relative_to(root)),**read(p)) for p in runs.glob('*/cost_ledger.json')]
m['gpu_cost_ledgers']=ledgers;m['gpu_hours_metered']=sum(x['gpu_hours_including_setup_checks'] for x in ledgers)
m['smoke_and_small_probes_allowance_gpu_hours']=0.05;m['accounted_gpu_hours_with_allowance']=m['gpu_hours_metered']+0.05
if (prof/'train_manifest.json').exists():m['size_profile']=read(prof/'train_manifest.json')
write(b/'train_manifest.json',m);write(b/'model_card.json',read(pilot/'model_card.json'));write(b/'loss_manifest.json',read(pilot/'loss_manifest.json'))
write(b/'selection.json',dict(status='not_evaluated',selected_arm=None,reason='CUDA pilot is numerical/throughput evidence; formal paired comparison waits for A evaluated recipe',AF3_integration_validated=False,long_rollout_validated=False))
runtime=read(root/'agent/gm_motion_memory_v3/runtime_paths.json');runtime.update(gpu_uuid='GPU-b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0',gpu_index=0,gpu_allocation_status='user_authorized_shared; A resources assignments unchanged',common_commit='5a745960f673d5397b883c2b129f39cf57593856',common_local_cherry_pick='0a556d7abc339f50f30472528a6e512b42e8858a',common_status='fixed common release absorbed; not evaluated recipe',reviewed_code_commit='3a484c88252ab8a6b953dd3616d8b130e183bd2d');write(root/'agent/gm_motion_memory_v3/runtime_paths.json',runtime)
rows=[json.loads(s) for s in (pilot/'train_metrics.jsonl').read_text().splitlines()]
# Current pilot cycles banks in preserved data-card order; don't infer motion scores from losses.
with (b/'per_system.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['system','role','atoms','raw_trajectories_audited','past_ps','future_ps','generated_motion_status'])
 for name,role,n,reps in [('1bkp_A','pilot',2307,3),('1by2_A','pilot',886,3),('1cpq_A','pilot',920,3),('3gwi_A','size_profile',1374,1),('4yal_A','size_profile',4975,1)]:w.writerow([name,role,n,reps,12800,4800,'not_evaluated_waiting_A_recipe'])
ident=read(b/'artifact_identities.json');ident['cuda_evidence']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pilot/'code_identity.json',pilot/'train_manifest.json',pilot/'teacher_checks.json',pilot/'train_metrics.jsonl',pilot/'load_report.json',runs/'execution_r01/checks.json']};write(b/'artifact_identities.json',ident)
print(json.dumps({'metered_gpu_hours':m['gpu_hours_metered'],'with_allowance':m['accounted_gpu_hours_with_allowance'],'profile_complete':(prof/'train_manifest.json').exists()}))
if (prof/'train_manifest.json').exists():
    profile_rows=[json.loads(s) for s in (prof/'train_metrics.jsonl').read_text().splitlines()]
    config=read(b/'pilot_config.json'); entries=[(entry,span,dt,lag) for entry in config['profile_banks'] for span,dt,lag in ((800,100,100),(3200,100,200),(12800,100,400),(12800,400,100))];entries.append((config['profile_banks'][0],800,100,100));entries*=2
    summary=[]
    for i,row in enumerate(profile_rows):
        trace=read(prof/f'profile_{i:03d}.json')
        wall={name:sum(e['dur'] for e in trace['traceEvents'] if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')==name)/1000 for name in ['frame_joint.backward','B.teacher_and_transfer']}
        entry,span,dt,lag=entries[i]
        summary.append(dict(step=row['step'],bank=entry['path'],batch_size=row['batch_size'],profile_pass=row['profile_pass'],mixed_history_dt=row['mixed_history_dt'],history_span_ps=span,observed_dt_ps=dt,query_lag_ps=lag,peak_allocated_GiB=row['peak_allocated_bytes']/2**30,step_seconds=row['step_seconds'],module_cuda_event_ms=row['module_ms'],trace_cpu_wall_ms=wall))
    write(b/'profile_summary.json',dict(source_commit='3a484c88252ab8a6b953dd3616d8b130e183bd2d',scope='18 training updates for throughput only; no quality evaluation',invalid_fields='Original train_metrics.profile_ranges key_averages cpu_us/device_us must not be used as timing evidence',timing_semantics='module_cuda_event_ms: stream event elapsed, inclusive overlapping ranges; trace_cpu_wall_ms: CPU annotation wall duration, includes launches/waits, not kernel-summed GPU time; step_seconds excludes trace serialization',conditions=summary))
    print('profile summary:',len(summary),'rows; peak GiB',max(r['peak_allocated_GiB'] for r in summary))
