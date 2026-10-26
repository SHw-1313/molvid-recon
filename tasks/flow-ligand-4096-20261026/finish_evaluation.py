"""Finite CPU tail; wait for existing lanes, never submit or retry GPU work."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from prepare_evaluation import ROOT,RUN,MAIN,ENGINE,ARMS,MODES,CASES,ORDER_SOURCES,read,cpu_only

TERMINAL={'completed','completed_with_failures','failed_or_budget_stopped','failed'}
TASK=Path(__file__).resolve().parent

def execution(run,arms):
    result={}
    for arm in arms:
        try:result[arm]=read(run/f'{arm}_execution.json')
        except (FileNotFoundError,json.JSONDecodeError):result[arm]={}
    return result

def ready(run,step,modes,cases,seeds,arms):
    for arm in arms:
        base=run/'generation'/arm if step==4096 else run/'generation_probes'/arm/f'step{step}'
        for mode in modes:
            for case in cases:
                try:
                    p=read(base/mode/case/'protocol.json')
                    if p.get('status')!='completed' or any(str(s) not in p.get('draws',{}) for s in seeds):return False
                except (FileNotFoundError,json.JSONDecodeError):return False
    return True

def run_cpu(a):
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True);receipt=out/'cpu_execution.json'
    if receipt.exists():raise FileExistsError('Preserve prior CPU attempt; choose a new --output-root')
    state=dict(status='running',run_root=str(a.run_root),output_root=str(out),checkpoint_step=a.checkpoint_step,
        requested_modes=a.modes,requested_arms=a.arms,stages=[],GPU_seconds=0,model_calls=0)
    def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(MAIN)+':/workspace/molvid_temporal_eval_v1')
    def step(label,cmd):
        start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);log=out/(label+'.log')
        with log.open('x') as f:r=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        item=dict(stage=label,command=list(map(str,cmd)),exit_code=r.returncode,wall_seconds=time.monotonic()-start,
            CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,log=str(log))
        state['stages'].append(item);save();print(json.dumps(item),flush=True)
        return r.returncode
    save()
    selected=['--run-root',a.run_root,'--output-root',out,'--checkpoint-step',a.checkpoint_step,
        '--modes',*a.modes,'--arms',*a.arms,'--cases',*a.cases,'--seeds',*a.seeds]
    extra=['--allow-partial'] if a.allow_partial else []
    if a.reuse_evaluation:selected+=['--reuse-evaluation',a.reuse_evaluation]
    step('prepare',[sys.executable,TASK/'prepare_evaluation.py',*selected,*extra])
    if (out/'common_input/manifest.json').exists():
        step('common_score',[sys.executable,MAIN/'tools/evaluate_common_panel.py','--manifest',out/'common_input/manifest.json',
            '--output',out/'common_scores','--engine-root',ENGINE,'--no-local-geometry'])
    if (out/'common_scores/panel_index.json').exists():
        step('merge_geometry',[sys.executable,TASK/'compare_results.py','--output-root',out,'--merge-panel'])
    if (out/'comparison_inputs/panel_index.json').exists():
        imports=read(out/'common_input/import_details.json');reuse=list(ORDER_SOURCES)
        reuse.extend(Path(p)/'temporal_order/sidecar' for p in sorted({r['evaluation'] for r in imports.get('cached_new_draws',[])}))
        reuse_args=[v for path in reuse for v in ('--reuse-sidecar',path)]
        step('temporal_order',[sys.executable,MAIN/'tasks/main-integration-20261016/evaluate_temporal_order.py',
            '--panel-index',out/'comparison_inputs/panel_index.json','--manifest',out/'common_input/manifest.json',
            '--models',*imports['arm_specs'],'--output',out/'temporal_order/sidecar',*reuse_args])
    if (out/'temporal_order/sidecar/draw_metrics.json').exists():
        step('compare_viewer',[sys.executable,TASK/'compare_results.py','--output-root',out,'--export-viewer',*extra])
        step('distance',[sys.executable,TASK/'compare_results.py','--output-root',out,'--distance'])
    if not a.skip_static:
        manifest_path=a.run_root/'static_probe/inputs/manifest.json';static_summaries=[]
        if manifest_path.exists():
            original=read(manifest_path);static_root=out/'static_probe';static_root.mkdir(exist_ok=False)
            for arm in a.arms:
                for count in (1024,2048,4096):
                    stage=f'{arm}{count}'
                    if stage not in original['stages']:continue
                    folder=a.run_root/'static_probe/generated'/stage
                    try:protocol=read(folder/'protocol.json')
                    except (FileNotFoundError,json.JSONDecodeError):protocol={}
                    expected=[s['id'] for s in original['snapshots']] if count==4096 else ['atlas_6e5y_A_f000','4J8T_f000']
                    requested=protocol.get('requested_snapshots',expected)
                    if set(requested)!=set(expected):
                        state.setdefault('static_errors',[]).append(dict(stage=stage,reason='Actual requested snapshots differ from approved 2/12 scope',actual=requested,expected=expected));continue
                    subset=copy.deepcopy(original);subset['snapshots']=[s for s in original['snapshots'] if s['id'] in requested]
                    local=static_root/(stage+'_inputs.json');local.write_text(json.dumps(subset,indent=2)+'\n')
                    target=static_root/stage
                    step('static_'+stage,[sys.executable,TASK/'static_probe.py','score','--stages',stage,'--manifest',local,
                        '--generation-root',a.run_root/'static_probe/generated','--output',target,'--allow-partial'])
                    if (target/'summary.json').exists():static_summaries.append(read(target/'summary.json'))
            static=dict(status='completed' if len(static_summaries)==3*len(a.arms) and all(s['status']=='completed' for s in static_summaries) else 'incomplete',
                source_manifest=str(manifest_path),source_score_files=[str(static_root/f'{arm}{step}/summary.json') for arm in a.arms for step in (1024,2048,4096)],
                rows=[r for s in static_summaries for r in s['rows']],per_dataset=[r for s in static_summaries for r in s['per_dataset']],
                completed_snapshots=sum(s['completed_snapshots'] for s in static_summaries),requested_snapshots=sum(s['requested_snapshots'] for s in static_summaries),
                errors=[e for s in static_summaries for e in s['errors']],
                aggregation='Q1 restoration separate from dynamic results. Each checkpoint retains actual2 or12 snapshots; no fabricated missing snapshots. Same original payload/corruption/source; original score definition.')
            (static_root/'summary.json').write_text(json.dumps(static,indent=2,allow_nan=False)+'\n');state['static_status']=static['status']
        else:state.update(static_status='NA',static_reason='Prepared static manifest is absent')
    summary_path=out/'comparison/comparison_summary.json'
    if summary_path.exists():
        s=read(summary_path);state.update(comparison_status=s['completion_status'],full_study_complete=s['full_study_complete'],
            expected_new_draws=s['expected_new_draws'],available_new_draws=s['available_new_draws'],comparison_errors=s['errors'],summary=str(summary_path))
        viewer=out/'comparison_viewer';state['viewer']=dict(index=str(viewer/'index.html'),HTML_files=len(list(viewer.glob('*.html'))),NPZ_files=len(list(viewer.rglob('*.npz'))),exists=(viewer/'index.html').is_file())
    else:state.update(comparison_status='failed',full_study_complete=False)
    additional_errors=[]
    for path in (out/'temporal_order/sidecar/summary.json',out/'distance/summary.json'):
        if path.exists():additional_errors+=read(path).get('errors',[])
        else:additional_errors.append(dict(reason='Missing required G/D sidecar',source=str(path)))
    state.update(sidecar_errors=additional_errors,CPU_wall_seconds=sum(r['wall_seconds'] for r in state['stages']),CPU_seconds=sum(r['CPU_seconds'] for r in state['stages']))
    static_ok=a.skip_static or (state.get('static_status')=='completed' and not state.get('static_errors'))
    state['status']='completed' if state['comparison_status']=='complete' and not additional_errors and static_ok and all(r['exit_code']==0 for r in state['stages']) else 'partial'
    if state['comparison_status']=='failed':state['status']='failed'
    save();print(json.dumps(dict(status=state['status'],receipt=str(receipt),new_draws=state.get('available_new_draws')),ensure_ascii=False),flush=True)
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES));p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));p.add_argument('--seeds',nargs='+',type=int,choices=[17,29],default=[17,29])
    p.add_argument('--checkpoint-step',type=int,choices=[1024,2048,4096],default=4096);p.add_argument('--reuse-evaluation',type=Path)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--skip-static',action='store_true');p.add_argument('--wait',action='store_true')
    p.add_argument('--max-wait-seconds',type=float,default=259200);p.add_argument('--no-early-evaluation',action='store_true')
    a=p.parse_args();cpu_only();a.run_root=a.run_root.resolve();started=time.monotonic();attempted=set()
    while a.wait:
        states=execution(a.run_root,a.arms);ended=all(s.get('status') in TERMINAL for s in states.values())
        # Three fixed authorized milestones, not an extensible scheduler. Raw remains owned by GPU lanes.
        if a.checkpoint_step==4096 and not a.no_early_evaluation:
            for count in (1024,2048,4096):
                cases=['atlas_6e5y_A','4J8T'] if count!=4096 else list(a.cases);seeds=[17] if count!=4096 else list(a.seeds)
                dest=a.run_root/(f'evaluation_probe{count}' if count!=4096 else 'evaluation_joint4096')
                if count in attempted or dest.exists():continue
                if ready(a.run_root,count,['joint'],cases,seeds,a.arms):
                    probe=copy.copy(a);probe.output_root=dest;probe.checkpoint_step=count;probe.cases=cases;probe.seeds=seeds;probe.modes=['joint'];probe.skip_static=True;probe.allow_partial=True;probe.reuse_evaluation=None
                    attempted.add(count)
                    try:run_cpu(probe)
                    except Exception as exc:
                        dest.mkdir(parents=True,exist_ok=True)
                        (dest/'tail_failure.json').write_text(json.dumps(dict(status='failed',reason=repr(exc),checkpoint_step=count),indent=2)+'\n')
                        print(json.dumps(dict(stage='early_evaluation',checkpoint_step=count,reason=repr(exc))),flush=True)
        if ended:break
        if a.checkpoint_step!=4096 and ready(a.run_root,a.checkpoint_step,a.modes,a.cases,a.seeds,a.arms):break
        if time.monotonic()-started>=a.max_wait_seconds:break
        time.sleep(30)
    a.output_root=a.output_root or a.run_root/('evaluation' if a.checkpoint_step==4096 else f'evaluation_probe{a.checkpoint_step}')
    if a.checkpoint_step!=4096:a.skip_static=True
    if a.wait:a.allow_partial=True  # A failed/partial lane never blocks valid outputs of the other lane.
    candidate=a.run_root/'evaluation_joint4096'
    if not a.reuse_evaluation and a.checkpoint_step==4096 and set(a.modes)==set(MODES) and (candidate/'distance/results.json').is_file():a.reuse_evaluation=candidate
    result=run_cpu(a)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
