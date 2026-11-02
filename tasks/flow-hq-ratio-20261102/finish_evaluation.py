"""Finite CPU import -> common metrics/G/D/seams/viewer for existing HQ outputs."""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
import prepare_evaluation as scope
TASK=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=scope.RUN);p.add_argument('--output-root',type=Path)
    p.add_argument('--phases',nargs='+',choices=('phase1','phase2'),default=['phase1','phase2']);p.add_argument('--cases',nargs='+',choices=scope.base.CASES,default=list(scope.base.CASES))
    p.add_argument('--reuse-evaluation',type=Path);p.add_argument('--settings',nargs='+',choices=scope.SETTINGS);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();scope.cpu_only()
    out=(a.output_root or a.run_root/'evaluation').resolve();out.mkdir(parents=True,exist_ok=True);receipt=out/'cpu_execution.json'
    if receipt.exists():raise FileExistsError('Preserve existing CPU receipt; choose a separate output for a partial preview')
    state=dict(status='running',pid=os.getpid(),stages=[],phases=a.phases,output_root=str(out),GPU_seconds=0,model_calls=0)
    def save():scope.write_json(receipt,state)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    def step(name,argv):
        log=out/(name+'.log');start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        with log.open('x') as f:result=subprocess.run([str(x) for x in argv],cwd=scope.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        r=dict(stage=name,exit_code=result.returncode,wall_seconds=time.monotonic()-start,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,log=str(log))
        state['stages'].append(r);save();print(json.dumps(r),flush=True)
    save();cmd=[sys.executable,'-u',TASK/'prepare_evaluation.py','--import-generated','--run-root',a.run_root,'--output-root',out,'--phases',*a.phases,'--cases',*a.cases]
    if a.allow_partial:cmd+=['--allow-partial']
    if a.settings:cmd+=['--settings',*a.settings]
    step('prepare',cmd)
    if (out/'imported/panel_index.json').exists():
        cmd=[sys.executable,'-u',TASK/'compare_results.py','--output-root',out,'--export-viewer']
        if a.reuse_evaluation:cmd+=['--reuse-evaluation',a.reuse_evaluation]
        step('compare',cmd)
    summary=out/'comparison/comparison_summary.json'
    if summary.exists():
        r=scope.read(summary);state.update({k:r[k] for k in ('status','full_study_complete','available_new_draws','expected_new_draws','errors','missing')})
        state['comparison_summary']=str(summary);state['reused_prior_phase_draws']=r.get('reused_prior_phase_draws',0);state['total_covered_draws']=r.get('total_covered_draws',r['available_new_draws'])
    else:state.update(status='failed',full_study_complete=False,errors=[dict(reason='No comparison summary')])
    if any(s['exit_code'] for s in state['stages']) and state['status']=='completed':state['status']='partial'
    state.update(CPU_wall_seconds=sum(s['wall_seconds'] for s in state['stages']),CPU_seconds=sum(s['CPU_seconds'] for s in state['stages']),
        viewer=dict(index=str(out/'viewer/index.html'),exists=(out/'viewer/index.html').is_file(),HTML_files=len(list((out/'viewer').glob('*.html'))),NPZ_files=len(list((out/'viewer').rglob('*.npz')))))
    save();print(json.dumps({k:state.get(k) for k in ('status','full_study_complete','available_new_draws','expected_new_draws','CPU_wall_seconds','CPU_seconds')}),flush=True)
    if state['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
