"""Three-arm Q4 study: frozen H4/MD50 inputs and completed baselines."""
import argparse,copy,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/flow_q4_align_20261030'
OLD=Path('/workspace/MOLVID-FLOW-KERNEL-20261027/tasks/flow-kernel-20261027')
T_ROOT=Path('/workspace/MOLVID-FLOW-TASK-ROUTE-20261029')
T_RUN=T_ROOT/'runs/flow_task_route_20261029';T_EVAL=T_RUN/'evaluation'
spec=importlib.util.spec_from_file_location('q4_existing_inputs',OLD/'prepare_evaluation.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
MAIN,MAIN_RUN,ENGINE,PARENT_EVAL=base.MAIN,base.MAIN_RUN,base.ENGINE,base.PARENT_EVAL
read,write_json,cpu_only,prepare_case,import_file=base.read,base.write_json,base.cpu_only,base.prepare_case,base.import_file
CASES=base.CASES;ARMS=('shared_joint_only','shared_q4_q1','routed_q4_q1');ENDPOINT_ORIGINS=()
OPTIMIZER_UPDATES=dict(zip(ARMS,(256,512,512)))


def prepare(run):
    cpu_only();dest=run/'eval_inputs';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():raise FileExistsError('Preserve fixed evaluation scope')
    source=T_RUN/'eval_inputs/manifest.json';m=copy.deepcopy(read(source))
    for key in ('endpoint_entries','rollout_entries','compatibility_entries'):m[key]=[]
    m['standard_entries']=[e for e in m['standard_entries'] if e['mode'] in ('q4','q12','joint')]
    for e in m['standard_entries']:
        for field in ('record_path','reference_npz','condition_npz','representation','geometry_calibration_record'):
            if not Path(e[field]).is_file():raise FileNotFoundError(e[field])
        e.update(raw_query_frames=60 if e['mode']=='q12' else 50,
                 block_lengths={'q4':[4]*12+[2],'q12':[12]*5,'joint':[50]}[e['mode']])
    priorities={'q4':1,'q12':2,'joint':3};jobs=[]
    for e in m['standard_entries']:
        for arm in ARMS:
            jobs.append(dict(arm=arm,entry_id=e['id'],checkpoint=str(run/arm/'coordinate_final.pt'),checkpoint_step=512,
                             optimizer_updates=OPTIMIZER_UPDATES[arm],seeds=[17,29],priority=priorities[e['mode']],
                             output=str(run/'generation'/arm/e['id']),matched_group=e['id']))
    jobs.sort(key=lambda j:(j['priority'],j['entry_id'],ARMS.index(j['arm'])))
    m.update(schema='q4_align_existing_input_scope_v1',source_frozen_inputs=str(source),endpoint_origins=[],
             prepared_counts={'standard_entries':18,'endpoint_entries':0,'rollout_entries':0,'compatibility_entries':0},
             generation_matrix_status='108 new trajectories; no new endpoints or coarse/fine',GPU_seconds=0,model_calls=0,
             protocol_note='Exact T sampler: Q4 twelve full4 blocks then final2; Q12 five full12/raw60 and first50.')
    plan=dict(schema='q4_align_finite_generation_v1',jobs=jobs,expected_new_draws=108,budget_seconds=86400,
              counts_by_priority={'1':36,'2':36,'3':36},training_cap_each=10800,
              parent_standard_reused_draws=36,T_standard_reused_draws=72,parent_evaluation=str(PARENT_EVAL),T_evaluation=str(T_EVAL),
              checkpoint_step_semantics='step is schedule slot512; optimizer_updates is256/512/512',
              order='Q4 then Q12 then joint; all three arms adjacent per case; no outcome-dependent selection')
    assert len(jobs)==54 and sum(len(j['seeds']) for j in jobs)==108
    write_json(dest/'manifest.json',m);write_json(dest/'genplan.json',plan)
    print(json.dumps({'manifest':str(dest/'manifest.json'),'jobs':54,'new_draws':108,'reused_draws':108}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--output-root',type=Path)
    p.add_argument('--import-generated',action='store_true');p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--arms',nargs='+',choices=ARMS);p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--priorities',nargs='+',type=int,choices=range(1,4));a=p.parse_args()
    a.run_root=a.run_root.resolve();a.output_root=a.output_root or a.run_root/'evaluation'
    if a.import_generated:base.import_generated(a)
    else:prepare(a.run_root)

if __name__=='__main__':main()
