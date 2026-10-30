"""Existing common scores for new draws; metadata-only merge of completed T."""
import argparse,copy,csv,importlib.util,sys
from pathlib import Path
import numpy as np
import prepare_evaluation as scope
sys.path.insert(0,str(scope.OLD))
spec=importlib.util.spec_from_file_location('q4_existing_comparison',scope.OLD/'compare_results.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
BASELINES={'shared':'T_shared','routed':'T_routed'}


def reuse_T(out):
    dest=out/'trajectories';result=scope.read(dest/'summary.json');panel=scope.read(dest/'panel_index.json')
    old=scope.read(scope.T_EVAL/'trajectories/summary.json')
    oldpanel={c['id']:c for c in scope.read(scope.T_EVAL/'trajectories/panel_index.json')['cases']}
    selected={c['id'] for c in panel['cases']}
    if result.get('T_reuse_completed'):raise ValueError('Do not duplicate completed T merge')
    def selected_row(row):return row.get('case') in selected and row.get('arm') in BASELINES
    def renamed(row):
        value=copy.deepcopy(row);value.update(arm=BASELINES[row['arm']],reused_from=str(scope.T_EVAL),source_arm=row['arm']);return value
    for case in panel['cases']:
        previous=oldpanel[case['id']]
        for key in common.KEYS:
            if case['entry'][key]!=previous['entry'][key]:raise ValueError('T reuse protocol mismatch: '+key)
        with np.load(case['reference_npz'],allow_pickle=False) as new,np.load(previous['reference_npz'],allow_pickle=False) as original:
            for key in ('positions','observed','time_ps','atom_identity'):
                if not np.array_equal(new[key],original[key]):raise ValueError('T reuse reference mismatch: '+key)
        for model in previous['models']:
            if model['name'] not in BASELINES:continue
            item=copy.deepcopy(model);item.update(name=BASELINES[model['name']],reused_from=str(scope.T_EVAL),source_arm=model['name'])
            for draw in item['draws']:
                for key in ('npz','local_geometry_json','metrics_json'):
                    if not Path(draw[key]).is_file():raise FileNotFoundError(draw[key])
                draw.update(reused_from=str(scope.T_EVAL),source_arm=model['name'])
            case['models'].append(item)
    extra=[renamed(r) for r in old['draw_rows'] if selected_row(r)]
    result['draw_rows']+=extra
    for filename in ('geometry_slices.json','temporal_order.json','distance.json'):
        current=scope.read(dest/filename);previous=scope.read(scope.T_EVAL/'trajectories'/filename)
        current += [renamed(r) for r in previous if selected_row(r)]
        scope.write_json(dest/filename,current)
    for filename in ('motion_frames.csv','seam_frames.csv'):
        with (dest/filename).open() as f:current=list(csv.DictReader(f))
        with (scope.T_EVAL/'trajectories'/filename).open() as f:previous=list(csv.DictReader(f))
        current += [renamed(r) for r in previous if selected_row(r)]
        common.shared.save_csv(dest/filename,current)
    rows=result['draw_rows'];means=common.averages(rows,['case','base_case','arm',*common.KEYS])
    distance=scope.read(dest/'distance.json')
    dmeans=common.averages([dict(d,status='ok',**{'D.'+k:d[k] for k in ('D_A2','shuffle_platform_A2','D_over_shuffle','D_A2_absolute_MD_error','D_over_shuffle_absolute_MD_error')}) for d in distance],
                           ['case','base_case','arm',*common.KEYS,'observable','lag_frames','lag_ps'])
    result.update(per_case=means,per_dataset=common.averages(means,['arm',*common.KEYS],'metrics'),
                  distance_per_case=dmeans,distance_per_dataset=common.averages(dmeans,['arm',*common.KEYS,'observable','lag_frames','lag_ps'],'metrics'),
                  reused_T_draws=sum(r['status']=='ok' for r in extra),T_reuse_completed=True,
                  T_reuse='Completed selected same-case/clock/protocol T draw rows, G/D, geometry, frames, seams and raw NPZ; no rescoring')
    # Keep new_trajectory_draws unchanged: old T rows cannot inflate new coverage.
    common.shared.save_csv(dest/'draw_metrics.csv',rows);common.shared.save_csv(dest/'distance.csv',distance)
    scope.write_json(dest/'summary.json',result);scope.write_json(dest/'panel_index.json',panel)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=scope.RUN/'evaluation')
    p.add_argument('--stage',choices=('endpoints','trajectories','compatibility','summarize'),required=True);p.add_argument('--export-viewer',action='store_true')
    a=p.parse_args();scope.cpu_only();common.load_engine(scope.ENGINE)
    if a.stage=='trajectories':common.trajectories(a.output_root);reuse_T(a.output_root);return
    if a.stage!='summarize':getattr(common,a.stage)(a.output_root);return
    result=common.summarize(a.output_root,a.export_viewer)
    rows=result['trajectory_per_case'];lookup={(r['case'],r['arm']):r for r in rows};contrasts=[]
    comparisons=[('shared_joint_only','T_shared'),('shared_q4_q1','T_shared'),('routed_q4_q1','T_routed'),
                 ('shared_q4_q1','shared_joint_only'),('routed_q4_q1','shared_q4_q1')]
    for arm,control in comparisons:
        for row in rows:
            if row['arm']!=arm:continue
            reference=lookup.get((row['case'],control))
            if reference:
                contrasts.append(dict(case=row['case'],base_case=row['base_case'],arm=arm,reference_arm=control,**{k:row[k] for k in common.KEYS},
                    metrics={k:v-reference['metrics'][k] for k,v in row['metrics'].items() if common.shared.finite(v) and common.shared.finite(reference['metrics'].get(k))}))
    result['matched_control_contrasts']+=contrasts
    traj=scope.read(a.output_root/'trajectories/summary.json')
    result.update(study='Q4 supervision alignment and Q1 removal, frozen equal-capacity adapters',reused_T_draws=traj['reused_T_draws'],
                  total_reused_draws=result['reused_parent_draws']+traj['reused_T_draws'],
                  parent_reuse='L4096 selected36 and completed T shared/routed selected72 are read-only exact-protocol references',
                  checkpoint_note='Final schedule slots512; actual optimizer updates A256, B/C512. A removes Q1 exposure, not an equal-update control.',
                  block_protocol='Exact T: Q4 [4]*12+[2] raw50; Q12 [12]*5 raw60/first50; truncated tails are not full-block trends.')
    scope.write_json(a.output_root/'comparison/comparison_summary.json',result)

if __name__=='__main__':main()
