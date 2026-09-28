"""Family-separated A validation tables, system bootstrap and residue figures."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from molvid.runtime import atomic_write_json


METRICS=('bond_rmse_A','clashes_per_1000_atoms','vdw_severity_A2_per_atom','residue_rmsf_mae_A','msd_curve_mae_A2',
         'atom_rmsf_mae_A','residue_rmsf_pearson','residue_rmsf_spearman','system_mean_rmsf_prediction_A','system_mean_rmsf_target_A')


def read_systems(root):
    families={}
    for family in ('ordinary','fixed_history'):
        with (root/family/'per_system.csv').open() as handle:
            rows=[r for r in csv.DictReader(handle) if r['path']=='generated' and r['clock']=='true']
        by_system=defaultdict(list)
        for r in rows:by_system[r['system']].append(r)
        expected=8 if family=='ordinary' else 4
        if len(by_system)!=8 or any(len(v)!=expected for v in by_system.values()):
            raise RuntimeError(f'{family}: expected eight systems with complete lag/H coverage')
        families[family]={system:{metric:float(np.mean([float(r[metric]) for r in values if r.get(metric)]))
                                  for metric in METRICS} for system,values in by_system.items()}
    return families


def distribution(values):
    pred=np.array([v['system_mean_rmsf_prediction_A'] for v in values.values()])
    true=np.array([v['system_mean_rmsf_target_A'] for v in values.values()])
    pearson=float(np.corrcoef(pred,true)[0,1]) if pred.std()>0 and true.std()>0 else None
    spread=float(pred.std()/true.std()) if true.std()>0 else None
    from scipy.stats import spearmanr
    return {'system_mean_mae_A':float(np.abs(pred-true).mean()),'system_pearson':pearson,
            'system_spearman':float(spearmanr(pred,true).statistic) if pearson is not None else None,
            'spread':spread,'signed_amplitude_bias_A':float((pred-true).mean()),'mean_generated_RMSF_A':float(pred.mean())}


def comparison(first,second,metric):
    systems=sorted(first)
    if systems!=sorted(second):raise ValueError('paired system sets differ')
    delta=np.array([second[s][metric]-first[s][metric] for s in systems])
    rng=np.random.default_rng(20260928)
    draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    base=np.mean([first[s][metric] for s in systems]);change=float(delta.mean())
    return {'delta':change,'relative_delta':change/base if abs(base)>1e-8 else None,
            'near_zero_baseline':bool(abs(base)<=1e-8),'paired_system_CI95':np.quantile(draws,[.025,.975]).tolist(),
            'system_deltas':dict(zip(systems,delta.tolist()))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',action='append',required=True,help='NAME=/absolute/evaluation/root');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True);(args.output/'per_residue_profiles').mkdir(exist_ok=True)
    roots={name:Path(path) for name,path in (arg.split('=',1) for arg in args.arm)}
    data={name:read_systems(root) for name,root in roots.items()}
    if 'A0' not in data:raise ValueError('A0 is required as the paired baseline')
    summaries={};comparisons={};flat=[]
    for arm,families in data.items():
        summaries[arm]={family:{m:float(np.mean([r[m] for r in rows.values()])) for m in METRICS} for family,rows in families.items()}
        summaries[arm]['ordinary'].update(distribution(families['ordinary']))
        for family,rows in families.items():
            flat.extend({'arm':arm,'family':family,'system':system,**values} for system,values in rows.items())
        if arm!='A0':
            comparisons[arm]={family:{m:comparison(data['A0'][family],families[family],m) for m in METRICS} for family in families}
    with (args.output/'per_system.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['arm','family','system',*METRICS]);writer.writeheader();writer.writerows(flat)
    guardrails={}
    base=summaries['A0']['ordinary']
    for arm in comparisons:
        ordinary=summaries[arm]['ordinary'];motion=comparisons[arm]
        flags=[]
        for family,metric in (('ordinary','residue_rmsf_mae_A'),('fixed_history','msd_curve_mae_A2')):
            value=motion[family][metric]
            if value['relative_delta'] is None:flags.append(f'{metric}: near-zero baseline requires interpretation')
            elif value['relative_delta']>.05:flags.append(f'{metric}: deterioration >5%')
        if ordinary['system_pearson'] is None or base['system_pearson'] is None:flags.append('system Pearson unavailable')
        elif ordinary['system_pearson']<base['system_pearson']-.05:flags.append('system Pearson drops >0.05')
        if not ordinary['spread'] or not base['spread']:flags.append('spread unavailable')
        elif abs(math.log(ordinary['spread']))>abs(math.log(base['spread']))+.10:flags.append('absolute log spread increases >0.10')
        guardrails[arm]={'pass':not flags,'flags':flags}
    # Preserve all families and H-specific buckets from evaluator JSON as appendices.
    buckets={arm:{family:json.loads((root/family/'summary.json').read_text()) for family in ('ordinary','fixed_history')} for arm,root in roots.items()}
    # All 8 systems are plotted (not hand-picked); representative ranking uses MD amplitude only.
    systems=sorted(data['A0']['ordinary'],key=lambda s:data['A0']['ordinary'][s]['system_mean_rmsf_target_A'])
    for arm,root in roots.items():
        residue=defaultdict(list);msd=defaultdict(list)
        for family in ('ordinary','fixed_history'):
            rows=[json.loads(line) for line in (root/family/'rows.jsonl').read_text().splitlines()]
            for row in rows:
                if row['path']!='generated':continue
                meta=row['meta'];system=meta['system']
                if family=='ordinary':
                    with np.load(root/family/'profiles'/row['profile_file']) as profile:
                        residue[(system,int(meta['history_frames']))].append((profile['residue_prediction_A'].copy(),profile['residue_target_A'].copy()))
                else:
                    for point in row['metrics']['msd']['curve']:
                        msd[(system,int(meta['lag_ps']),point['horizon_ps'])].append((point['prediction_A2'],point['target_A2']))
        for (system,h),values in residue.items():
            pred=np.mean([v[0] for v in values],axis=0);true=np.mean([v[1] for v in values],axis=0)
            np.savetxt(args.output/'per_residue_profiles'/f'{arm}_{system}_H{h}.csv',np.c_[np.arange(len(pred)),true,pred],delimiter=',',header='residue_index,MD_RMSF_A,generated_RMSF_A',comments='')
            fig,ax=plt.subplots(figsize=(7,3));ax.plot(true,label='MD');ax.plot(pred,label=f'{arm} generated');ax.set(xlabel='residue index',ylabel='window RMSF (Å)',title=f'{system} ordinary H{h}');ax.legend();fig.tight_layout();fig.savefig(args.output/'figures'/f'{arm}_{system}_H{h}_RMSF.png');plt.close(fig)
        for system in systems:
            fig,ax=plt.subplots(figsize=(6,4))
            for lag in (100,200,300,400):
                entries=sorted((t,np.mean(v,axis=0)) for (s,l,t),v in msd.items() if s==system and l==lag)
                ax.plot([t for t,v in entries],[v[0] for t,v in entries],label=f'generated {lag}ps')
                ax.plot([t for t,v in entries],[v[1] for t,v in entries],linestyle='--',alpha=.6,label=f'MD {lag}ps')
            ax.set(xlabel='physical horizon (ps)',ylabel='MSD (Å²)',title=f'{arm} {system} fixed-history');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/'figures'/f'{arm}_{system}_MSD.png');plt.close(fig)
        pred=[data[arm]['ordinary'][s]['system_mean_rmsf_prediction_A'] for s in systems];true=[data[arm]['ordinary'][s]['system_mean_rmsf_target_A'] for s in systems]
        fig,ax=plt.subplots();ax.scatter(true,pred);ax.plot([0,max(true+pred)],[0,max(true+pred)],linestyle='--');ax.set(xlabel='MD system mean RMSF (Å)',ylabel='generated system mean RMSF (Å)',title=arm);fig.tight_layout();fig.savefig(args.output/'figures'/f'{arm}_system_RMSF_scatter.png');plt.close(fig)
    release_trigger=False
    if 'A2' in data and 'A3' in data:
        a2=summaries['A2'];a3=summaries['A3']
        bond_better=sum(a3[f]['bond_rmse_A'] for f in ('ordinary','fixed_history'))<sum(a2[f]['bond_rmse_A'] for f in ('ordinary','fixed_history'))
        motion_worse=a3['ordinary']['residue_rmsf_mae_A']>1.05*a2['ordinary']['residue_rmsf_mae_A'] or a3['fixed_history']['msd_curve_mae_A2']>1.05*a2['fixed_history']['msd_curve_mae_A2']
        amplitude_worse=a3['ordinary']['mean_generated_RMSF_A']<.9*a2['ordinary']['mean_generated_RMSF_A'] and abs(a3['ordinary']['signed_amplitude_bias_A'])>abs(a2['ordinary']['signed_amplitude_bias_A'])
        release_trigger=bond_better and (motion_worse or amplitude_worse)
    eligible=['A0']+[arm for arm,v in guardrails.items() if v['pass']]
    axes=[(family,metric) for family in ('ordinary','fixed_history')
          for metric in ('vdw_severity_A2_per_atom','clashes_per_1000_atoms','bond_rmse_A')]
    axes += [('ordinary','residue_rmsf_mae_A'),('fixed_history','msd_curve_mae_A2')]
    pareto=[]
    for arm in eligible:
        a=np.array([summaries[arm][f][m] for f,m in axes])
        if not any(np.all(np.array([summaries[b][f][m] for f,m in axes])<=a) and np.any(np.array([summaries[b][f][m] for f,m in axes])<a) for b in eligible if b!=arm):pareto.append(arm)
    atomic_write_json(args.output/'selection.json',{'status':'exploratory_validation','guardrails':guardrails,'pareto_candidates':pareto,'keep_release_trigger':release_trigger,
        'paired_comparisons_to_A0':comparisons,'summary':summaries,'bucket_appendices':buckets,'bootstrap_unit':'system','bootstrap_draws':10000,
        'main_aggregation':'within each family: four lag grids equal; ordinary H4/H8 equal; systems equal',
        'representative_systems_by_MD_amplitude':[systems[0],systems[len(systems)//2],systems[-1]],'test_opened':False})


if __name__=='__main__':main()
