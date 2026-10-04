"""Reproducible aggregation of bounded review-followup evidence; no selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from molvid.runtime import atomic_write_json, sha256_file


KEYS=['bond_rmse','clashes_per_1000_atoms','vdw_severity_A2_per_atom',
      'msd.curve_mae_A2','rmsf.mean_amplitude_ratio','rmsf.residue_profile.mae_A',
      'internal.distance_increment.mae_A','internal.torsion_increment_radian.mae_radian',
      'angle_cosine_rmse','torsion_circular_error','latent_h_mse','latent_v_mse']


def value(row,key):
    item=row['metrics']
    for part in key.split('.'):
        if part not in item: return None
        item=item[part]
    return item


def estimate(per_system):
    x=np.array(list(per_system.values()),dtype=float);rng=np.random.default_rng(51)
    boot=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95_system_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
                per_system=per_system,systems=len(x))


def metrics(rows):
    result={}
    for key in KEYS:
        groups=defaultdict(list)
        for row in rows:
            v=value(row,key)
            if v is not None: groups[row['system']].append(v)
        if groups: result[key]=estimate({s:float(np.mean(v)) for s,v in groups.items()})
    return result


def summarize(root,output):
    if output.exists(): raise FileExistsError(output)
    inputs=[root/'n1/rows.json',root/'n1solver32/rows.json',root/'n2/summary.json',
        root/'n3u/paired_comparisons.json',root/'n3k/paired_comparisons.json',
        root/'n3ridge/summary.json',root/'n4/summary.json']
    rows=json.loads(inputs[0].read_text())+json.loads(inputs[1].read_text())
    primary={}
    for variant,steps in [('oracle',0),('persistence',0),('generated',8),('generated',16),('generated',32)]:
        primary[f'{variant}_{steps}']=metrics([r for r in rows if r['lag_ps']==100 and r['variant']==variant and r['steps']==steps])
    paired={}
    for first,second in [(8,16),(16,32)]:
        a=primary[f'generated_{first}'];b=primary[f'generated_{second}']
        paired[f'{second}_minus_{first}']={k:estimate({s:b[k]['per_system'][s]-a[k]['per_system'][s]
            for s in a[k]['per_system']}) for k in a}
    density_files=[root/'n1density/rows.json',root/'n1density/common_endpoint_features.pt',root/'n1/common_endpoint_features.pt']
    inputs+=density_files
    dense_rows=json.loads(density_files[0].read_text())
    dense_endpoint=torch.load(density_files[1],map_location='cpu',weights_only=True)
    base_endpoint=torch.load(density_files[2],map_location='cpu',weights_only=True)
    for system,endpoint in dense_endpoint.items():
        if not torch.equal(endpoint['target'],base_endpoint[system]['target']):
            raise ValueError('density intervention changed the reference endpoint')
    baseline=metrics([r for r in rows if r['variant']=='generated' and r['steps']==16 and r['lag_ps']==100 and r['anchor']==450])
    dense=metrics([r for r in dense_rows if r['variant']=='generated'])
    density=dict(H4=baseline,H7=dense,common_reference_endpoint_identical=True,
        H7_minus_H4={k:estimate({s:dense[k]['per_system'][s]-baseline[k]['per_system'][s]
            for s in baseline[k]['per_system']}) for k in baseline})
    heads={}
    for arm in ('last_mu','delta_mu','history_mu','last_muvar','delta_muvar','history_muvar'):
        files=[root/'n3u'/f'{arm}_{seed}_metrics.json' for seed in (101,102,103)];inputs+=files
        reports=[json.loads(p.read_text())['calibrated'] for p in files]
        heads[arm]={metric:{lag:dict(seed_means=[r[metric][lag]['mean'] for r in reports],
            mean=float(np.mean([r[metric][lag]['mean'] for r in reports]))) for lag in ('100','400','1200')}
            for metric in ('mse','nll','coverage90','interval_width90')}
    atomic_write_json(output,dict(status='completed',input_sha256={str(p):sha256_file(p) for p in inputs},
        N1_primary=primary,N1_paired_solver=paired,N1_observation_density=density,N3_calibrated_heads=heads,
        limitations=['N1 system bootstrap conditions on one training checkpoint and two source seeds.',
            'N1 grid extension is one draw/system: no common-endpoint distribution equivalence conclusion.',
            'N3 seed mean is descriptive, not a combined system/training-seed confidence interval.',
            'All results are development diagnostics, not sealed evaluation or checkpoint promotion.'],test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();summarize(a.root,a.output)
