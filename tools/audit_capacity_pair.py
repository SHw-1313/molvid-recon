"""Reaggregate completed C4/C8 rows; this tool performs NO model training."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from molvid.evaluation.ca_detail import flat_metrics,system_estimate
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file

METRICS=('geometry.bond_rmse','physical.clashes_per_1000_atoms',
         'physical.vdw_severity_A2_per_atom','msd.curve_mae_A2',
         'rmsf.mean_amplitude_ratio','rmsf.residue_profile.mae_A',
         'geometry.angles.mae_radian','internal.torsion_increment_radian.mae_radian')


def audit(root,output):
    if output.exists(): raise FileExistsError(output)
    files=[root/arm/family/name for arm in ('C4','C8') for family in ('ordinary','fixed_history')
           for name in ('summary.json','protocol.json','rows.jsonl')]
    freeze_run_identity(output,inputs=files,purpose='Read-only completed C4/C8 paired aggregation',
                        protocol=dict(test_opened=False,new_training=False,metrics=METRICS))
    results={}
    for family,expected in [('ordinary',1920),('fixed_history',960)]:
        arms={};protocols={}
        for arm in ('C4','C8'):
            path=root/arm/family
            protocol=json.loads((path/'protocol.json').read_text());protocols[arm]=protocol
            summary=json.loads((path/'summary.json').read_text())
            if protocol['test_opened'] or summary['test_opened']:raise ValueError('sealed data violation')
            if summary['checkpoint_sha256']!=protocol['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
            rows=[json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
            if len(rows)!=expected or len(rows)!=summary['rows']:raise ValueError(f'incomplete {arm}/{family}')
            keyed={(r['meta']['sample_id'],r['path'],r['clock'],r['seed']):r for r in rows}
            if len(keyed)!=len(rows):raise ValueError('duplicate paired row')
            arms[arm]=keyed
        for key in protocols['C4']:
            if key!='checkpoint_sha256' and protocols['C4'][key]!=protocols['C8'][key]:
                raise ValueError(f'protocol mismatch {family}/{key}')
        if arms['C4'].keys()!=arms['C8'].keys():raise ValueError('unpaired output coverage')
        grouped=defaultdict(list);paired=defaultdict(list)
        for key,row4 in arms['C4'].items():
            row8=arms['C8'][key];system=row4['meta']['system']
            if row4['meta']!=row8['meta']:raise ValueError('paired record metadata changed')
            if row4['clock']!='true':continue
            m4,m8=flat_metrics(row4['metrics']),flat_metrics(row8['metrics'])
            for metric in METRICS:
                if metric not in m4 or metric not in m8:continue
                if not math.isfinite(m4[metric]) or not math.isfinite(m8[metric]):raise ValueError('nonfinite metric')
                for arm,m in [('C4',m4),('C8',m8)]:
                    grouped[(arm,row4['path'],metric)].append((system,m[metric]))
                paired[(row4['path'],metric)].append((system,m8[metric]-m4[metric]))
        results[family]=dict(rows_per_arm=expected,
            arms={'|'.join(k):system_estimate(v) for k,v in grouped.items()},
            paired_C8_minus_C4={'|'.join(k):system_estimate(v) for k,v in paired.items()},
            checkpoint_sha256={a:p['checkpoint_sha256'] for a,p in protocols.items()})
    atomic_write_json(output/'summary.json',dict(status='completed',results=results,test_opened=False,
        new_training_updates=0,limitations=['existing valid8 only','single training seed',
        'C4/C8 capacity intervention includes depth and added G blocks; not a pure parameter-count law']))
    print(json.dumps({f:{k:v['mean'] for k,v in r['paired_C8_minus_C4'].items() if k.startswith('generated|')}
                      for f,r in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    audit(args.root,args.output)
