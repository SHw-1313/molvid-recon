"""System-paired development comparisons, with explicit baseline exposure strata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from molvid.evaluation.ca_detail import system_estimate
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json


METRICS=('bond_rmse','clashes_per_1000_atoms','vdw_severity_A2_per_atom',
    'msd.curve_mae_A2','rmsf.residue_profile.mae_A','rmsf.mean_amplitude_ratio',
    'angle_cosine_rmse','torsion_increment_circular_error')


def paired_rows(left,right,systems):
    def keyed(rows):
        out={}
        for r in rows:
            if r['system'] not in systems: continue
            key=(r['system'],r['sample_id'],r['seed'])
            if key in out: raise ValueError('duplicate prediction identity')
            out[key]=r
        return out
    a,b=keyed(left),keyed(right)
    if not a or a.keys()!=b.keys(): raise ValueError('unpaired draws; do not silently drop cases')
    result={}
    for metric in METRICS:
        if not all(np.isfinite(r['metrics'][metric]) for r in list(a.values())+list(b.values())):
            raise ValueError('nonfinite metric')
        result[metric]=system_estimate([(k[0],a[k]['metrics'][metric]-b[k]['metrics'][metric]) for k in a])
    return dict(systems=len(systems),paired_draws=len(a),delta_left_minus_right=result)


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    groups=defaultdict(list);inputs=[args.exposure];protocols=[]
    for directory in args.generated:
        summary=json.loads((directory/'summary.json').read_text())
        if summary['status']!='completed' or summary['systems']!=48 or summary.get('pilot',False):
            raise ValueError('completed48-system results only')
        protocol=json.loads((directory/'protocol.json').read_text());protocols.append(protocol)
        inputs += [directory/'summary.json',directory/'rows.json',directory/'protocol.json']
        rows=json.loads((directory/'rows.json').read_text())
        variants={r['variant'] for r in rows}
        if variants & groups.keys(): raise ValueError('variant name collision')
        for row in rows: groups[row['variant']].append(row)
    records=[{e['sample_id']:e['sha256'] for e in p['records']} for p in protocols]
    if any(r!=records[0] for r in records): raise ValueError('different evaluation clips')
    systems={r['system'] for r in groups['a1']}
    overlap=json.loads(args.exposure.read_text())['overlap']['development']
    exposed=set(overlap['pvb_train'])
    strata=dict(all_development=systems,pvb_known_dynamic_train=systems&exposed,
                pvb_not_known_dynamic_train=systems-exposed)
    pairs=[(name,'a1') for name in sorted(groups) if name!='a1']
    if 'pvb' in groups:
        pairs += [(name,'pvb') for name in sorted(groups) if name not in ('pvb','a1')]
    for pair in args.pair:
        names=pair.split(':')
        if len(names)!=2 or any(n not in groups for n in names): raise ValueError('explicit pair must name two available variants')
        if tuple(names) not in pairs: pairs.append(tuple(names))
    comparisons={}
    for left,right in pairs:
        comparisons[f'{left}_minus_{right}']={label:paired_rows(groups[left],groups[right],members)
            for label,members in strata.items() if members}
    freeze_run_identity(args.output,inputs=inputs,purpose='Paired expanded development comparison',protocol=dict(
        compute='CPU',model_calls=0,training_updates=0,test_opened=False,
        bootstrap_unit='system; anchors and inference seeds averaged within system',
        training_seed_uncertainty='not estimated from two training runs',
        pvb_condition='native H1 versus Molvid H4',independent_baseline_test=False))
    atomic_write_json(args.output/'summary.json',dict(status='completed',comparisons=comparisons,
        model_calls=0,training_updates=0,promotion='none; require motion, chirality, independent and long-time evidence',
        amplitude_ratio_note='delta is not an improvement direction; ideal amplitude ratio is one',
        strata_qualification='known dynamic train IDs only, not static pretraining or fold isolation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,nargs='+',required=True)
    p.add_argument('--exposure',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair',action='append',default=[],help='additional left:right matched comparison')
    run(p.parse_args())
