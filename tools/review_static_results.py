"""N2: reaggregate stable S results, never retrain or overwrite originals."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from molvid.evaluation.static_response import paired_comparison, _flatten
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    arms=('S0','Sdec','Strunk');dynamic={};static={};identities={};inputs=[];protocols=[]
    for arm in arms:
        base=a.source/arm
        summary=json.loads((base/'summary.json').read_text())
        assert summary['status']=='completed' and summary['validation_completed'] and not summary['test_opened']
        assert sha256_file(base/'final.pt')==summary['checkpoint_sha256']
        inputs.extend([base/'summary.json',base/'validation/protocol.json',base/'validation/dynamic_rows.json',base/'validation/static_rows.json'])
        protocol=json.loads((base/'validation/protocol.json').read_text())
        assert protocol.pop('checkpoint_sha256')==summary['checkpoint_sha256']
        protocols.append(protocol);identities[arm]=summary
        dynamic[arm]=json.loads((base/'validation/dynamic_rows.json').read_text())
        static[arm]=json.loads((base/'validation/static_rows.json').read_text())
    assert all(p==protocols[0] for p in protocols)
    freeze_run_identity(a.output,inputs=inputs,purpose='N2 completed S reaggregation',protocol=protocols[0])
    result={'identities':identities,'dynamic_means':{},'static_by_sigma':{},'test_opened':False}
    for arm in arms:
        groups=defaultdict(lambda:defaultdict(list))
        for row in dynamic[arm]:
            if row['variant']=='generated':
                for key,value in _flatten(row['metrics']).items(): groups[key][row['system']].append(value)
        result['dynamic_means'][arm]={k:float(np.mean([np.mean(v) for v in systems.values()])) for k,systems in groups.items()}
        table={(r['system'],r['sigma_A'],r['variant']):r['metrics']['coordinate_mse'] for r in static[arm]}
        result['static_by_sigma'][arm]={}
        for sigma in sorted({r['sigma_A'] for r in static[arm]}):
            systems=sorted({r['system'] for r in static[arm] if r['sigma_A']==sigma})
            pred=np.array([table[s,sigma,'sampled_trunk'] for s in systems])
            identity=np.array([table[s,sigma,'identity_input'] for s in systems]);delta=pred-identity
            rng=np.random.default_rng(31);boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
            result['static_by_sigma'][arm][str(sigma)]=dict(mse=float(pred.mean()),identity_mse=float(identity.mean()),
                delta=float(delta.mean()),ci95_system_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
                systems_better=int((delta<0).sum()),systems=len(systems),per_system_delta=dict(zip(systems,delta.tolist())))
    result['dynamic_paired']={f'{second}_minus_{first}':paired_comparison(dynamic[first],dynamic[second])
                             for first,second in [('S0','Sdec'),('S0','Strunk'),('Sdec','Strunk')]}
    passes=all(r['ci95_system_bootstrap'][1]<0 for r in result['static_by_sigma']['Strunk'].values())
    result['decision']=dict(static_beats_identity_at_all_declared_sigmas=passes,
        scale_up='not_authorized_by_evidence',
        clipping_factorial='not_triggered' if not passes else 'requires_dynamic_regression_check',
        rationale='Require net restoration before more static training; OOD bypass is not an independently trained decoder control.')
    result['missing_metrics']=['angle/chirality not measured in original S outputs','training-seed uncertainty (one training seed)',
                               'independently trained lightweight static decoder control']
    atomic_write_json(a.output/'summary.json',result)
    print(json.dumps({'dynamic_means':result['dynamic_means'],'static_by_sigma':result['static_by_sigma'],
                      'decision':result['decision']},indent=2))


if __name__=='__main__': main()
