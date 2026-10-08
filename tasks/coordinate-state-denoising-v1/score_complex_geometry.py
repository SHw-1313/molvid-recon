"""Full-atom frozen-rule geometry on complete raw complex64 forecast panels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.evaluation.cached_clashes import CachedClashScorer
from molvid.evaluation.ca_detail import system_estimate
from molvid.geometry.types import StaticTopologyMetadata
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.runtime import atomic_write_json
from molvid.training.predictive_data import RELEASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generated', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.generated/'summary.json').read_text())
    source_rows = json.loads((args.generated/'rows.json').read_text())
    if summary['status']!='completed' or summary.get('conditions',summary.get('successful_conditions'))!=128 or len(source_rows)!=128:
        raise ValueError('Complete64-system128-condition panel required')
    inventory = json.loads((args.data/'summary.json').read_text())
    release = json.loads(RELEASE.read_text())
    recipe = Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    config = PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output/'protocol.json', dict(generated=str(args.generated), data=str(args.data),
        physical=config.contract(), units='Angstrom', raw_output=True, model_calls=0,
        clash_rule='Existing sampled_clash_metrics: exclude graph-distance<=2; separate1-4 allowance; undirected pairs once; no neighbor cap',
        component_normalization='Every clash category uses total valid complex atom count, so counts/severity are additive; affected fractions need not be additive',
        bond_rule='RMSE of all valid undirected covalent lengths versus real future, across draws and frames'))
    store = ClipMMapDataset(args.data/'valid')
    lookup = {fields[0]: i for i,fields in enumerate(store._clip_index_fields)}
    rows = []
    started = time.monotonic()
    try:
        for index, source_row in enumerate(source_rows):
            record = store[lookup[source_row['sample_id']]]
            batch = collate_clip_records([record])
            topology = StaticTopologyMetadata.from_batch(batch)
            scorer = CachedClashScorer(topology, batch.loss_mask, config)
            saved = torch.load(args.generated/f'coordinates_{index:04d}.pt', map_location='cpu', weights_only=False)
            if saved['sample_id']!=record['sample_id'] or len(saved['coordinates'])!=8 or saved['variant']!=summary['variant']:
                raise ValueError('Original complete eight-draw forecast required')
            draws = saved['coordinates'].numpy()
            truth = np.asarray(record['x'])[4:]
            if draws.shape != (8,*truth.shape):
                raise ValueError('Shared complete future atom/frame protocol required')
            per_draw = [scorer.trajectory(draw) for draw in draws]
            metrics = {key:float(np.mean([r[key] for r in per_draw])) for key in per_draw[0]}
            metrics.update({'MD.'+k:v for k,v in scorer.trajectory(truth).items()})
            bond = np.sort(np.asarray(record['bond_index']).T, axis=1)
            bond = np.unique(bond[bond[:,0]!=bond[:,1]], axis=0)
            valid = np.asarray(record['loss_mask'], dtype=bool)
            bond = bond[valid[bond].all(-1)]
            i,j = bond.T
            component = np.asarray(record['component_id'])
            error = np.square(np.linalg.norm(draws[:,:,i]-draws[:,:,j],axis=-1)
                -np.linalg.norm(truth[:,i]-truth[:,j],axis=-1))
            for name, mask in [('all',np.ones(len(i),dtype=bool)),('protein',(component[i]==0)&(component[j]==0)),
                               ('ligand',(component[i]==1)&(component[j]==1))]:
                if not mask.any(): raise ValueError('Expected protein and ligand covalent topology')
                metrics[name+'_bond_rmse_A'] = float(np.sqrt(error[:,:,mask].mean()))
            rows.append(dict(system=record['system_id'],sample_id=record['sample_id'],metrics=metrics))
            atomic_write_json(args.output/'rows.json',rows)
            print(json.dumps(dict(conditions=len(rows),seconds=time.monotonic()-started,model_calls=0)),flush=True)
    finally:
        store.close()
    strata = dict(all=inventory['systems'],old16=inventory['old16'],new48=inventory['new48'])
    estimates = {}
    for name, systems in strata.items():
        grouped = defaultdict(list)
        for row in rows:
            if row['system'] in systems:
                for metric,value in row['metrics'].items():grouped[metric].append((row['system'],value))
        estimates[name] = {metric:system_estimate(values) for metric,values in grouped.items()}
    atomic_write_json(args.output/'summary.json',dict(status='completed',variant=summary['variant'],
        systems=64,conditions=128,model_calls=0,estimates=estimates['all'],strata=estimates,
        seconds=time.monotonic()-started,limitations=['Reused development64, not sealed independent confirmation',
            'Geometry scores must be read jointly with motion; static coordinates can have good geometry',
            'Same legacy TRAIN radius table as protein scoring; not a new physical force-field evaluation']))


if __name__=='__main__':main()
