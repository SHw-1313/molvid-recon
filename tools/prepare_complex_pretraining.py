"""Two matched complex-mixture objectives; reuse already completed GQ controls."""
import json
from pathlib import Path
from molvid.runtime import atomic_write_json


def run():
    output=Path('runs/complex_pretraining_20261007')
    data=Path('/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2')
    for seed in (71,72):
        source=Path('/workspace/molvid-query-geometry-training-20261006/runs/query_geometry_training_20261006/recipes_v1')/f'query_geometry_{seed}.json'
        for arm,fraction in (('atom',None),('balanced',.5)):
            raw=json.loads(source.read_text())
            raw['coordinate_query_geometry_recompute']=True
            raw['coordinate_training']['ligand_rf_fraction']=fraction
            raw['coordinate_complex_mixture']=dict(data=str(data),activation_offload_above_atoms=4000)
            destination=output/f'{arm}_{seed}/recipe.json'
            if destination.exists():raise FileExistsError(destination)
            atomic_write_json(destination,raw)
    atomic_write_json(output/'experiment.json',dict(
        arms=['atom_71','balanced_71','atom_72','balanced_72'],
        updates_per_arm=2048,protein_updates=1024,complex_updates=1024,
        original_control='/workspace/molvid-query-geometry-training-20261006/runs/query_geometry_training_20261006/formal_v2/gq_{seed}',
        initialization='same frozen coordinate2048 seed-specific parent as original GQ controls',
        treatment='mixed complex/protein data; role-balanced versus atom-mean RF',
        equal_compute=False,checkpoint_selection='fixed_final_only',
        static_updates=0,independent_test=False,
        evaluation='both unchanged protein development48 and full-complex validation16; native clocks',
        limitations=['domain and clock mixture are changed together versus protein-only control',
            'query activation recomputation changes execution strategy, not architecture or objective',
            'this is a dynamics mixture experiment, not yet joint static pretraining or equilibrium validation']))


if __name__=='__main__':run()
