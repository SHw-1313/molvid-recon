"""Declare the paired static->MD followup after completed MD-only sampling."""
import argparse
import copy
import json
from pathlib import Path

from molvid.runtime import atomic_write_json


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    recipes={s:json.loads((args.recipes/f'{s}.json').read_text()) for s in ('flow','edm')}
    registration=Path(recipes['flow']['data']['static_registration'])
    sources=json.loads((registration/'registered_sources.json').read_text())
    protein=json.loads((args.protein/'complete.json').read_text())
    complexes=json.loads((args.complex_data/'summary.json').read_text())
    protected={e['system'].split('_')[1].upper() for e in protein['files']} | set(complexes['old16']+complexes['new48'])
    overlap=[r['system_id'] for r in sources if r['system_id'].split('_')[0].upper() in protected]
    if overlap:
        raise ValueError(f'Direct development PDB overlap: {overlap}')
    for scheme,base in recipes.items():
        static=copy.deepcopy(base)
        static['updates']=2048
        static['static_corruption_A']=base['scales']['sigma_data_A']
        static['data']['sixteen_step_tasks']=['static:protein']*8+['static:complex']*8
        md=copy.deepcopy(base)
        md['initialization']='coordinate_state_weights_only'
        atomic_write_json(args.output/f'{scheme}_static.json',static)
        atomic_write_json(args.output/f'{scheme}_md.json',md)
    atomic_write_json(args.output/'protocol.json',dict(
        status='prepared_not_trained',seed=71,static_updates_each=2048,MD_updates_each=2048,
        total_additional_optimizer_updates=8192,
        static_sources=len(sources),static_maximum_atoms=max(r['atoms'] for r in sources),
        static_sampling='Equal source-system cycles;1024 registered protein and1024 complex single-structure examples per arm; not advertised as sequence-cluster-balanced',
        static_complex_source='Uniform frame from existing selected128 TRAIN complex systems; singleton coordinate recovery, not a zero-motion trajectory',
        static_target='Clean singleton; only iid-corrupted singleton is available as reference/condition; no physical future labels',
        corruption_scale_A=recipes['flow']['scales']['sigma_data_A'],
        corruption_reason='Frozen TRAIN-derived MD displacement component scale, shared by A/B; not an AF3 scale or selected on validation',
        initialization='Fresh seed71 random weights for static; own static final weights for MD, all trainable; no MD-only checkpoint inherited',
        MD_stage='Same256-system MD subset,2048 updates and seed as MD-only baseline; fresh optimizer/RNG/data visits after weights-only stage transition',
        direct_development_PDB_overlap=overlap,
        historical_static_registration_flag='training_qualified=false records that original registration performed no neural training. Reuse existing source identities without rewriting that record; this run establishes only its own actual execution evidence.',
        additional_sampling='First compare fixed-final protein100/native80 and complex64; do not automatically expand to extra seeds or source/network/loss sweeps',
        limitations=['Metadata-only exact PDB isolation, not new family/scaffold independence.',
            'Protein static subset is not a complete bound-state complex; complex singletons come from authorized MD TRAIN.',
            'Added static exposure/compute versus MD-only is confounded without an equal-resource longer-MD control.',
            'Static geometry training does not supply temporal labels or prove corrected trajectory correlations.']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ('recipes','protein','complex-data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
