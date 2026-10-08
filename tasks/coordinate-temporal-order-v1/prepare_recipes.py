"""Prepare one matched order ablation; never launch training or copy data."""
import argparse
import json
from pathlib import Path

from molvid.runtime import atomic_write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for scheme in ('flow', 'edm'):
        source = args.parent/'recipes'/f'{scheme}_md.json'
        settings = json.loads(source.read_text())
        if (settings['schema'] != 'molvid.coordinate_state.recipe.v1'
                or settings['scheme'] != scheme
                or settings['initialization'] != 'coordinate_state_weights_only'):
            raise ValueError('Expected the original paired static-to-MD recipe')
        parent_checkpoint = args.parent/'seed71'/scheme/'static'/'coordinate_final.pt'
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
        settings = dict(settings, schema='molvid.coordinate_state.recipe.v2',
                        interaction_order='tokens_then_temporal')
        destination = args.output/f'{scheme}_md.json'
        atomic_write_json(destination, settings)
        arms[scheme] = dict(recipe=str(destination), source_recipe=str(source),
            warm_start=str(parent_checkpoint), seed=71, proposed_updates=settings['updates'],
            matched_control=str(args.parent/'seed71'/scheme/'md'/'coordinate_final.pt'))
    atomic_write_json(args.output/'protocol.json', dict(status='prepared_not_submitted', arms=arms,
        changed_fields=['schema', 'interaction_order'], model_calls=0, training_updates=0,
        decision='No automatic launch; finish current paired complex/time/rollout evidence first',
        static_retraining=False, data_copied=False, default_promotion=False))
    print(json.dumps(dict(status='prepared_not_submitted', schemes=list(arms), model_calls=0)))


if __name__ == '__main__':
    main()
