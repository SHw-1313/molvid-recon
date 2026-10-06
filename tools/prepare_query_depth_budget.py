"""Register query-trainable arms against the exact completed long controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, sha256_file
from molvid.training.coordinate_budget import QUERY_LONG_BUDGET, resolve_coordinate_budget


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old = Path('/workspace/molvid-coordinate-budget8192-20261006/runs/coordinate_budget8192_20261006')
    manifest = old / 'recipes_v2/manifest.json'
    references = json.loads(manifest.read_text())['runs']
    if {(r['depth'], r['seed']) for r in references} != {(d, s) for d in (4, 8) for s in (71, 72)} or len(references) != 4:
        raise ValueError('exactly four registered controls required')
    inputs = [manifest]; rows = []
    for ref in references:
        config = Path(ref['configuration'])
        if sha256_file(config) != ref['configuration_sha256']:
            raise ValueError('reference recipe changed')
        raw = json.loads(config.read_text())
        if resolve_coordinate_budget(raw) != 8192:
            raise ValueError('long control required')
        parent = Path(raw['coordinate_initialization']['checkpoint'])
        if sha256_file(parent) != ref['parent_sha256']:
            raise ValueError('parent changed')
        name = f"query{ref['depth']}_{ref['seed']}"
        recipe = deepcopy(raw)
        recipe.update(coordinate_train_query_geometry=True, coordinate_query_geometry_recompute=True,
                      coordinate_budget_experiment=dict(QUERY_LONG_BUDGET))
        resolve_coordinate_budget(recipe)
        path = args.output / f'{name}.json'
        atomic_write_json(path, recipe)
        prefix = old / f"qualification_v1/{ref['name']}_pilot17/coordinate_step_0016.pt"
        control = old / f"formal_v1/{ref['name']}"
        state = json.loads((control / 'status.json').read_text())
        if state.get('status') != 'completed' or state.get('forecasts') != 192:
            raise ValueError('completed fixed-final control required')
        rows.append(dict(name=name, seed=ref['seed'], depth=ref['depth'],
            configuration=str(path), configuration_sha256=sha256_file(path),
            legacy_configuration=str(config), legacy_configuration_sha256=sha256_file(config),
            legacy_prefix=str(prefix), legacy_prefix_sha256=sha256_file(prefix),
            parent_sha256=ref['parent_sha256'], selected_schedule_sha256=ref['selected_schedule_sha256'],
            control_root=str(control)))
        inputs.extend((config, path, parent, prefix, control / 'status.json'))
    freeze_run_identity(args.output, inputs=list(dict.fromkeys(inputs)),
        purpose='Query geometry x depth x long-budget registration; no neural calls',
        protocol=dict(training_updates=0, forecasts=0, test_opened=False, independent_test=False,
            budget=QUERY_LONG_BUDGET, hypotheses='query feature adaptation may interact with depth and optimization'))
    atomic_write_json(args.output / 'manifest.json', dict(status='prepared_not_submitted', runs=rows,
        formal_updates_planned=32768, forecasts_planned=768, default_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output = args.output.resolve()
    run(args)
