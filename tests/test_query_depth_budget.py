from copy import deepcopy
import pytest
import torch
from molvid.training.coordinate_budget import LONG_BUDGET, QUERY_LONG_BUDGET, resolve_coordinate_budget, final_coordinate_budget
from tools.run_query_depth_budget import compare_default, compare_resume, require_cuda


def recipe(query=True):
    return dict(coordinate_training={'total_updates':8192}, coordinate_schedule_offset=2048,
        coordinate_initialization={'checkpoint':'parent'}, coordinate_dit_depth=8,
        coordinate_train_query_geometry=query, coordinate_query_geometry_recompute=query,
        coordinate_budget_experiment=deepcopy(QUERY_LONG_BUDGET if query else LONG_BUDGET))


def test_registered_and_default_budget():
    assert resolve_coordinate_budget({}) == 2048
    assert resolve_coordinate_budget(recipe()) == 8192
    assert resolve_coordinate_budget(recipe(False)) == 8192


@pytest.mark.parametrize('key,value', [('coordinate_query_geometry_recompute', False),
    ('coordinate_train_query_geometry', 1), ('coordinate_budget_experiment', LONG_BUDGET),
    ('coordinate_source_sigma_angstrom', .7), ('coordinate_static', {}),
    ('coordinate_schedule_offset', 0), ('coordinate_local_geometry_input', 'current')])
def test_unregistered_mixture_rejected(key, value):
    raw = recipe(); raw[key] = value
    with pytest.raises(ValueError): resolve_coordinate_budget(raw)


def test_final_marker_requires_actual_model():
    contract = dict(coordinate_budget_experiment=QUERY_LONG_BUDGET,
        training={'total_updates':8192}, schedule_offset=2048, selected_schedule_sha256='schedule')
    with pytest.raises(ValueError): final_coordinate_budget(contract)
    contract['model'] = dict(query_encoder='trainable_query_copy_frozen_observation_current_radius_graph',
        query_geometry_execution=QUERY_LONG_BUDGET['query_execution'])
    assert final_coordinate_budget(contract) == 8192


def test_state_and_scientific_contract_must_match():
    old = dict(cursor=16, model={'x':torch.ones(2)}, optimizer={'step':16}, rng={'flow':torch.ones(4)},
        contract={'runtime_source_sha256':{'old':'sha'}, 'seed':71})
    new = deepcopy(old); new['contract']['runtime_source_sha256'] = {'new':'sha'}
    compare_default(old, new)
    for key in ('model', 'optimizer', 'rng'):
        bad = deepcopy(new); bad[key] = {'changed':1}
        with pytest.raises(ValueError): compare_default(old, bad)
    new['contract']['seed'] = 72
    with pytest.raises(ValueError): compare_default(old, new)
    active = deepcopy(old); active['cursor'] = 17
    active['contract'] = dict(coordinate_budget_experiment=QUERY_LONG_BUDGET, model={'dit_depth':8})
    compare_resume(active, deepcopy(active), 8)
    with pytest.raises(ValueError): compare_resume(active, deepcopy(active), 4)


@pytest.mark.parametrize('tag', ['skipped', 'failure', 'error'])
def test_cuda_skips_never_pass(tmp_path, tag):
    path = tmp_path / 'cuda.xml'
    path.write_text('<testsuites><testsuite>' + '<testcase/>' * 5 + f'<testcase><{tag}/></testcase></testsuite></testsuites>')
    with pytest.raises(ValueError): require_cuda(path)
