"""Finite route checks; CUDA execution belongs to the engineer's study budget.

No CUDA is run by importing this file. A single small real TRAIN case covers
the inherited parent, native source, Q1 RF and existing ligand tail auxiliary.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_flow_capacity_cuda import batch_fixture
from molvid.codec.frame import slice_clip_frames
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.flow.task_adapter import TaskAdapters
from molvid.training.coordinate_state import load_coordinate_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Budgeted real CUDA checks required')
PARENT = Path('/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt')
TASK = ROOT / 'tasks/flow-task-route-20261029'


def recipe(mode='routed', small=False):
    result = json.loads((TASK / 'configs/shared.json').read_text())
    if mode == 'none':
        result.pop('task_adapter')
    else:
        result['task_adapter']['mode'] = mode
    if small:
        result.update(width=32, spatial_layers=4, num_rbf=8, query_post_extra_layers=1)
        if mode != 'none':
            result['task_adapter']['bottleneck'] = 8
    return result


def legacy_class():
    code = subprocess.check_output(['git', 'show', 'eca03dd:molvid/flow/coordinate_state.py'], cwd=ROOT, text=True)
    module = ModuleType('molvid.flow._task_route_legacy')
    module.__package__ = 'molvid.flow'
    sys.modules[module.__name__] = module
    exec(compile(code, 'eca03dd_coordinate_state', 'exec'), module.__dict__)
    return module.CoordinateStateDenoiser


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def test_default_shared_initialization_rng_and_explicit_task_identity():
    torch.manual_seed(71)
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    old = legacy_class()(recipe('none', small=True))
    after = torch.get_rng_state().clone()
    models = []
    for mode in ('none', 'shared', 'routed'):
        torch.set_rng_state(cpu)
        model = CoordinateStateDenoiser(recipe(mode, small=True))
        assert torch.equal(torch.get_rng_state(), after)
        assert torch.equal(torch.cuda.get_rng_state(), cuda)
        for key, value in old.state_dict().items():
            assert torch.equal(value, model.state_dict()[key]), key
        if mode == 'none':
            assert model.contract() == old.contract()
            assert set(model.state_dict()) == set(old.state_dict())
        else:
            assert all(name.startswith('task_adapters.') == p.requires_grad for name, p in model.named_parameters())
        models.append(model.cuda().eval())
    assert set(models[1].state_dict()) == set(models[2].state_dict())
    assert all(torch.equal(value, models[2].state_dict()[key]) for key, value in models[1].state_dict().items())
    clip = batch_fixture()
    observed = slice_clip_frames(clip, 0, 3)
    current, times = clip.x[3:], clip.time_ps[:, 3:]
    with torch.no_grad():
        baseline = old.cuda().eval()(current, condition=old.prepare_condition(observed),
            query_time_ps=times, progress=current.new_tensor([.3, .6]))
        for model, mode in zip(models, ('none', 'shared', 'routed')):
            condition = model.prepare_condition(observed, route_task=None if mode == 'none' else 'joint')
            actual = model(current, condition=condition, query_time_ps=times, progress=current.new_tensor([.3, .6]))
            torch.testing.assert_close(actual, baseline, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='explicit'):
        models[-1].prepare_condition(observed)
    with pytest.raises(ValueError, match='static Q1'):
        models[-1].prepare_condition(slice_clip_frames(clip, 0, 1), task='static', route_task='q1')


def test_adapter_equal_compute_initial_gradients_soft_routing_and_equivariance():
    torch.manual_seed(31)
    shared = TaskAdapters(32, 8, 'shared').cuda()
    routed = TaskAdapters(32, 8, 'routed').cuda()
    routed.load_state_dict(shared.state_dict(), strict=True)
    h, v = torch.randn(2, 10, 32, device='cuda'), torch.randn(2, 10, 3, 32, device='cuda')
    for adapter in (shared, routed):
        dh, dv = adapter(h, v, 'joint')
        assert torch.equal(dh, h) and torch.equal(dv, v)
        loss = dh.square().mean() + dv.square().mean()
        loss.backward()
        for branch in adapter.branches:
            assert branch.up_h.weight.grad.norm() > 0 and branch.up_v.weight.grad.norm() > 0
            assert branch.down_h.weight.grad.count_nonzero() == 0
            assert branch.down_v.weight.grad.count_nonzero() == 0
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    # A test-only learned state after the zero outputs have opened; no production
    # initializer is changed and both branches remain active on both tasks.
    with torch.no_grad():
        for branch in shared.branches:
            branch.up_h.weight.normal_(std=.02)
            branch.up_v.weight.normal_(std=.02)
    routed.load_state_dict(shared.state_dict(), strict=True)
    counts = [0, 0]
    def count(index):
        def hook(*unused): counts[index] += 1
        return hook
    hooks = [branch.register_forward_hook(count(i)) for i, branch in enumerate(routed.branches)]
    try:
        a = routed(h, v, 'joint')
        b = routed(h, v, 'q1')
    finally:
        for hook in hooks: hook.remove()
    assert counts == [2, 2]
    c = shared(h, v, 'joint')
    d = shared(h, v, 'q1')
    for av, bv, cv, dv in zip(a, b, c, d):
        assert torch.equal(cv, dv)
        torch.testing.assert_close((av + bv) * .5, cv, atol=5e-7, rtol=1e-6)
        assert (av - bv).abs().max() > 1e-4
    rotation = h.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated_v = torch.einsum('ij,qnjd->qnid', rotation, v)
    equivariant = routed(h, rotated_v, 'joint')
    torch.testing.assert_close(equivariant[0], a[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(equivariant[1], torch.einsum('ij,qnjd->qnid', rotation, a[1]), atol=1e-6, rtol=1e-6)
    changed_h, changed_v = h.clone(), v.clone()
    changed_h[:, 6:] += 7.; changed_v[:, 6:] *= 3.
    separated = routed(changed_h, changed_v, 'joint')
    assert torch.equal(separated[0][:, :6], a[0][:, :6])
    assert torch.equal(separated[1][:, :6], a[1][:, :6])


def test_frozen_backbone_real_input_gradients_rng_and_optimizer_resume():
    torch.manual_seed(91)
    model = CoordinateStateDenoiser(recipe('routed', small=True)).cuda().train()
    clip = batch_fixture()
    observed = slice_clip_frames(clip, 0, 3)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith('task_adapters.')}
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-3, weight_decay=.01)
    generator = torch.Generator(device='cuda').manual_seed(14)
    def update(m, opt, gen, route):
        opt.zero_grad(set_to_none=True)
        q = 1 if route == 'q1' else 2
        condition = m.prepare_condition(observed, route_task=route)
        loss, metrics = coordinate_state_objective(m, condition, clip.x[3:3+q], clip.time_ps[:, 3:3+q], generator=gen)
        loss.backward()
        assert torch.isfinite(loss)
        for branch in m.task_adapters.branches:
            assert branch.up_h.weight.grad.norm() > 0
            assert branch.up_v.weight.grad.norm() > 0
        assert all(p.grad is None for name,p in m.named_parameters() if not name.startswith('task_adapters.'))
        torch.nn.utils.clip_grad_norm_(trainable(m), 1., error_if_nonfinite=True)
        opt.step()
        return metrics
    update(model, optimizer, generator, 'joint')
    snapshot = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                    generator=generator.get_state().clone(), cpu=torch.get_rng_state().clone(),
                    cuda=torch.cuda.get_rng_state().clone())
    expected = update(model, optimizer, generator, 'q1')
    resumed = CoordinateStateDenoiser(recipe('routed', small=True)).cuda().train()
    resumed.load_state_dict(snapshot['model'], strict=True)
    resumed_opt = torch.optim.AdamW(trainable(resumed), lr=1e-3, weight_decay=.01)
    resumed_opt.load_state_dict(snapshot['optimizer'])
    resumed_gen = torch.Generator(device='cuda'); resumed_gen.set_state(snapshot['generator'])
    torch.set_rng_state(snapshot['cpu']); torch.cuda.set_rng_state(snapshot['cuda'])
    actual = update(resumed, resumed_opt, resumed_gen, 'q1')
    for name in ('source_epsilon_sum', 'source_epsilon_square_sum', 'progress_mean'):
        assert actual[name] == expected[name]
    assert torch.equal(generator.get_state(), resumed_gen.get_state())
    for key, value in model.state_dict().items():
        if key in frozen:
            assert torch.equal(value, frozen[key]) and torch.equal(resumed.state_dict()[key], frozen[key]), key
        else:
            torch.testing.assert_close(resumed.state_dict()[key], value, atol=2e-6, rtol=1e-5)
    assert any(p.grad.norm() > 0 for name,p in resumed.task_adapters.named_parameters() if 'down_' in name)


def test_real_parent_shared_q1_source_and_ligand_auxiliary():
    from molvid.data.batch import collate_clip_records
    from molvid.data.bridge_mixture import BridgeMixture
    from molvid.training.coordinate_state import LigandSourceGeometry
    from molvid.training.kernel_tasks import default_rng_state
    cfg = recipe()
    mixture = BridgeMixture(cfg['data'], 71)
    try:
        buckets = mixture.groups['complex']['3M35']
        key = min(buckets, key=lambda s:float(s.split('_')[1][:-2]))
        store, index = buckets[key][0]
        raw = store[index]
        assert raw['split'] == 'train'
        clip = collate_clip_records([raw]).to('cuda')
    finally:
        mixture.close()
    models = []
    cpu_start = torch.get_rng_state().clone()
    rng_after = None
    for mode in ('none', 'shared', 'routed'):
        torch.set_rng_state(cpu_start)
        model = CoordinateStateDenoiser(recipe(mode)).cuda().eval()
        report = load_coordinate_weights(model, PARENT)
        assert report['parent_step'] == 4096
        if rng_after is None: rng_after = torch.get_rng_state().clone()
        else: assert torch.equal(rng_after, torch.get_rng_state())
        models.append(model)
    old, shared, routed = models
    for key, value in old.state_dict().items():
        assert torch.equal(value, shared.state_dict()[key]) and torch.equal(value, routed.state_dict()[key]), key
    assert sum(p.numel() for p in trainable(routed)) == 166528
    assert sum(p.numel() for p in routed.parameters()) == 10175884
    observed = slice_clip_frames(clip, 0, 4)
    current, query = clip.x[4:6] + .2, clip.time_ps[:, 4:6]
    with torch.no_grad():
        fields = [m(current, condition=m.prepare_condition(observed, route_task=None if m is old else 'joint'),
                    query_time_ps=query, progress=current.new_tensor([.5])) for m in models]
    for field in fields[1:]:
        torch.testing.assert_close(field, fields[0], atol=2e-4, rtol=2e-5)
    # H1 is still dynamic Q1. Same source stream and native 18 intervals /35NFE.
    obs1 = slice_clip_frames(observed, 3, 4)
    query = clip.time_ps[:, 4:5]
    source_rng, source_outputs = [], []
    first_states = []
    for model in (old, routed):
        condition = model.prepare_condition(obs1, route_task=None if model is old else 'q1')
        gen = torch.Generator(device='cuda').manual_seed(17)
        def capture(module, args, kwargs):
            if bool((kwargs['progress'] == 0).all()): first_states.append(args[0].detach().clone())
        hook = model.register_forward_pre_hook(capture, with_kwargs=True)
        calls = model.forward_calls
        try: output = model.sample(condition, query, generator=gen)
        finally: hook.remove()
        assert model.forward_calls-calls == 35 and torch.isfinite(output).all()
        source_outputs.append(output); source_rng.append(gen.get_state())
    assert len(first_states) == 2 and torch.equal(*first_states) and torch.equal(*source_rng)
    torch.testing.assert_close(source_outputs[1], source_outputs[0], atol=2e-4, rtol=2e-5)
    # Only adapters train, but frozen suffix layers must carry the input gradient.
    routed.train(); routed.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        condition = routed.prepare_condition(observed, route_task='q1')
        loss, rf = coordinate_state_objective(routed, condition, clip.x[4:5], query,
            generator=torch.Generator(device='cuda').manual_seed(23))
    loss.backward()
    rf_grad = float(torch.nn.utils.clip_grad_norm_(trainable(routed), 1., error_if_nonfinite=True))
    assert rf_grad > 0 and all(p.grad is None for name,p in routed.named_parameters() if not name.startswith('task_adapters.'))
    routed.zero_grad(set_to_none=True)
    aux = LigandSourceGeometry(cfg['ligand_geometry'], 71, 'cuda')
    before = copy.deepcopy(default_rng_state())
    result = aux.backward(routed, observed, clip.x[4:5], query, bf16=True,
                          band_reference=clip.x, route_task='q1')
    after = default_rng_state()
    assert torch.equal(before['cpu'], after['cpu'])
    assert all(torch.equal(a,b) for a,b in zip(before['cuda'], after['cuda']))
    aux_grad = float(torch.nn.utils.clip_grad_norm_(trainable(routed), 1., error_if_nonfinite=True))
    assert result['ligand_geometry_applied'] and aux_grad > 0
    assert all(p.grad is None for name,p in routed.named_parameters() if not name.startswith('task_adapters.'))
    print(json.dumps(dict(real_train_sample=clip.sample_id[0], atoms=clip.atom_count,
        inherited_base_parameters=10009356,trainable_adapter_parameters=166528,
        source35=True,source_exact=True,RF_gradient=rf_grad,auxiliary_gradient=aux_grad,auxiliary=result)))
