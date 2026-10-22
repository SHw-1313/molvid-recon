"""Small CUDA checks; the P1 owner schedules and charges their execution."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from molvid.codec.frame import slice_clip_frames
from molvid.flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from molvid.training import coordinate_state as trainer
from test_flow_capacity_cuda import batch_fixture, settings

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real CUDA required')
ROOT = Path(__file__).resolve().parents[1]
STATIC_TASKS = ['static:protein']*8+['static:atlas']*4+['static:misato']*4


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    else:
        assert a == b


def replay_config():
    return dict(enabled=True, every=4, weight=.25, corruption_A=.2,
                seed_offset=20261023, data=dict(sixteen_step_tasks=STATIC_TASKS))


def fake_mixture_type(clip):
    class FixtureMixture:
        """Deterministic records, while deliberately exercising all global RNGs."""
        def __init__(self, config, seed):
            self.config, self.rng, self.visits = config, random.Random(seed), 0
            self.probe_rng()

        @staticmethod
        def probe_rng():
            return [random.random(), float(np.random.rand()), float(torch.rand(())),
                    *[float(torch.rand((), device=f'cuda:{i}')) for i in range(torch.cuda.device_count())]]

        def next(self):
            index = self.rng.randrange(10000)
            probe = self.probe_rng()
            task, domain = self.config['sixteen_step_tasks'][self.visits % 16].split(':')
            history = (4, 8)[self.visits % 2] if task == 'dynamic' else 1
            self.visits += 1
            batch = replace(clip, x=clip.x+index*1e-5, bpos=clip.bpos+index*1e-5)
            if task == 'static':
                batch = slice_clip_frames(batch, 0, 1)
            return batch, task, history, ([] if task == 'static' else list(range(history,16))), dict(
                system=f'fixture{index}', visit=self.visits, domain=domain, fixture_global_rng=probe)

        def state_dict(self):
            return dict(rng=self.rng.getstate(), visits=self.visits)

        def load_state_dict(self, state):
            self.rng.setstate(state['rng']); self.visits = state['visits']

        def close(self):
            pass
    return FixtureMixture


class StochasticRF(nn.Module):
    """Fast trainer fixture; the next test separately uses the real RF model."""
    def __init__(self, config):
        super().__init__()
        self.settings, self.scheme = config, 'flow'
        self.rho_A = config['scales']['flow_rho_A']
        self.sigma_data = config['scales']['sigma_data_A']
        self.gain = nn.Parameter(torch.randn(()))
        self.seen = []

    def contract(self):
        return dict(test_fixture='stochastic_RF')

    def prepare_condition(self, observed, task):
        return SimpleNamespace(observed=observed, reference_A=observed.x[-1], task=task)

    def forward(self, current, *, condition, query_time_ps, progress):
        mask = nn.functional.dropout(torch.ones_like(current), .2, training=self.training)
        self.seen.append((condition.task, current.detach().clone(), progress.detach().clone(), mask.clone()))
        return self.gain*(current-condition.reference_A)*mask


def recipe(updates, replay=False):
    value = settings(updates=updates, warmup_updates=2, save_every=4, bf16=False,
        cpu_threads=1, initialization='random_from_scratch_all_modules_trainable',
        history_sampling='alternating_optimizer_updates', static_corruption_A=.2,
        data=dict(sixteen_step_tasks=['dynamic:protein']*8+['dynamic:complex']*8))
    if replay:
        value['static_replay'] = replay_config()
    return value


def load_run(path):
    saved = torch.load(path/'coordinate_final.pt', map_location='cpu', weights_only=False)
    rows = [json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()]
    return saved, rows


def test_replay_off_matches_frozen_trainer_and_consumes_no_new_rng(monkeypatch, tmp_path):
    clip = batch_fixture(frames=16)
    source = subprocess.check_output(['git','show','76102d8:molvid/training/coordinate_state.py'], cwd=ROOT, text=True)
    legacy = ModuleType('molvid.training._static_replay_legacy_test')
    legacy.__package__ = 'molvid.training'
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, '76102d8_coordinate_trainer', 'exec'), legacy.__dict__)
    for module in (legacy, trainer):
        monkeypatch.setattr(module, 'BridgeMixture', fake_mixture_type(clip))
        monkeypatch.setattr(module, 'CoordinateStateDenoiser', StochasticRF)
    before = trainer._global_rng_state()
    assert trainer.static_replay_from_settings(recipe(2), 71, 'cuda') is None
    assert trainer.static_replay_from_settings(dict(recipe(2), static_replay={'enabled':False}),71,'cuda') is None
    assert_same(before, trainer._global_rng_state())
    legacy.train(recipe(2), tmp_path/'old', 71)
    trainer.train(recipe(2), tmp_path/'new', 71)
    old, old_rows = load_run(tmp_path/'old')
    new, new_rows = load_run(tmp_path/'new')
    for key in ('model','optimizer','generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng','mixture','exposure'):
        assert_same(old[key], new[key])
    assert 'static_replay' not in new and 'replay_exposure' not in new
    assert [r['history'] for r in new_rows] == [4, 8]
    for a, b in zip(old_rows, new_rows):
        assert_same({k:v for k,v in a.items() if k!='elapsed_seconds'},
                    {k:v for k,v in b.items() if k!='elapsed_seconds'})


def test_real_q1_adds_finite_gradients_and_restores_every_dynamic_rng(monkeypatch):
    clip = batch_fixture(frames=16)
    monkeypatch.setattr(trainer, 'BridgeMixture', fake_mixture_type(clip))
    model = CoordinateStateDenoiser(settings(spatial_layers=6, query_post_extra_layers=1,
                                             checkpoint_query_blocks=True)).cuda().train()
    dynamic_generator = torch.Generator(device='cuda').manual_seed(915)
    condition = model.prepare_condition(slice_clip_frames(clip,0,4), task='dynamic')
    loss, _ = coordinate_state_objective(model,condition,clip.x[4:6],clip.time_ps[:,4:6],generator=dynamic_generator)
    loss.backward()
    before_grads = {name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    before_rng = trainer._global_rng_state()
    replay = trainer.StaticReplay(replay_config(), 71, 'cuda')
    assert_same(before_rng, trainer._global_rng_state())
    dedicated_before = dynamic_generator.get_state().clone()
    metadata = replay.backward(model, bf16=True)
    assert_same(before_rng, trainer._global_rng_state())
    assert torch.equal(dynamic_generator.get_state(), dedicated_before)
    delta = sum(float((p.grad-before_grads.get(name,torch.zeros_like(p))).float().square().sum())
                for name,p in model.named_parameters() if p.grad is not None)
    assert delta > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert metadata['RF_loss'] > 0 and metadata['weighted_loss'] == .25*metadata['RF_loss']
    assert metadata['query_frames'] == 1 and metadata['corruption_A'] == .2
    assert replay.exposure['backward_calls'] == 1
    assert replay.exposure['static_examples'] == clip.batch_size
    assert replay.exposure['atom_query_pairs'] == int(clip.loss_mask.sum())
    replay.close()


def test_same_dynamic_streams_single_update_and_exact_replay_resume(monkeypatch, tmp_path):
    clip = batch_fixture(frames=16)
    monkeypatch.setattr(trainer, 'BridgeMixture', fake_mixture_type(clip))
    models = []
    def factory(config):
        model = StochasticRF(config); models.append(model); return model
    monkeypatch.setattr(trainer, 'CoordinateStateDenoiser', factory)
    trainer.train(recipe(8), tmp_path/'off', 71)
    trainer.train(recipe(8,True), tmp_path/'on', 71)
    trainer.train(recipe(8,True), tmp_path/'resume', 71, stop_after=4)
    trainer.train(recipe(8,True), tmp_path/'resume', 71,
                  resume=tmp_path/'resume/checkpoint_latest.pt')
    off, off_rows = load_run(tmp_path/'off')
    on, on_rows = load_run(tmp_path/'on')
    resumed, resumed_rows = load_run(tmp_path/'resume')
    for key in ('generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng','mixture','exposure'):
        assert_same(off[key], on[key])
    for key in ('model','optimizer','generator_rng','torch_rng','cuda_rng','python_rng','numpy_rng',
                'mixture','exposure','static_replay','replay_exposure'):
        assert_same(on[key], resumed[key])
    dynamic = lambda model: [r for r in model.seen if r[0]=='dynamic']
    assert_same(dynamic(models[0]), dynamic(models[1]))  # Actual noisy tensors, progress AND dropout masks.
    assert_same(dynamic(models[1]), dynamic(models[2])+dynamic(models[3]))
    for a, b, c in zip(off_rows, on_rows, resumed_rows):
        for key in ('sample','system','visit','domain','fixture_global_rng','history','queries','progress_mean'):
            assert_same(a[key], b[key]); assert_same(b[key], c[key])
        left, right = copy.deepcopy(b), copy.deepcopy(c)
        for row in (left, right):
            row.pop('elapsed_seconds')
            if 'replay' in row:
                assert row['replay'].pop('wall_seconds') >= 0
        assert_same(left, right)
    assert [r['step'] for r in on_rows if r['replay_applied']] == [4, 8]
    assert on['replay_exposure']['backward_calls'] == 2
    assert on['exposure']['updates'] == 8
    assert all(int(v['step']) == 8 for v in on['optimizer']['state'].values())
    # This fixture has ONE scalar parameter. With positive gradients above the
    # common clip threshold, both arms can have the identical clipped gradient
    # and AdamW update even though Q1 adds a real gradient. Inspect the logged
    # pre-clip norm and summed objective instead of demanding different weights.
    replay_steps = [(a,b) for a,b in zip(off_rows,on_rows) if b['replay_applied']]
    assert any(a['grad_norm'] != b['grad_norm'] for a,b in replay_steps)
    for _, row in replay_steps:
        assert row['loss'] == row['dynamic_RF_loss']+row['replay']['weighted_loss']
        assert row['replay']['weighted_loss'] > 0
