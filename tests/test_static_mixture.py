"""Q1 snapshot sampling, source balance, and exact data-stream replay on CPU."""
import copy
import json
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molvid.data import bridge_mixture as module
from molvid.data.bridge_mixture import BridgeMixture


def _record(sample_id, source, frames, offset=0):
    xyz = np.arange(frames * 4 * 3, dtype=np.float32).reshape(frames, 4, 3) + offset
    return dict(
        schema_version='pvb.clip.v1', sample_id=sample_id, source=source, split='train',
        task='static' if frames == 1 else 'trajectory', coordinate_unit='angstrom',
        time_bucket_id='static' if frames == 1 else 'dt_100ps',
        time_ps=np.arange(frames, dtype=np.float32) * 100,
        delta_time_ps=np.full(frames - 1, 100, dtype=np.float32),
        frame_mask=np.ones(frames, dtype=bool), x=xyz, bpos=xyz[:, [0, 0, 2, 2]],
        atype=np.array([5, 5, 6, 6]), btype=np.array([118, 118, 119, 119]),
        block_id=np.array([0, 0, 1, 1]),
        component_id=np.array([0, 0, 1, 1]) if source == 'misato' else np.zeros(4, dtype=np.int64),
        atom_source_index=np.arange(4), atom_identity=[f'{sample_id}:a{i}' for i in range(4)],
        edge_mask=np.zeros(4, dtype=np.int64), loss_mask=np.ones(4, dtype=bool),
        align_mask=np.array([True, True, True, False]),
        bond_index=np.array([[0, 1, 2, 3], [1, 0, 3, 2]]))


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    atlas = [_record('atlas_A_A_R1_dt_100ps_w000000', 'atlas', 16)]
    atlas += [_record(f'atlas_B_A_R1_dt_100ps_w{i:06}', 'atlas', 16, i * 1000) for i in range(3)]
    misato = [_record(f'misato_1ABC_MD_dt_100ps_w{i:06}', 'misato', 16, i * 1000) for i in range(2)]
    for record in atlas + misato:
        record['frame_mask'][13:] = False
    # Legacy serialized TRAIN records may omit frame_mask (all frames valid).
    del misato[1]['frame_mask']
    pdb = [_record(f'pdb::{i}', 'screened_public_mmcif', 1, i * 100) for i in range(2)]
    stores = {'atlas': atlas, 'misato': misato, 'pdb': pdb}
    monkeypatch.setattr(BridgeMixture, 'open', lambda self, path: stores[str(path)])
    monkeypatch.setattr(module, 'get_clip_specs', lambda store: [SimpleNamespace(
        frames=r['x'].shape[0], time_bucket_id=r['time_bucket_id'], sample_id=r['sample_id'], index=i)
        for i, r in enumerate(store)])
    (tmp_path / 'registered_sources.json').write_text(json.dumps([
        dict(source_sample_id=r['sample_id'], diagnostic_store_index=i) for i, r in enumerate(pdb)]))
    (tmp_path / 'run_identity.json').write_text(json.dumps({'input_sha256': {'pdb/data.bin': 'fixture'}}))
    shards = tmp_path / 'shards.json'
    shards.write_text('[]')
    settings = dict(protein_base='atlas', protein_shards=str(shards), complex_train='misato',
                    static_registration=str(tmp_path), expected_systems=dict(protein=2, complex=1, static_protein=2),
                    clock_sampling='staggered_system_cycles', query_frames='all',
                    sixteen_step_tasks=['static:protein'] * 8 + ['static:atlas'] * 4 + ['static:misato'] * 4)
    return settings, stores


def _signature(result):
    batch, task, history, queries, meta = result
    return (task, history, queries, meta, batch.sample_id, batch.x.numpy().tobytes(),
            batch.bpos.numpy().tobytes(), batch.bond_index.numpy().tobytes())


def test_balanced_q1_snapshots_keep_selected_frame_and_topology(inventory):
    settings, stores = inventory
    untouched = copy.deepcopy(stores)
    sampler = BridgeMixture(settings, 71)
    domains, atlas_systems = [], []
    for _ in range(16):
        batch, task, history, queries, meta = sampler.next()
        domains.append(meta['domain'])
        assert (task, history, queries, batch.frames) == ('static', 1, [], 1)
        assert batch.time_ps.tolist() == [[0.0]]
        if meta['domain'] == 'protein':
            continue
        record = stores[meta['domain']][meta['source_store_index']]
        frame = meta['source_frame_index']
        assert record.get('frame_mask', np.ones(record['x'].shape[0], dtype=bool))[frame]
        assert meta['source_sample_id'] == record['sample_id']
        np.testing.assert_array_equal(batch.x.numpy(), record['x'][frame:frame + 1])
        np.testing.assert_array_equal(batch.bpos.numpy(), record['bpos'][frame:frame + 1])
        for key in ('atype', 'btype', 'block_id', 'component_id', 'atom_source_index', 'bond_index', 'loss_mask', 'align_mask'):
            np.testing.assert_array_equal(getattr(batch, key).numpy(), record[key])
        if meta['domain'] == 'atlas':
            atlas_systems.append(meta['system'])
    assert Counter(domains) == dict(protein=8, atlas=4, misato=4)
    # The 1-window and 3-window systems receive the same number of visits.
    assert set(Counter(atlas_systems).values()) == {2}
    for name, records in stores.items():
        for before, after in zip(untouched[name], records):
            np.testing.assert_array_equal(before['x'], after['x'])
            np.testing.assert_array_equal(before['bpos'], after['bpos'])


def test_mid_cycle_checkpoint_replays_every_choice_without_global_rng(inventory):
    settings, _ = inventory
    sampler = BridgeMixture(settings, 71)
    for _ in range(7):
        sampler.next()
    state = copy.deepcopy(sampler.state_dict())
    global_python = random.getstate()
    global_numpy = copy.deepcopy(np.random.get_state())
    global_torch = torch.get_rng_state().clone()
    expected = [_signature(sampler.next()) for _ in range(25)]
    restored = BridgeMixture(settings, 999)
    restored.load_state_dict(copy.deepcopy(state))
    assert [_signature(restored.next()) for _ in range(25)] == expected
    assert restored.state_dict() == sampler.state_dict()
    assert random.getstate() == global_python
    assert np.random.get_state()[0] == global_numpy[0]
    np.testing.assert_array_equal(np.random.get_state()[1], global_numpy[1])
    assert np.random.get_state()[2:] == global_numpy[2:]
    assert torch.equal(torch.get_rng_state(), global_torch)


def test_old_static_protein_source_order_and_rng_are_unchanged(inventory):
    settings, stores = inventory
    settings = {**settings, 'sixteen_step_tasks': ['static:protein'] * 16}
    sampler = BridgeMixture(settings, 71)
    legacy_rng = random.Random(71 + 71007)
    pending, order = [], []
    for _ in range(35):
        if not pending:
            pending = list(settings['sixteen_step_tasks'])
            legacy_rng.shuffle(pending)
        pending.pop()
        if not order:
            order = sorted(r['sample_id'] for r in stores['pdb'])
            legacy_rng.shuffle(order)
        expected = order.pop()
        result = sampler.next()
        assert result[-1]['system'] == expected
        assert result[0].sample_id == (expected,)
        assert sampler.rng.getstate() == legacy_rng.getstate()


def test_nontrain_snapshot_is_rejected(inventory):
    settings, stores = inventory
    for record in stores['misato']:
        record['split'] = 'val'
    sampler = BridgeMixture({**settings, 'sixteen_step_tasks': ['static:misato']}, 71)
    with pytest.raises(ValueError, match='Non-TRAIN'):
        sampler.next()
