"""Train-only external complexes, preserving the ordinary packed atom interface.

Half the reference update positions are replaced, not extra hidden updates.
Complex sampling is system-equal, then balances native 80/160/320 ps clocks.
No validation coordinates are opened here and no atoms are cropped.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from ..data.sampling import get_clip_specs
from ..data.store import ClipMMapDataset
from ..runtime import canonical_hash
from .execution import execution_manifest
from .predictive_data import group_identity


@contextmanager
def complex_training_input(base, base_hash, settings, seed, reference):
    root = Path(settings['data'])
    summary = json.loads((root/'summary.json').read_text())
    if summary['status'] != 'completed' or summary['test_payload_opened'] is not False:
        raise ValueError('completed train/validation conversion required')
    inventory = {r['system'] for r in summary['systems'] if r['split'] == 'train'}
    if len(inventory) != 128 or len(reference) != 2048:
        raise ValueError('this experiment uses128 complex systems and2048 updates')
    store = ClipMMapDataset(root/'train')
    try:
        dataset = ConcatDataset([base, store])
        specs = list(get_clip_specs(dataset))
        by_system = defaultdict(lambda: defaultdict(list))
        for spec in specs[len(base):]:
            system = spec.sample_id.split('_')[1]
            if system not in inventory or spec.frames != 16:
                raise ValueError('unexpected complex training record')
            by_system[system][spec.time_bucket_id].append(spec.index)
        buckets = ('dt_80ps', 'dt_160ps', 'dt_320ps')
        if set(by_system) != inventory or any(set(x) != set(buckets) for x in by_system.values()):
            raise ValueError('missing native clock or training system')
        rng = np.random.default_rng(seed + 61007)
        order = rng.permutation(sorted(inventory)).tolist()
        entries = list(reference)
        complex_exposure = defaultdict(int)
        for ordinal, position in enumerate(range(1, len(entries), 2)):
            cycle, system_position = divmod(ordinal, len(order))
            system = order[system_position]
            bucket = buckets[(cycle + system_position) % len(buckets)]
            options = by_system[system][bucket]
            index = options[int(rng.integers(len(options)))]
            # H4/H8 alternate per system across its visits, independently of
            # atom count. Clock is actual stored ps, never re-labelled.
            history = (4, 8)[(cycle + system_position) % 2]
            entries[position] = replace(entries[position], indices=(index,),
                view='ordinary', time_bucket_id=bucket, history_frames=history,
                auxiliary=False)
            complex_exposure[system] += 1
        proteins = sorted({group_identity(specs[i].sample_id)[0]
            for entry in entries[::2] for i in entry.indices})
        if len(proteins) != 192:
            raise ValueError('retained protein updates do not cover192 systems')
        manifest = execution_manifest(entries)
        contract = dict(protein_systems=proteins, complex_systems=sorted(inventory),
            protein_updates=1024, complex_updates=1024, complex_exposure=dict(complex_exposure),
            policy='replace odd reference updates; system-equal complex then balanced native clocks',
            no_crop=True, data=str(root), test_payload_opened=False,
            interpretation='different domain, clocks, atom counts and compute; same total optimizer updates')
        yield dataset, specs, tuple(entries), manifest, canonical_hash(dict(
            base=base_hash, complex_summary=summary, schedule=manifest['sha256'])), contract
    finally:
        store.close()
