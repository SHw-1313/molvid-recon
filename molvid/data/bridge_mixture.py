"""System-balanced TRAIN sampling through the common molecular record API.

Dataset-specific IDs are resolved here, not inside the neural architecture.
Only existing ordinary native-clock stores and registered static TRAIN records
are read. Each shuffled system cycle visits every system before repetition.
"""
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import random

from .batch import collate_clip_records
from .sampling import get_clip_specs
from .static import trajectory_record_to_static
from .store import ClipMMapDataset
from ..training.predictive_data import group_identity


class BridgeMixture:
    def __init__(self, settings, seed):
        self.stack = ExitStack()
        self.rng = random.Random(seed + 71007)
        self.groups = {}
        self.orders, self.visits = {}, defaultdict(int)
        self.counts = defaultdict(int)
        self.pending = []
        self.settings = settings
        protein_paths = [settings['protein_base']]
        shards = json.loads(Path(settings['protein_shards']).read_text())
        protein_paths += [r['directory'] for r in shards]
        for domain, paths in [('protein', protein_paths), ('complex', [settings['complex_train']])]:
            grouped = defaultdict(lambda: defaultdict(list))
            for path in paths:
                store = self.open(path)
                for spec in get_clip_specs(store):
                    if spec.frames != 16 or not spec.time_bucket_id.startswith('dt_'):
                        continue
                    system = (group_identity(spec.sample_id)[0] if domain == 'protein'
                              else spec.sample_id.split('_')[1])
                    grouped[system][spec.time_bucket_id].append((store, spec.index))
            self.groups[domain] = dict(grouped)
        registration = Path(settings['static_registration'])
        rows = json.loads((registration/'registered_sources.json').read_text())
        identity = json.loads((registration/'run_identity.json').read_text())
        paths = [Path(p).parent for p in identity['input_sha256'] if Path(p).name == 'data.bin']
        if len(paths) != 1:
            raise ValueError('Static registration must name its one existing source store')
        store = self.open(paths[0])
        self.static = {r['source_sample_id']: (store, r['diagnostic_store_index']) for r in rows}
        for domain, expected in settings['expected_systems'].items():
            actual = len(self.static if domain == 'static_protein' else self.groups[domain])
            if actual != expected:
                raise ValueError(f'{domain}: expected {expected}, found {actual}')

    def open(self, path):
        store = ClipMMapDataset(path)
        self.stack.callback(store.close)
        return store

    def close(self):
        self.stack.close()

    def next_system(self, key, systems):
        if not self.orders.get(key):
            self.orders[key] = sorted(systems)
            self.rng.shuffle(self.orders[key])
        system = self.orders[key].pop()
        visit = self.visits[(key, system)]
        self.visits[(key, system)] += 1
        return system, visit

    def next(self):
        if not self.pending:
            self.pending = list(self.settings['sixteen_step_tasks'])
            self.rng.shuffle(self.pending)
        key = self.pending.pop()
        task, domain = key.split(':')
        if key == 'static:protein':
            system, visit = self.next_system(key, self.static)
            store, index = self.static[system]
            history = 1
        else:
            systems = self.groups[domain]
            system, visit = self.next_system(key, systems)
            buckets = sorted(systems[system], key=lambda x: float(x.split('_')[1][:-2]))
            # Six dynamic visits cover all 3 clocks x 2 history lengths.
            bucket = buckets[(visit // 2) % len(buckets)]
            store, index = self.rng.choice(systems[system][bucket])
            history = (4, 8)[visit % 2]
        record = store[index]
        if record['split'] != 'train':
            raise ValueError('Non-TRAIN record reached training')
        if task == 'static' and record['x'].shape[0] > 1:
            record = trajectory_record_to_static(record, frame_index=self.rng.randrange(record['x'].shape[0]))
            history = 1
        batch = collate_clip_records([record])
        queries = ([] if task == 'static' else sorted(self.rng.sample(
            range(history, batch.frames), self.settings['query_frames'])))
        self.counts[key] += 1
        return batch, task, history, queries, dict(system=system, visit=visit, domain=domain)

    def state_dict(self):
        return dict(rng=self.rng.getstate(), orders=self.orders, visits=dict(self.visits),
                    counts=dict(self.counts), pending=self.pending)

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.orders = state['orders']
        self.visits = defaultdict(int, state['visits'])
        self.counts = defaultdict(int, state['counts'])
        self.pending = state['pending']
