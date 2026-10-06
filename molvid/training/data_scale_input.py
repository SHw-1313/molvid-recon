"""Read a frozen, qualified TRAIN512 schedule without re-sampling or cropping.

This input boundary changes no model arithmetic. The caller retains ownership
of the base192 store; this context owns only added shard handles. All generated
records still enter the ordinary ClipBatch/coordinate-update path.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from ..data.sampling import get_clip_specs
from ..data.store import ClipMMapDataset
from ..runtime import canonical_hash, sha256_file
from .execution import ExecutionEntry, execution_manifest
from .predictive_data import group_identity


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(path, digest):
    path = Path(path)
    if not path.is_absolute() or sha256_file(path) != digest:
        raise ValueError(f'absolute SHA-bound input required: {path}')
    return path


def validate_reference(raw):
    settings = raw['coordinate_data_scale']
    reference = bound_file(settings['reference_recipe'], settings['reference_recipe_sha256'])
    if {k:v for k,v in raw.items() if k != 'coordinate_data_scale'} != read(reference):
        raise ValueError('data expansion must be the only recipe change')
    if (raw['coordinate_schedule_offset'] != 2048 or raw['coordinate_training']['total_updates'] != 8192
            or not raw['coordinate_train_query_geometry'] or raw['coordinate_training']['handedness_weight'] != 0):
        raise ValueError('registered RF-only query long-budget control required')


def require_cuda_gate(settings):
    folder = Path(settings['cuda_gate'])
    bound_file(folder/'status.json', settings['cuda_gate_status_sha256'])
    bound_file(folder/'run_identity.json', settings['cuda_gate_identity_sha256'])
    state = read(folder/'status.json')
    if (state.get('status') != 'completed' or state.get('actual_discarded_updates') != 6
            or state.get('exact_resume_both_depths') is not True):
        raise ValueError('actual uncropped H4/H8 CUDA+resume qualification required')
    runtime = Path(__file__).resolve().parents[2]
    for name,digest in read(folder/'run_identity.json')['source_files_sha256'].items():
        if name.startswith('molvid/') and sha256_file(runtime/name) != digest:
            raise ValueError('CUDA-qualified numerical model/runtime differs')
    audit = Path(settings['audit'])
    bound_file(audit/'summary.json',settings['audit_summary_sha256'])
    maximum = read(audit/'summary.json')['combined']['max_atoms']
    for depth in (4,8):
        parity = read(folder/f'depth{depth}/resume_parity.json')
        if parity.get('exact') is not True or parity.get('actual_updates') != 3:
            raise ValueError('whole-state depth-specific resume parity missing')
        for label,digest in zip(('continuous','resume'),parity['checkpoint_sha256'],strict=True):
            bound_file(folder/f'depth{depth}'/label/'step_2.pt',digest)
            evidence = read(folder/f'depth{depth}'/label/'summary.json')
            if (evidence.get('status') != 'completed' or evidence.get('atoms') != maximum
                    or evidence.get('no_crop') is not True or evidence.get('depth') != depth
                    or evidence.get('updates') != (2 if label == 'continuous' else 1)):
                raise ValueError('actual full-inventory maximum was not exercised')


def validate_inventory(summary, snapshot, specs):
    if (summary.get('status') != 'completed' or summary.get('full512_inventory') is not True
            or summary.get('old_prefix_indices_and_specs_exact') is not True
            or summary.get('shard_local_fixed_views_reconstructed') is not True
            or summary['combined']['systems'] != 512):
        raise ValueError('full TRAIN512 real-record concat qualification required')
    added = [s for row in snapshot for s in row['systems']]
    names = [s.sample_id for s in specs]
    systems = {group_identity(s)[0] for s in names}
    if (len(added) != 320 or len(set(added)) != 320 or len(systems) != 512
            or len(names) != len(set(names)) or len(names) != summary['combined']['records']
            or canonical_hash(names) != summary['combined']['sample_identity_sha256']):
        raise ValueError('closed inventory/order/sample identities differ')


def resolve_entries(manifest, reference, specs, summary):
    rows = tuple(ExecutionEntry(**dict(row,indices=tuple(row['indices']))) for row in manifest['entries'])
    if (len(rows) != 8192 or len(reference) != 8192
            or canonical_hash(execution_manifest(rows)) != canonical_hash(manifest)
            or manifest['sha256'] != summary['matched_sha256']
            or execution_manifest(reference)['sha256'] != summary['reference_sha256']):
        raise ValueError('fixed8192 entry manifest or reference changed')
    for a,b in zip(reference,rows):
        if ((a.view,a.time_bucket_id,a.history_frames,len(a.indices),a.auxiliary)
                != (b.view,b.time_bucket_id,b.history_frames,len(b.indices),b.auxiliary)):
            raise ValueError('per-update physical-time/history/packing calendar differs')
    if len({row.global_batch_index for row in rows}) != len(rows):
        raise ValueError('candidate batches repeated')
    by_index = {s.index:s for s in specs}
    if len(by_index) != len(specs) or set(by_index) != set(range(len(specs))):
        raise ValueError('noncanonical global dataset indices')
    systems = set()
    for row in rows:
        for index in row.indices:
            if index not in by_index:
                raise ValueError('schedule index outside qualified data')
            item = by_index[index]
            if item.frames != 16 or item.time_bucket_id != row.time_bucket_id or item.task_type != 'trajectory':
                raise ValueError('scheduled record/time-view metadata differs')
            systems.add(group_identity(item.sample_id)[0])
    if len(systems) != 512 or systems != set(summary['selected']['system_clip_exposure']):
        raise ValueError('schedule must cover every registered training system')
    return rows


@contextmanager
def expanded_training_input(base, base_hash, raw, seed, reference):
    """Validate full immutable bytes, return existing concatenation and calendar.

    The CUDA gate is checked separately before model construction by the trainer;
    CPU loader audits can call this function without claiming GPU qualification.
    It never opens validation/reserve/sealed coordinates or launches a model.
    """
    validate_reference(raw)
    settings = raw['coordinate_data_scale']; audit = Path(settings['audit']); calendar = Path(settings['calendar'])
    pins = [(audit/'summary.json','audit_summary_sha256'),(audit/'run_identity.json','audit_identity_sha256'),
            (audit/'closed_shards_snapshot.json','snapshot_sha256'),(calendar/'summary.json','calendar_summary_sha256'),
            (calendar/'run_identity.json','calendar_identity_sha256'),(calendar/f'matched_{seed}.json','matched_manifest_sha256')]
    for path,key in pins:
        bound_file(path,settings[key])
    combined_hashes = {}
    for root in (audit,calendar):
        for path,digest in read(root/'run_identity.json')['input_sha256'].items():
            if path in combined_hashes and combined_hashes[path] != digest:
                raise ValueError('qualified input identities conflict')
            combined_hashes[path] = digest
    for path,digest in combined_hashes.items():
        bound_file(path,digest)
    old = read(audit/'summary.json'); matched = read(calendar/'summary.json')
    if (matched.get('status') != 'completed' or matched.get('full512_inventory') is not True
            or matched.get('training_updates') != 0 or matched.get('model_calls') != 0):
        raise ValueError('completed full512 calendar qualification required')
    if base_hash != raw['data']['expected_data_hash']:
        raise ValueError('base192 data identity changed')
    snapshot = read(audit/'closed_shards_snapshot.json'); stores = []
    try:
        for row in snapshot:
            store = ClipMMapDataset(row['directory']); stores.append(store)
            if len(store) != row['records']:
                raise ValueError('closed shard record count changed')
        data = torch.utils.data.ConcatDataset([base,*stores])
        specs = list(get_clip_specs(data))
        validate_inventory(old,snapshot,specs)
        manifest = read(calendar/f'matched_{seed}.json')
        entries = resolve_entries(manifest,reference,specs,matched['seeds'][str(seed)])
        digest = canonical_hash(dict(base_data_hash=base_hash,closed_shards=snapshot))
        contract = dict(policy=matched['policy'], settings=settings,
            reference_selected_sha256=matched['seeds'][str(seed)]['reference_sha256'],
            matched_selected_sha256=manifest['sha256'],full512_inventory=True,
            exact_per_step_calendar=True,exposure=matched['seeds'][str(seed)],
            interpretation='same updates/calendar; different systems/atom sizes/noise shapes/compute')
        yield data,specs,entries,manifest,digest,contract,[path for path,_ in pins]
    finally:
        for store in stores:
            store.close()
