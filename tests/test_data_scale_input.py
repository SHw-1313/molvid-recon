from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from molvid.runtime import sha256_file
from molvid.training.data_scale_input import validate_reference,require_cuda_gate,resolve_entries
import molvid.training.data_scale_input as input_module
from molvid.training.execution import ExecutionEntry,execution_manifest
from molvid.data.sampling import ClipItemSpec


def test_only_data_recipe_may_change(tmp_path):
    raw=dict(coordinate_schedule_offset=2048,coordinate_train_query_geometry=True,
        coordinate_training=dict(total_updates=8192,handedness_weight=0),model=dict(depth=4))
    path=tmp_path/'reference.json';path.write_text(json.dumps(raw))
    raw['coordinate_data_scale']=dict(reference_recipe=str(path),reference_recipe_sha256=sha256_file(path))
    validate_reference(raw)
    changed=deepcopy(raw);changed['model']['depth']=8
    with pytest.raises(ValueError,match='only recipe change'):
        validate_reference(changed)


@pytest.mark.parametrize('state',[dict(status='waiting'),dict(status='failed'),
    dict(status='completed',actual_discarded_updates=0,exact_resume_both_depths=True),
    dict(status='completed',actual_discarded_updates=6,exact_resume_both_depths=False)])
def test_cpu_or_queued_state_cannot_qualify_gpu(tmp_path,state):
    status=tmp_path/'status.json';status.write_text(json.dumps(state))
    identity=tmp_path/'run_identity.json';identity.write_text('{}')
    settings=dict(cuda_gate=str(tmp_path),cuda_gate_status_sha256=sha256_file(status),
                  cuda_gate_identity_sha256=sha256_file(identity))
    with pytest.raises(ValueError,match='actual uncropped'):
        require_cuda_gate(settings)


def test_cuda_evidence_requires_whole_maximum_and_unchanged_runtime(tmp_path):
    from pathlib import Path
    runtime=Path(input_module.__file__).resolve().parents[2]
    name='molvid/flow/coordinate.py'
    status=tmp_path/'status.json';status.write_text(json.dumps(dict(status='completed',
        actual_discarded_updates=6,exact_resume_both_depths=True)))
    identity=tmp_path/'run_identity.json'
    identity.write_text(json.dumps(dict(source_files_sha256={name:sha256_file(runtime/name)})))
    audit=tmp_path/'audit';audit.mkdir()
    summary=audit/'summary.json';summary.write_text(json.dumps(dict(combined=dict(max_atoms=17002))))
    settings=dict(cuda_gate=str(tmp_path),cuda_gate_status_sha256=sha256_file(status),
        cuda_gate_identity_sha256=sha256_file(identity),audit=str(audit),audit_summary_sha256=sha256_file(summary))
    for depth in (4,8):
        hashes=[]
        for label in ('continuous','resume'):
            folder=tmp_path/f'depth{depth}'/label;folder.mkdir(parents=True)
            checkpoint=folder/'step_2.pt';checkpoint.write_bytes(b'synthetic gate fixture; not GPU evidence')
            hashes.append(sha256_file(checkpoint))
            (folder/'summary.json').write_text(json.dumps(dict(status='completed',atoms=17002,
                no_crop=True,depth=depth,updates=2 if label=='continuous' else 1)))
        (tmp_path/f'depth{depth}/resume_parity.json').write_text(json.dumps(dict(exact=True,
            actual_updates=3,checkpoint_sha256=hashes)))
    require_cuda_gate(settings)
    # A smaller actual probe cannot qualify a larger training inventory.
    summary.write_text(json.dumps(dict(combined=dict(max_atoms=18000))))
    settings['audit_summary_sha256']=sha256_file(summary)
    with pytest.raises(ValueError,match='full-inventory maximum'):
        require_cuda_gate(settings)


def fixture():
    specs=[ClipItemSpec(i,10,16,'trajectory','dt_100ps',sample_id=f'atlas_s{i}_A_R1_dt_100ps_w000000')
           for i in range(512)]
    rows=tuple(ExecutionEntry(i,(i%512,),'ordinary','dt_100ps',4,False) for i in range(8192))
    manifest=json.loads(json.dumps(execution_manifest(rows)))
    summary=dict(matched_sha256=manifest['sha256'],reference_sha256=manifest['sha256'],
        selected=dict(system_clip_exposure={f'atlas_s{i}_A':16 for i in range(512)}))
    return specs,rows,manifest,summary


def test_full_manifest_load_preserves_every_entry_and_global_rng():
    specs,rows,manifest,summary=fixture();before=np.random.get_state()
    assert resolve_entries(manifest,rows,specs,summary)==rows
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


@pytest.mark.parametrize('change',['calendar','outside','duplicate','partial','metadata'])
def test_changed_calendar_indices_inventory_fail_closed(change):
    specs,reference,manifest,summary=fixture();rows=list(reference)
    if change=='calendar':rows[0]=replace(rows[0],history_frames=8)
    if change=='outside':rows[0]=replace(rows[0],indices=(512,))
    if change=='duplicate':rows[0]=replace(rows[0],global_batch_index=1)
    if change=='partial':rows=[replace(r,indices=(0,)) for r in rows]
    if change=='metadata':specs[0]=replace(specs[0],time_bucket_id='dt_400ps')
    manifest=execution_manifest(rows);summary['matched_sha256']=manifest['sha256']
    with pytest.raises(ValueError):
        resolve_entries(manifest,reference,specs,summary)


def test_repaired_calendar_reuses_only_completed_full_concat(tmp_path):
    from tools.audit_data_scale_training_input import qualified_locations
    concat=tmp_path/'old_concat';calendar=tmp_path/'new_calendar'
    for folder in (concat,calendar):
        folder.mkdir();(folder/'summary.json').write_text(json.dumps(dict(status='completed',full512_inventory=True)))
    assert qualified_locations(calendar,dict(status='completed'),concat)==(concat,calendar)
    with pytest.raises(ValueError):qualified_locations(calendar,dict(status='waiting'),concat)
    (concat/'summary.json').write_text(json.dumps(dict(status='completed',full512_inventory=False)))
    with pytest.raises(ValueError):qualified_locations(calendar,dict(status='completed'),concat)
    assert qualified_locations(tmp_path,dict(status='completed',full512_cpu_qualified=True))==(tmp_path/'concat',tmp_path/'calendar')
