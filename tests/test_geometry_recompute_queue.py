import copy
import json

import pytest

from tools.queue_geometry_recompute import completed_baseline, paired_execution, require_cuda_tests


def test_full_original_queue_terminal_required(tmp_path):
    rows = []
    for mode in ('frozen_query','trainable_query'):
        for history in (4,8):
            name = f'{mode}_H{history}'; destination = tmp_path/name; destination.mkdir()
            row = dict(case=name,status='failed',error='retained diagnostic OOM')
            (destination/'comparison.json').write_text(json.dumps(row)); rows.append(row)
    status = tmp_path/'status.json'
    status.write_text(json.dumps(dict(status='needs_remediation',cases=rows)))
    assert completed_baseline(tmp_path)['status'] == 'needs_remediation'
    status.write_text(json.dumps(dict(status='running',cases=rows)))
    with pytest.raises(ValueError, match='terminated'): completed_baseline(tmp_path)
    status.write_text(json.dumps(dict(status='needs_remediation',cases=rows[:-1])))
    with pytest.raises(ValueError, match='four terminal'): completed_baseline(tmp_path)


@pytest.mark.parametrize('body', ['', '<skipped/>', '<failure/>', '<error/>'])
def test_cuda_junit_does_not_accept_skips_or_failures(tmp_path,body):
    path = tmp_path/'result.xml'
    path.write_text('<testsuites><testsuite>'+f'<testcase>{body}</testcase>'*5+'</testsuite></testsuites>')
    if body:
        with pytest.raises(ValueError, match='five actual CUDA'): require_cuda_tests(path)
    else:
        require_cuda_tests(path)


def test_exact_execution_pair_does_not_hide_weight_rng_or_optimizer_changes():
    control = dict(recompute_geometry=False,model_sha256='model',optimizer_sha256='optimizer',
        rng_sha256='rng',sample_id='sample',atoms=1578,history_frames=4,query_frames=12,
        cursor_end=2,updates=[dict(peak_allocated_bytes=100)],seconds=2.)
    candidate = dict(control,recompute_geometry=True,updates=[dict(peak_allocated_bytes=50)])
    assert paired_execution(control,candidate)['recompute_peak_bytes'] == 50
    for field in ('model_sha256','optimizer_sha256','rng_sha256','sample_id','history_frames'):
        bad = copy.deepcopy(candidate); bad[field] = 'different'
        with pytest.raises(ValueError, match='altered actual two-update'): paired_execution(control,bad)
