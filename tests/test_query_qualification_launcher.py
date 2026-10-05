import json
import pytest
from tools.qualify_query_geometry import completed_solver, qualification_environment


def test_deterministic_cublas_environment_is_set_before_child_start(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','invalid-inherited-value')
    value=qualification_environment('dedicated-test-device')
    assert value['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert value['CUDA_VISIBLE_DEVICES']=='dedicated-test-device'
    assert value['OMP_NUM_THREADS']=='2'


def test_completed_solver_requires_full_call_count_and_coverage(tmp_path):
    value=dict(status='completed',rows=192,systems=48,failures=[],coordinate_field_calls=12288)
    (tmp_path/'summary.json').write_text(json.dumps(value))
    completed_solver(tmp_path)
    for key,replacement in [('status','running'),('rows',190),('systems',47),
                            ('failures',['failure']),('coordinate_field_calls',3072)]:
        wrong=dict(value);wrong[key]=replacement
        (tmp_path/'summary.json').write_text(json.dumps(wrong))
        with pytest.raises(ValueError): completed_solver(tmp_path)
