from copy import deepcopy
import pytest
from tools.run_data_scale_arm import check_input_audit


@pytest.mark.parametrize('change',['none','pending','partial','short_schedule','no_actual_reads','wrong_seed'])
def test_require_actual_full_input_audit(change):
    report=dict(status='completed',model_calls=0,training_updates=0,protected_payloads_read=0,
        seeds={'71':dict(systems=512,records=184320,selected_updates=8192,real_batches_decoded=11)})
    report=deepcopy(report)
    if change=='pending':report['status']='waiting'
    if change=='partial':report['seeds']['71']['systems']=436
    if change=='short_schedule':report['seeds']['71']['selected_updates']=2048
    if change=='no_actual_reads':report['seeds']['71']['real_batches_decoded']=0
    if change=='wrong_seed':
        with pytest.raises(KeyError):check_input_audit(report,72)
    elif change=='none':assert check_input_audit(report,71)['systems']==512
    else:
        with pytest.raises(ValueError):check_input_audit(report,71)
