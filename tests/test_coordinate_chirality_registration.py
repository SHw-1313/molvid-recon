import copy
import pytest

from tools.freeze_coordinate_chirality_training import calibrated_weight


def inputs():
    summary=dict(status='completed',optimizer_updates=0,model_state_bitwise_unchanged=True,
        systems=8,rf_backwards=16,actual_source_auxiliary_backwards=8)
    protocol=dict(training_only=True,test_opened=False,query_frames=12,history_frames=4,
        actual_source_steps=16,reference_solver_steps=16,systems=[f's{i}' for i in range(8)])
    rows=[dict(system=s,flow_time=t,generated_auxiliary={'loss':.01},generated16={'loss':.01},
        groups={group:{'static_to_dynamic_norm':.5} for group in ('dit','readout')})
        for s in protocol['systems'] for t in (.25,.75)]
    return summary,protocol,rows


def test_fixed_train_only_maximum_budget():
    summary,protocol,rows=inputs()
    assert calibrated_weight(summary,protocol,rows)==(.2,.5)
    rows[0]['groups']['dit']['static_to_dynamic_norm']=2.
    assert calibrated_weight(summary,protocol,rows)==(.05,2.)


@pytest.mark.parametrize('change',['short_query','short_solver','updates','missing','duplicate','nonfinite','no_signal','solver_difference'])
def test_reject_unqualified_calibration(change):
    summary,protocol,rows=inputs()
    if change=='short_query': protocol['query_frames']=2
    elif change=='short_solver': protocol['actual_source_steps']=4
    elif change=='updates': summary['optimizer_updates']=1
    elif change=='missing': rows.pop()
    elif change=='duplicate': rows[-1]=copy.deepcopy(rows[0])
    elif change=='nonfinite': rows[0]['groups']['dit']['static_to_dynamic_norm']=float('nan')
    elif change=='no_signal':
        for row in rows:
            for group in row['groups'].values(): group['static_to_dynamic_norm']=0
    else: rows[0]['generated_auxiliary']['loss']=.02
    with pytest.raises(ValueError): calibrated_weight(summary,protocol,rows)
