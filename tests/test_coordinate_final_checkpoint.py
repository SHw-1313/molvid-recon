"""CPU metadata gates only; no neural loading or training."""
import pytest
from tools.evaluate_expanded_native import require_final_coordinate_checkpoint


@pytest.mark.parametrize('key',['largest_nominal_batch_profile_only','largest_query_atom_frame_profile_only'])
def test_fixed_final_accepts_explicit_old_or_new_nonprofile_contract(key):
    require_final_coordinate_checkpoint(dict(cursor=2048,contract={key:False}))
    for cursor,flag in ((1024,False),(17,False),(2048,True),(2048,0)):
        with pytest.raises(ValueError,match='fixed-final2048'):
            require_final_coordinate_checkpoint(dict(cursor=cursor,contract={key:flag}))


def test_profile_status_missing_or_ambiguous_is_rejected():
    for contract in ({},{'largest_nominal_batch_profile_only':False,'largest_query_atom_frame_profile_only':False}):
        with pytest.raises(ValueError,match='fixed-final2048'):
            require_final_coordinate_checkpoint(dict(cursor=2048,contract=contract))
