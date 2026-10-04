import pytest
from molvid.evaluation.static_response import paired_comparison


def test_pairing_uses_system_not_atom_or_draw_count():
    first=[];second=[]
    for system,count,delta in [('small',1,1.),('large',3,3.)]:
        for seed in range(count):
            common=dict(system=system,sample_id=system,variant='generated',seed=seed)
            first.append(dict(common,metrics={'bond_rmse':2.,'missing':{'available':False,'value':0.}}))
            second.append(dict(common,metrics={'bond_rmse':2.+delta,'missing':{'available':False,'value':9.}}))
    result=paired_comparison(first,second)
    assert result['bond_rmse']['delta_second_minus_first']==pytest.approx(2.)
    assert result['bond_rmse']['systems']==2
    assert result['bond_rmse']['paired_rows']==4
    assert 'missing.value' not in result
    with pytest.raises(ValueError): paired_comparison(first,second[:-1])
    with pytest.raises(ValueError): paired_comparison(first,second+[second[0]])
