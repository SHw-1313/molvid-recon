import pytest
from tools.compare_expanded_native import METRICS, paired_rows


def row(system,sample,seed,value):
    return dict(system=system,sample_id=sample,seed=seed,metrics={k:value for k in METRICS})


def test_paired_system_weighting_and_coverage():
    a=[row('A','a',1,2),row('A','a',2,4),row('B','b',1,9)]
    b=[row('A','a',1,1),row('A','a',2,1),row('B','b',1,1)]
    out=paired_rows(a,b,{'A','B'})
    assert out['paired_draws']==3
    assert out['delta_left_minus_right']['bond_rmse']['mean']==5
    assert out['delta_left_minus_right']['bond_rmse']['per_system']=={'A':2.,'B':8.}
    with pytest.raises(ValueError,match='unpaired'):
        paired_rows(a,b[:-1],{'A','B'})
    with pytest.raises(ValueError,match='duplicate'):
        paired_rows(a+a[:1],b,{'A','B'})
