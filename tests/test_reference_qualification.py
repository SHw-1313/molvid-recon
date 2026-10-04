import numpy as np
import pytest
from molvid.evaluation.reference_qualification import correlation_summary


def test_reference_ess_is_scalar_bounded_and_marks_constants():
    rng=np.random.default_rng(42);white=rng.normal(size=4000)
    slow=np.empty_like(white);slow[0]=white[0]
    for i in range(1,len(slow)): slow[i]=.98*slow[i-1]+white[i]
    a,b=correlation_summary(white),correlation_summary(slow)
    assert 0<b['ess']<a['ess']<=4000
    assert len(b['four_contiguous_block_means'])==4
    assert correlation_summary(np.ones(100))['ess'] is None
    with pytest.raises(ValueError): correlation_summary(np.array([np.nan]*100))
