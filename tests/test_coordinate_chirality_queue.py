from copy import deepcopy
import pytest

from tools.run_coordinate_chirality_training import qualify_pair, qualified_profiles


def pair():
    a=dict(coordinate_training=dict(handedness_weight=0.,handedness_every=8,
        handedness_steps=16,total_updates=2048,dit_lr=5e-5),coordinate_schedule_offset=2048,
        coordinate_initialization=dict(checkpoint='/parent',sha256='same'),data={'train':'same'})
    b=deepcopy(a); b['coordinate_training']['handedness_weight']=.1
    return a,b


def test_exact_weight_only_pair():
    qualify_pair(*pair())


@pytest.mark.parametrize('change',['parent','data','lr','cadence','solver','offset','zero_weight'])
def test_reject_confounding_or_unregistered_pair(change):
    a,b=pair()
    if change=='parent': b['coordinate_initialization']['sha256']='different'
    elif change=='data': b['data']['train']='different'
    elif change=='lr': b['coordinate_training']['dit_lr']=1e-4
    elif change=='cadence':
        a['coordinate_training']['handedness_every']=b['coordinate_training']['handedness_every']=1
    elif change=='solver':
        a['coordinate_training']['handedness_steps']=b['coordinate_training']['handedness_steps']=4
    elif change=='offset': a['coordinate_schedule_offset']=b['coordinate_schedule_offset']=0
    else: b['coordinate_training']['handedness_weight']=0.
    with pytest.raises(ValueError): qualify_pair(a,b)


def test_both_actual_profiles_required():
    status=dict(status='completed',optimizer_updates_discarded=2,
        profiles=[dict(seed=seed,profile_updates=1) for seed in (71,72)])
    qualified_profiles(status)
    status['profiles'][1]['seed']=71
    with pytest.raises(ValueError): qualified_profiles(status)
