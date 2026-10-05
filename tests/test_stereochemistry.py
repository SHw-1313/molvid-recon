import torch
from molvid.evaluation.stereochemistry import ca_handedness


def test_ca_sign_detects_mirror_but_not_rotation_or_translation():
    x=torch.tensor([[[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]]])
    identity=[f'protein:{name}:C' for name in ('CA','N','C','CB')]
    kwargs=dict(observed=x,atom_identity=identity,block_id=[0,0,0,0])
    original=ca_handedness(x,**kwargs)
    assert original['ca_handedness_flip_fraction']==0
    mirror=x.clone();mirror[...,0]*=-1
    assert ca_handedness(mirror,**kwargs)['ca_handedness_flip_fraction']==1
    rotation=torch.tensor([[.36,.48,-.8],[-.8,.6,0.],[.48,.64,.6]])
    assert ca_handedness(x@rotation+3,**kwargs)['ca_handedness_flip_fraction']==0
    planar=x.clone();planar[...,2]*=0
    assert ca_handedness(planar,**kwargs)['ca_near_planar_fraction']==1
