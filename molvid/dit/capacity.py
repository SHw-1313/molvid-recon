"""Function-preserving 4→8 depth expansion without extra geometry branches.

Call before creating an optimizer. Copies consume no RNG. Old blocks keep
their tensor objects and order at positions2/4/6/8; newly inserted residual
blocks start as identities, but their gates remain trainable.
"""
from copy import deepcopy

import torch
from torch import nn


def validate_coordinate_depth(value):
    if type(value) is not int or value not in (4,8):
        raise ValueError('registered coordinate depth must be integer4 or8')
    return value


def mapped_dit_name(name):
    if not name.startswith('blocks.'):
        return name
    _,index,rest=name.split('.',2)
    index=int(index)
    if not 0<=index<4: raise ValueError('parent block outside depth4')
    return f'blocks.{2*index+1}.{rest}'


@torch.no_grad()
def expand_coordinate_dit(dit):
    """Retain learned local-geometry count, resetting only NEW residual gates."""
    if dit.depth!=4 or len(dit.blocks)!=4:
        raise ValueError('only fresh depth4→8 expansion is supported')
    expected=tuple(i+1 for i,b in enumerate(dit.blocks) if b.local_geometry is not None)
    if expected!=tuple(dit.geometry_block_indices):
        raise ValueError('geometry branch inventory does not match declared layout')
    old_names=tuple(dit.state_dict())
    old_parameters=sum(p.numel() for p in dit.parameters())
    expanded=[]
    for old in dit.blocks:
        new=deepcopy(old,{id(old.local_geometry):None} if old.local_geometry is not None else {})
        # The depth experiment must not also double local-geometry modules.
        new.local_geometry=None
        for name in ('history_adaln','spatial_adaln','temporal_adaln','ffn_adaln'):
            modulation=getattr(new,name).modulation
            modulation.weight.zero_(); modulation.bias.zero_()
        if new.motion_context is not None:
            new.motion_context.scalar_residual_gate.zero_()
            new.motion_context.vector_residual_gate.zero_()
        new.history_adaln.vector_norm.scale.requires_grad_(False)
        expanded.extend((new,old))
    dit.blocks=nn.ModuleList(expanded)
    dit.depth=8
    dit.geometry_block_indices=tuple(2*i for i in expected)
    dit._layout_cache=None
    mapped={name:mapped_dit_name(name) for name in old_names}
    initialized=sorted(set(dit.state_dict())-set(mapped.values()))
    return dict(source_depth=4,target_depth=8,parent_blocks_one_based=[2,4,6,8],
        identity_blocks_one_based=[1,3,5,7],geometry_blocks_one_based=list(dit.geometry_block_indices),
        added_geometry_branches=0,old_to_new_state_keys=mapped,new_state_keys=initialized,
        dit_parameters_before=old_parameters,dit_parameters_after=sum(p.numel() for p in dit.parameters()),
        initialization='clone corresponding learned block, remove new G, zero new AdaLN/motion residuals',
        optimizer_must_be_created_after_expansion=True)
