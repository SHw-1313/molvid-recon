"""Observed-reference protein C-alpha handedness (not all-atom R/S assignment)."""
from collections import defaultdict
import torch


def ca_handedness(predicted, *, observed, atom_identity, block_id):
    """Coordinates [T,N,3] Angstrom; chirality reference uses observed atoms only.

    Excludes glycine/missing N,C,CB; near-planar references cannot define a sign.
    This complements bond/clash scores, which cannot detect mirror inversion.
    """
    groups=defaultdict(dict)
    for index,(identity,block) in enumerate(zip(atom_identity,torch.as_tensor(block_id).cpu().tolist())):
        name=str(identity).rsplit(':',2)[-2].strip().upper()
        if name in groups[block]: raise ValueError('ambiguous residue atom identity')
        groups[block][name]=index
    indices=[tuple(atoms[name] for name in ('CA','N','C','CB'))
        for _,atoms in sorted(groups.items()) if all(name in atoms for name in ('CA','N','C','CB'))]
    if not indices: return dict(ca_chiral_centers=0)
    center,a,b,c=torch.tensor(indices,device=predicted.device).T
    def volume(x):
        return (torch.cross(x[...,a,:]-x[...,center,:],x[...,b,:]-x[...,center,:],dim=-1)*
                (x[...,c,:]-x[...,center,:])).sum(-1)
    ref=volume(observed[-1].to(predicted));valid=ref.abs()>.2
    if not valid.any(): return dict(ca_chiral_centers=0)
    value=volume(predicted)[:,valid];reference=ref[valid]
    return dict(ca_chiral_centers=int(valid.sum()),
        ca_handedness_flip_fraction=float((value*reference<0).float().mean()),
        ca_near_planar_fraction=float((value.abs()<.1*reference.abs()).float().mean()),
        ca_signed_volume_relative_mae=float(((value-reference)/reference.abs()).abs().mean()))
