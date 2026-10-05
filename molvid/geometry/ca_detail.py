"""Protein-only residue anchors and an explicit observed-only detail condition."""
from dataclasses import dataclass, replace
import torch
from torch import Tensor


@dataclass(frozen=True)
class CADetailCondition:
    observed: Tensor                 # [H,N,3], Angstrom
    observed_time_ps: Tensor         # [H]
    query_time_ps: Tensor            # [Q]
    atom_type: Tensor                # [N]
    residue_type: Tensor             # [N]
    ca_index: Tensor                 # [R], actual model atom indices
    residue_index: Tensor            # [N], packed residue index into ca_index
    edges: Tensor                   # [2,E], fixed covalent + atom/CA
    bond_index: Tensor               # [2,B], actual covalent bonds

    def to(self,device):
        return replace(self,**{name:getattr(self,name).to(device) for name in self.__dataclass_fields__})


def condition_from_record(record,history=4):
    """Never read future x/bpos; identities and graph must be static metadata."""
    ids=record['atom_identity'];n=len(ids)
    component=torch.as_tensor(record['component_id']).long()
    if bool((component!=0).any()): raise ValueError('CA pilot supports proteins only; ligands need a separate anchor contract')
    blocks=torch.as_tensor(record['block_id']).long()
    unique,inverse=torch.unique(blocks,sorted=True,return_inverse=True)
    names=[str(s).rsplit(':',2)[-2].strip().upper() for s in ids]
    ca=[]
    for block in unique:
        indices=[i for i in torch.where(blocks==block)[0].tolist() if names[i]=='CA']
        if len(indices)!=1: raise ValueError('each residue must have exactly one CA atom')
        ca.append(indices[0])
    ca=torch.tensor(ca,dtype=torch.long)
    obs=torch.as_tensor(record['x'][:history]).float().clone()
    if not torch.allclose(obs[:,ca[inverse]],torch.as_tensor(record['bpos'][:history]),atol=1e-5,rtol=0):
        raise ValueError('CA mapping disagrees with observed residue anchors')
    bond=torch.as_tensor(record['bond_index']).long()
    atoms=torch.arange(n);nonca=atoms!=ca[inverse]
    star=torch.stack((atoms[nonca],ca[inverse][nonca]))
    edges=torch.cat((bond,bond.flip(0),star,star.flip(0)),1).T.unique(dim=0).T.contiguous()
    times=torch.as_tensor(record['time_ps']).float()
    return CADetailCondition(obs,times[:history].clone(),times[history:].clone(),
        torch.as_tensor(record['atype']).long(),torch.as_tensor(record['btype']).long(),
        ca,inverse,edges,bond)


def rigid_transport(condition,ca):
    if ca.shape!=(len(condition.query_time_ps),len(condition.ca_index),3):
        raise ValueError('CA query shape mismatch')
    ref=condition.observed[-1]
    offset=ref-ref[condition.ca_index][condition.residue_index]
    xyz=ca[:,condition.residue_index]+offset
    # Exact anchors, also if low-precision roundoff would affect zero offsets.
    xyz[:,condition.ca_index]=ca
    return xyz
