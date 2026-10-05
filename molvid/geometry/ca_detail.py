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


def ca_frame_transport(condition,ca):
    """Parameter-free local-frame rotation control, defined by CA graph only.

    Interior residues use their two chain neighbors; termini use two-hop CA
    neighbors. Degenerate/missing frames explicitly fall back to translation.
    This is a diagnostic baseline, not a trainable detail generator.
    """
    c=condition;r=len(c.ca_index);neighbors=[set() for _ in range(r)]
    for a,b in c.residue_index[c.bond_index].detach().cpu().T.tolist():
        if a!=b:neighbors[a].add(b);neighbors[b].add(a)
    pairs=[];supported=[]
    for i,first in enumerate(neighbors):
        candidates=sorted(first)
        if len(candidates)==1:
            candidates+=sorted(neighbors[candidates[0]]-{i}-set(candidates))[:1]
        ok=len(candidates)==2
        pairs.append(candidates[:2] if ok else [i,i]);supported.append(ok)
    pair=torch.tensor(pairs,device=ca.device).T
    supported=torch.tensor(supported,device=ca.device,dtype=torch.bool)
    def frames(x):
        u=x[...,pair[0],:]-x;v=x[...,pair[1],:]-x
        axis=torch.nn.functional.normalize(u,dim=-1,eps=1e-6)
        normal=torch.cross(axis,v,dim=-1)
        valid=(u.norm(dim=-1)>1e-5)&(normal.norm(dim=-1)>1e-5)&supported
        normal=torch.nn.functional.normalize(normal,dim=-1,eps=1e-6)
        second=torch.cross(normal,axis,dim=-1)
        return torch.stack((axis,second,normal),-1),valid
    ref=c.observed[-1];reference_frame,reference_valid=frames(ref[c.ca_index])
    target_frame,target_valid=frames(ca)
    parent=c.residue_index;offset=ref-ref[c.ca_index][parent]
    local=torch.einsum('ni,nij->nj',offset,reference_frame[parent])
    rotated=torch.einsum('ni,qnki->qnk',local,target_frame[:,parent])
    valid=target_valid[:,parent]&reference_valid[parent][None]
    xyz=ca[:,parent]+torch.where(valid[...,None],rotated,offset[None])
    xyz[:,c.ca_index]=ca
    return xyz,dict(frame_supported_residues=int(supported.sum()),total_residues=r,
        fallback_fraction=float((~valid).float().mean()))
