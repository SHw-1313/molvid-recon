"""Equivariant CA-conditioned detail recovery, without future latent input."""
import torch
from torch import nn
from ..geometry.ca_detail import rigid_transport


class DetailMessage(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
        self.coordinate=nn.Linear(width,1,bias=False)
        nn.init.zeros_(self.coordinate.weight)

    def forward(self,h,xyz,reference,edges,ca_index,ca):
        source,dest=edges
        delta=xyz[:,source]-xyz[:,dest]
        distance=delta.square().sum(-1,keepdim=True)
        ref_distance=(reference[source]-reference[dest]).square().sum(-1)[None,:,None].expand(len(xyz),-1,-1)
        invariant=torch.cat((distance/25,ref_distance/25,(distance-ref_distance)/25),-1)
        message=self.message(torch.cat((h[:,source],h[:,dest],invariant),-1))
        degree=torch.bincount(dest,minlength=xyz.shape[1]).clamp_min(1).to(xyz.dtype)[None,:,None]
        agg=torch.zeros_like(h).index_add(1,dest,message)/degree
        weights=.25*torch.tanh(self.coordinate(message))
        move=torch.zeros_like(xyz).index_add(1,dest,delta/(1+(distance+1e-8).sqrt())*weights)/degree
        xyz=xyz+move
        xyz=xyz.index_copy(1,ca_index,ca)
        return h+self.update(torch.cat((h,agg),-1)),xyz


class CADetailDecoder(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__()
        self.config=dict(width=width,depth=depth)
        self.atom=nn.Embedding(118,width);self.residue=nn.Embedding(138,width)
        self.features=nn.Linear(5,width)
        self.layers=nn.ModuleList([DetailMessage(width) for _ in range(depth)])

    def forward(self,condition,ca):
        c=condition;xyz=rigid_transport(c,ca);ref=c.observed[-1]
        span=(c.observed_time_ps[-1]-c.observed_time_ps[0]).clamp_min(1)
        horizon=(c.query_time_ps-c.observed_time_ps[-1])/100
        if bool((horizon<=0).any()): raise ValueError('query must follow observed prefix')
        parent=c.ca_index[c.residue_index]
        displacement=ca[:,c.residue_index]-ref[parent]
        offset=ref-ref[parent]
        velocity=(ref-c.observed[0])/span*100
        fixed=torch.stack((offset.square().sum(-1)/25,velocity.square().sum(-1)/25),-1)
        scalar=torch.cat((fixed[None].expand(len(ca),-1,-1),
            displacement.square().sum(-1,keepdim=True)/25,
            (displacement*offset).sum(-1,keepdim=True)/25,
            torch.log1p(horizon)[:,None,None].expand(-1,len(ref),1)),-1)
        h=(self.atom(c.atom_type)+self.residue(c.residue_type))[None]+self.features(scalar)
        for layer in self.layers: h,xyz=layer(h,xyz,ref,c.edges,c.ca_index,ca)
        return xyz


def detail_loss(prediction,target,condition,ca):
    parent=condition.ca_index[condition.residue_index]
    nonca=torch.arange(prediction.shape[1],device=prediction.device)!=parent
    pred_offset=prediction-ca[:,condition.residue_index]
    true_offset=target-target[:,parent]
    coordinate=(pred_offset[:,nonca]-true_offset[:,nonca]).square().mean()
    a,b=condition.bond_index
    pred_length=torch.linalg.vector_norm(prediction[:,a]-prediction[:,b],dim=-1)
    true_length=torch.linalg.vector_norm(target[:,a]-target[:,b],dim=-1)
    bond=(pred_length-true_length).square().mean()
    return coordinate+bond,dict(detail_mse=coordinate,bond_mse=bond)
