"""Current-coordinate equivariant geometry recovery shared across physical frames.

This is a geometry intervention, not a temporal generator. Radius topology is
rebuilt from predicted coordinates at every layer; only discrete edge selection
is detached. Targets are absent from the forward interface.
"""
from dataclasses import dataclass
import torch
from torch import nn

from .neighbors import CudaRadiusNeighborList
from ..data.chemistry import NUM_ATOM_TYPE,NUM_BLOCK_TYPE


@dataclass(frozen=True)
class RecoveryConfig:
    width: int = 96
    depth: int = 4
    cutoff_A: float = 4.5
    max_neighbors: int = 128
    max_layer_shift_A: float = .25


class RecoveryLayer(nn.Module):
    def __init__(self, width, max_shift, *, update_hidden):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2*width+18,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.scalar = nn.Linear(width,width) if update_hidden else None
        self.force = nn.Linear(width,1,bias=False)
        nn.init.zeros_(self.force.weight)
        self.norm = nn.LayerNorm(width) if update_hidden else None
        self.max_shift = max_shift

    def forward(self,x,h,edges,bond_flag,centers,abid,batch_size,valid):
        source,destination=edges
        difference=x[source]-x[destination]
        distance=difference.norm(dim=-1).clamp_min(1e-6)
        rbf=torch.exp(-4*(distance[:,None]-centers[None]).square())
        features=torch.cat((h[source],h[destination],rbf,bond_flag[:,None].float(),distance[:,None]),dim=-1)
        message=self.edge(features)
        # Central forces are vector-equivariant; xyz components share all gates.
        direction=difference/distance[:,None]
        vector=torch.zeros_like(x).index_add(0,destination,self.force(message)*direction)
        degree=torch.bincount(destination,minlength=len(x)).to(x).clamp_min(1)[:,None]
        shift=vector/degree.sqrt()
        # Geometry recovery must not add whole-system translational motion.
        counts=torch.bincount(abid[valid],minlength=batch_size).to(x).clamp_min(1)[:,None]
        mean=torch.zeros(batch_size,3,device=x.device).index_add(0,abid,shift*valid[:,None])/counts
        shift=(shift-mean[abid])*valid[:,None]
        norm=shift.norm(dim=-1,keepdim=True)
        # One invariant scale per system preserves the zero-sum correction.
        maximum=torch.zeros(batch_size,1,device=x.device).scatter_reduce(0,abid[:,None],norm,reduce='amax')
        shift=shift*(self.max_shift/maximum[abid].clamp_min(1e-12)).clamp_max(1)
        if self.scalar is not None:
            scalar=torch.zeros_like(h).index_add(0,destination,self.scalar(message))
            h=self.norm(h+scalar/degree)
        return x+shift,h


class CurrentGeometryRecovery(nn.Module):
    def __init__(self,config=RecoveryConfig()):
        super().__init__();self.config=config
        if config.depth<1 or config.width<1 or config.max_layer_shift_A<=0: raise ValueError('invalid recovery config')
        self.atom_embedding=nn.Embedding(NUM_ATOM_TYPE,config.width)
        self.block_embedding=nn.Embedding(NUM_BLOCK_TYPE,config.width)
        self.layers=nn.ModuleList([RecoveryLayer(config.width,config.max_layer_shift_A,
            update_hidden=i<config.depth-1) for i in range(config.depth)])
        self.register_buffer('centers',torch.linspace(0,config.cutoff_A,16))
        self.neighbors=CudaRadiusNeighborList(cutoff_lower=0,cutoff_upper=config.cutoff_A,
            max_num_neighbors=config.max_neighbors,loop=False)

    def forward(self,coordinates,*,topology,frame_mask):
        """[T,N,3] Angstrom, packed static topology, [B,T] frame validity.

        Padded systems are omitted from each radius graph, never connected to
        real systems. Covalent edges are included regardless of current length.
        Types outside the vocabulary fail instead of being silently folded.
        """
        if coordinates.device.type!='cuda' or coordinates.dtype!=torch.float32:
            raise ValueError('current recovery requires FP32 CUDA coordinates')
        if coordinates.ndim!=3 or coordinates.shape[1:]!=(topology.num_atoms,3):
            raise ValueError('expected [T,N,3]')
        if frame_mask.shape!=(topology.batch_size,len(coordinates)): raise ValueError('frame mask shape')
        if bool(((topology.atom_type<0)|(topology.atom_type>=NUM_ATOM_TYPE)).any()):
            raise ValueError('atom type outside shared chemistry vocabulary')
        if bool(((topology.block_type<0)|(topology.block_type>=NUM_BLOCK_TYPE)).any()):
            raise ValueError('block type outside shared chemistry vocabulary')
        h0=self.atom_embedding(topology.atom_type)+self.block_embedding(topology.block_type)
        n=topology.num_atoms;a,b=topology.covalent_bond_index
        bonds=torch.cat((a*n+b,b*n+a)).unique(sorted=True)
        output=[]
        for frame,x0 in enumerate(coordinates):
            valid=frame_mask[:,frame].index_select(0,topology.abid)
            active=valid.nonzero().flatten()
            if not len(active): output.append(torch.zeros_like(x0));continue
            x=x0;h=h0
            for layer in self.layers:
                graph=self.neighbors(x[active],topology.abid[active],batch_size=topology.batch_size)
                radius=active[graph.edge_index]
                active_bonds=bonds[valid[bonds//n]&valid[bonds%n]]
                keys=torch.cat((radius[0]*n+radius[1],active_bonds)).unique(sorted=True)
                edges=torch.stack((keys//n,keys%n))
                bond_flag=torch.isin(keys,active_bonds)
                x,h=layer(x,h,edges,bond_flag,self.centers,topology.abid,topology.batch_size,valid)
            output.append(x*valid[:,None])
        return torch.stack(output)
