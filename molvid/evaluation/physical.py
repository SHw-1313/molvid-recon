"""Frame-resolved vdW geometry diagnostics using the frozen A threshold table."""

import torch

from ..losses.distribution import _sample_metadata
from ..losses.sampled_physical import _topological_distances


@torch.no_grad()
def sampled_clash_metrics(coordinates, batch, config):
    """Each undirected nonexcluded pair counts once; units are Å and Å².

    Returns per-frame affected-atom fractions and clashes/1000 atoms, followed
    by sample-equal summaries. Legacy <0.7Å per-nonbond-pair is reported by the
    separate historical geometry metric, not redefined here.
    """
    topology=batch.observed_context.topology
    excluded,one_four=_topological_distances(topology)
    systems=[]
    for sample in range(topology.batch_size):
        nodes,_,_,frames,_=_sample_metadata(batch,sample)
        ids=nodes.cpu().tolist();local={atom:i for i,atom in enumerate(ids)}
        radii=coordinates.new_tensor([config.vdw_radii_A[str(t)] for t in topology.atom_type[nodes].cpu().tolist()],dtype=torch.float32)
        xyz=coordinates[frames][:,nodes].float()
        counts=torch.zeros(len(frames),device=xyz.device,dtype=torch.long)
        severity=torch.zeros(len(frames),device=xyz.device)
        affected=torch.zeros((len(frames),len(nodes)),device=xyz.device,dtype=torch.bool)
        for start in range(0,len(nodes),config.distance_chunk_size):
            stop=min(start+config.distance_chunk_size,len(nodes))
            valid=torch.arange(start,stop,device=xyz.device)[:,None]<torch.arange(len(nodes),device=xyz.device)[None,:]
            is14=torch.zeros_like(valid)
            for relations,mask,value in ((excluded,valid,False),(one_four,is14,True)):
                pairs=[(i-start,local[b]) for i,a in enumerate(ids[start:stop],start) for b in relations[a] if b in local]
                if pairs:
                    index=torch.tensor(pairs,device=xyz.device,dtype=torch.long).T
                    mask[index[0],index[1]]=value
            threshold=(radii[start:stop,None]+radii[None,:]-torch.where(is14,config.one_four_overlap_A,config.allowed_overlap_A)).clamp_min(0)
            for frame,positions in enumerate(xyz):
                distance=torch.linalg.vector_norm(positions[start:stop,None]-positions[None,:],dim=-1)
                overlap=(threshold-distance).relu()*valid
                a,b=torch.where(overlap>0)
                counts[frame]+=len(a);severity[frame]+=overlap.square().sum()/len(nodes)
                affected[frame,start+a]=True;affected[frame,b]=True
        rows=[{'sample':sample,'query_frame':int(frame),'query_time_ps':float(batch.query.time_ps[sample,frame]),
               'clashes':int(counts[i]),'clashes_per_1000_atoms':float(counts[i])*1000/len(nodes),
               'affected_atom_fraction':float(affected[i].float().mean()),'vdw_severity_A2_per_atom':float(severity[i])}
              for i,frame in enumerate(frames.tolist())]
        systems.append({'frames':rows,'clashes_per_1000_atoms':float(counts.float().mean())*1000/len(nodes),
                        'affected_atom_fraction':float(affected.float().mean()),'vdw_severity_A2_per_atom':float(severity.mean())})
    return {'samples':systems,**{key:sum(s[key] for s in systems)/len(systems)
            for key in ('clashes_per_1000_atoms','affected_atom_fraction','vdw_severity_A2_per_atom')}}
