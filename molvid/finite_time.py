"""Finite-horizon conditional expectations of train-defined response kernels.

The response is a projected scalar-teacher change, not a metastable state,
first-passage event, committor, or full path distribution.
"""
import torch
from torch import nn


class FiniteTimeResponse(nn.Module):
    def __init__(self,input_width,regions,last_width,*,history=True):
        super().__init__();self.last_width=last_width;self.history=history
        self.network=nn.Sequential(nn.Linear(input_width+4,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,regions))

    def forward(self,observed_features,horizon_ps):
        if not bool(torch.isfinite(horizon_ps).all()) or bool((horizon_ps<=0).any()):
            raise ValueError('finite positive horizon in ps required')
        x=observed_features
        if not self.history:
            x=torch.cat((x[...,:self.last_width],torch.zeros_like(x[...,self.last_width:])),dim=-1)
        t=horizon_ps.float()/100.
        clock=torch.stack((torch.log1p(t),t/12.,torch.sin(t/12.),torch.cos(t/12.)),dim=-1)
        return torch.sigmoid(self.network(torch.cat((x,clock),dim=-1)))


@torch.no_grad()
def fit_response_regions(fit_targets,*,regions=16,seed=3027):
    """Fit only on real fit-split responses; global scales are horizon-independent."""
    if fit_targets.ndim!=4 or not bool(torch.isfinite(fit_targets).all()):
        raise ValueError('finite fit targets [clips,atoms,horizons,channels] required')
    flat=fit_targets.flatten(0,2);rng=torch.Generator(device=flat.device).manual_seed(seed)
    indices=torch.randperm(len(flat),device=flat.device,generator=rng)[:min(8192,len(flat))]
    points=flat[indices]
    if len(points)<regions: raise ValueError('not enough training responses')
    centers=points[:regions].clone()
    for _ in range(20):
        assignment=torch.cdist(points,centers).argmin(-1)
        for i in range(regions):
            group=points[assignment==i]
            if len(group): centers[i]=group.mean(0)
    distance=torch.cdist(points,centers)/points.shape[-1]**.5
    assignment=distance.argmin(-1)
    global_scale=distance.min(-1).values.median().clamp_min(.05)
    widths=[];counts=[]
    for i in range(regions):
        d=distance[assignment==i,i];counts.append(len(d))
        widths.append((torch.quantile(d,.8) if len(d) else global_scale).clamp_min(.25*global_scale))
    return dict(centers=centers,bandwidth=torch.stack(widths),fit_counts=counts,seed=seed,
                fit_rows=len(points),metric='RMS distance in globally fit-scaled projected scalar-change space')


def response_kernels(changes,regions):
    centers=regions['centers'];widths=regions['bandwidth']
    distance=(changes.unsqueeze(-2)-centers).square().mean(-1)
    return torch.exp(-.5*distance/widths.square())
