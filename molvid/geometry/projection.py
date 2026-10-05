"""Observed-only coordinate repair baseline, NOT a trained dynamics model.

Updates use scalar second moments per xyz vector to preserve SO(3) covariance.
No future reference, force-field energy, or equilibrium interpretation is used.
"""
from dataclasses import dataclass
import torch
from torch import Tensor

from ..losses.sampled_physical import packed_clash_loss


@dataclass(frozen=True)
class ProjectionConfig:
    steps: int = 32
    step_size_A: float = .03
    max_shift_A: float = .5
    bond_weight: float = 10.
    clash_weight: float = 1.
    retention_weight: float = .1
    increment_retention_weight: float = .1


def repair_coordinates(generated: Tensor, *, observed: Tensor, topology, loss_mask,
                       frame_mask, physical, config=ProjectionConfig()):
    """Return repaired [Q,N,3] Å coordinates; only observed bond lengths used.

    Initial implementation accepts one molecule/system, all query frames valid.
    Explicitly rejects packed/padded batches rather than crossing their masks.
    Optimization is over coordinates; no neural parameter updates occur.
    """
    if generated.device.type!='cuda':raise RuntimeError('CUDA required')
    if topology.batch_size!=1 or not bool(frame_mask.all()) or not bool(loss_mask.all()):
        raise ValueError('initial projection requires one all-valid system')
    if generated.ndim!=3 or observed.ndim!=3 or generated.shape[1:]!=observed.shape[1:]:
        raise ValueError('coordinate shapes differ')
    if config.steps<0 or config.max_shift_A<=0 or config.step_size_A<=0:
        raise ValueError('invalid projection budget')
    source=generated.detach().float();x=source.clone()
    a,b=topology.covalent_bond_index
    target_length=torch.linalg.vector_norm(observed[-1,a]-observed[-1,b],dim=-1).detach()
    first=torch.zeros_like(x);second=torch.zeros_like(x[...,:1]);trace=[]
    with torch.enable_grad():
        for step in range(config.steps):
            x=x.detach().requires_grad_(True)
            lengths=torch.linalg.vector_norm(x[:,a]-x[:,b],dim=-1)
            bond=(lengths-target_length).square().mean() if a.numel() else x.sum()*0
            clash=packed_clash_loss(x,topology=topology,loss_mask=loss_mask,
                frame_mask=frame_mask,config=physical).loss if config.clash_weight else x.sum()*0
            delta=x-source
            retention=delta.square().sum(-1).mean()
            increment=(delta[1:]-delta[:-1]).square().sum(-1).mean() if len(x)>1 else x.sum()*0
            objective=config.bond_weight*bond+config.clash_weight*clash+config.retention_weight*retention+config.increment_retention_weight*increment
            grad,=torch.autograd.grad(objective,x)
            if not torch.isfinite(grad).all():raise FloatingPointError('nonfinite projection gradient')
            with torch.no_grad():
                first=.9*first+.1*grad
                second=.999*second+.001*grad.square().sum(-1,keepdim=True)
                move=config.step_size_A*(first/(1-.9**(step+1)))/((second/(1-.999**(step+1))).sqrt()+1e-12)
                x=x-move
                shift=x-source;norm=shift.norm(dim=-1,keepdim=True)
                x=source+shift*(config.max_shift_A/norm.clamp_min(1e-12)).clamp_max(1)
            trace.append(dict(step=step+1,objective=float(objective),bond_mse_A2=float(bond),
                clash_severity=float(clash),retention_A2=float(retention),increment_change_A2=float(increment)))
    return x.detach(),trace
