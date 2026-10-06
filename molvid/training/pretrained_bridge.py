"""Stochastic-interpolant objective for the pretrained geometric bridge.

Clean static coordinates and dynamic futures enter here, never conditioning.
The source encoder remains differentiable, as in the released PVB objective.
"""
import torch


def bridge_objective(model, condition, target_A, query_time_ps, *, generator=None):
    """Velocity/drift MSE plus source KL; no coordinate-RF target substitution.

target_A: [Q,N,3] Angstrom, query_time_ps: [B,Q] ps. Per-role reduction follows
PVB: mean over all selected atoms plus mean over ligand atoms when present.
Packed batches therefore use atom weighting; experiments must report their
batching policy instead of calling this system-equal weighting.
    """
    from module.interpolant_matcher import INTERP_MATCHER
    batch = condition.observed
    frames = target_A.shape[0]
    if target_A.shape != (frames, batch.x.shape[1], 3):
        raise ValueError('Targets must be [Q,N,3] in the observed atom order')
    source = model.draw_source(condition, frames, generator=generator)
    flow_time = torch.rand((batch.batch_size,), device=target_A.device, generator=generator)
    atom_time = flow_time.index_select(0, batch.abid)[None, :, None]
    noise = torch.randn(target_A.shape, device=target_A.device,
                        dtype=target_A.dtype, generator=generator)
    sigma = model.geometry.sigma
    # Reuse the released gamma and stabilized velocity/drift definitions.
    state = source + atom_time * (target_A - source) + INTERP_MATCHER.gamma_(atom_time, sigma) * noise
    velocity_target = INTERP_MATCHER.velocity(target_A - source, noise, atom_time, sigma)
    drift_target = INTERP_MATCHER.drift(target_A - source, noise, atom_time, sigma)
    velocity, drift = model(state, source, condition=condition,
                             query_time_ps=query_time_ps, flow_time=flow_time)
    mask = batch.loss_mask
    ligand = mask & (batch.edge_mask == 1)
    vel_all = (velocity[:, mask] - velocity_target[:, mask]).square().mean()
    drift_all = (drift[:, mask] - drift_target[:, mask]).square().mean()
    vel_lig = vel_all.new_zeros(())
    drift_lig = drift_all.new_zeros(())
    if bool(ligand.any()):
        vel_lig = (velocity[:, ligand] - velocity_target[:, ligand]).square().mean()
        drift_lig = (drift[:, ligand] - drift_target[:, ligand]).square().mean()
    reconstruction = vel_all + drift_all + vel_lig + drift_lig
    loss = model.geometry.kl_weight * condition.kl_loss + model.geometry.re_weight * reconstruction
    return loss, dict(velocity_all=vel_all.detach(), drift_all=drift_all.detach(),
        velocity_ligand=vel_lig.detach(), drift_ligand=drift_lig.detach(),
        kl=condition.kl_loss.detach(), flow_time_mean=flow_time.detach().mean())
