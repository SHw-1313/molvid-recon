"""Experimental pretrained geometric stochastic bridge, not coordinate RF.

The loaded released PVB geometry is owned by this new model; the reference
repository/weights are read-only. CLI entrypoints must put PVB_origin on the
import path. Coordinates [Q,N,3] are Angstrom, physical times are ps, and
flow progress is dimensionless. No future target enters conditioning/sampling.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..codec.bridge_history import BridgeHistoryConditioner
from ..data.batch import ClipBatch


class RecomputedGeometryLayer(nn.Module):
    """Owned pretrained layer, unchanged arithmetic with activation recompute."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.layer, *args, use_reentrant=False,
                              preserve_rng_state=True, **kwargs)
        return self.layer(*args, **kwargs)


@dataclass(frozen=True)
class BridgeCondition:
    observed: ClipBatch
    history_h: torch.Tensor
    history_v: torch.Tensor
    history_lag_ps: torch.Tensor
    history_displacement_A: torch.Tensor
    source_mean_A: torch.Tensor
    source_log_variance: torch.Tensor
    kl_loss: torch.Tensor
    task: str


class PretrainedGeometryBridge(nn.Module):
    """Shared pretrained geometry with zero-initialized temporal residuals.

Uniform valid H/Q per packed minibatch is currently required. H may differ
between minibatches; molecular roles and topology use the existing ClipBatch
interface for proteins and complexes. This is an explicit prototype limit,
not a claim of arbitrary padded-batch support.
    """
    def __init__(self, geometry, *, recompute_frames=True, temporal_heads=4):
        super().__init__()
        if geometry.using_ode or geometry.backbone != 'torchmdnet':
            raise ValueError('Released TorchMD stochastic bridge is required')
        self.geometry = geometry
        self.conditioner = BridgeHistoryConditioner(geometry.hidden_dim, temporal_heads)
        self.recompute_frames = bool(recompute_frames)
        if self.recompute_frames:
            # Bound backward memory for full 17k-atom structures, without crops.
            self.geometry.decoder.attention_layers = nn.ModuleList([
                RecomputedGeometryLayer(layer) for layer in geometry.decoder.attention_layers])

    def contract(self):
        return dict(schema='molvid.pretrained_geometry_bridge.v1',
            geometry='released_PVB_static_pretrain_encoder_decoder_velocity_drift',
            temporal='zero_residual_observed_history_and_joint_query_attention',
            physical_time_unit='ps', coordinate_unit='Angstrom',
            flow_family='PVB_stochastic_interpolant_velocity_and_drift_not_RF',
            static='same_geometry_and_heads_temporal_bypass_corrupted_structure_input',
            padded_frames_supported=False, recompute_frames=self.recompute_frames)

    def _graph(self, xyz, batch):
        # Import only when this experimental architecture is selected.
        from module.graph import construct_edges
        g = self.geometry
        return construct_edges(Z=batch.atype, X=xyz, bid=batch.abid,
            mask=batch.edge_mask, bond_index=batch.bond_index,
            cutoff_lower=g.cutoff_lower, cutoff_upper=g.cutoff_upper,
            cutoff_H=g.cutoff_H, k_neighbors=g.k_neighbors)

    def _encode_frame(self, xyz, batch):
        edge, distance, direction, kind = self._graph(xyz, batch)
        h, v, *_ = self.geometry.encoder(z=batch.atype, b=batch.btype,
            pos=xyz, batch=batch.abid, edge_index=edge, edge_weight_t=distance,
            edge_vec_t=direction, bond_type=kind)
        return h, v

    def _encode(self, xyz, batch):
        if self.recompute_frames and self.training and torch.is_grad_enabled():
            return checkpoint(lambda x: self._encode_frame(x, batch), xyz,
                              use_reentrant=False, preserve_rng_state=True)
        return self._encode_frame(xyz, batch)

    def prepare_condition(self, observed, *, task='dynamic', source_perturbation_A=0., generator=None):
        """Only observed prefix (or an already-corrupted static singleton).

For static training, the caller must corrupt x before constructing this
condition and keep clean coordinates solely in the loss. Dynamic source
perturbation affects only the source encoder; history remains observed data.
        """
        if task not in ('dynamic', 'static') or (task == 'static' and observed.x.shape[0] != 1):
            raise ValueError('static requires a singleton; otherwise dynamic')
        if not bool(observed.frame_mask.all()):
            raise ValueError('Use a uniform valid-history batch, without padding')
        if source_perturbation_A < 0 or not math.isfinite(source_perturbation_A):
            raise ValueError('source perturbation must be finite and nonnegative')
        encoded = [self._encode(x, observed) for x in observed.x]
        h = torch.stack([a for a, _ in encoded])
        v = torch.stack([a for _, a in encoded])
        last = observed.x[-1]
        source_mean = last
        source_h = h[-1]
        if source_perturbation_A:
            source_mean = last + source_perturbation_A * torch.randn(
                last.shape, dtype=last.dtype, device=last.device, generator=generator)
            source_h, _ = self._encode(source_mean, observed)
        logvar = -self.geometry.W_vec_log_var(source_h).abs()
        mask = observed.loss_mask
        kl = -.5 * (1. + logvar[mask] - math.log(self.geometry.coord_prior_var)
                    - logvar[mask].exp() / self.geometry.coord_prior_var).sum() / mask.sum()
        times = observed.time_ps.index_select(0, observed.abid).transpose(0, 1)
        return BridgeCondition(observed, h, v, times - times[-1:],
            observed.x - last[None], source_mean, logvar, kl, task)

    @staticmethod
    def draw_source(condition, query_frames, *, generator=None):
        mean, logvar = condition.source_mean_A, condition.source_log_variance
        noise = torch.randn((query_frames, *mean.shape), device=mean.device,
                            dtype=mean.dtype, generator=generator)
        return mean[None] + (.5 * logvar).exp()[None] * noise

    def _decode_frame(self, state, source, flow_time_atoms, batch):
        edge, distance, direction, kind = self._graph(state, batch)
        reference_direction = source[edge[0]] - source[edge[1]]
        reference_distance = torch.linalg.vector_norm(reference_direction, dim=-1)
        h, v, *_ = self.geometry.decoder(z=batch.atype, b=batch.btype,
            pos=state, batch=batch.abid, t=flow_time_atoms,
            edge_index=edge, edge_weight_0=reference_distance,
            edge_vec_0=reference_direction, edge_weight_t=distance,
            edge_vec_t=direction, bond_type=kind)
        return h, v

    def forward(self, state_A, source_A, *, condition, query_time_ps, flow_time):
        """Return velocity/drift [Q,N,3], retaining the original bridge heads."""
        batch = condition.observed
        if state_A.shape != source_A.shape or state_A.ndim != 3 or state_A.shape[1:] != batch.x.shape[1:]:
            raise ValueError('state/source must both be [Q,N,3]')
        frames, atoms, _ = state_A.shape
        if query_time_ps.shape != (batch.batch_size, frames):
            raise ValueError('Physical query times must be [B,Q] in ps')
        if flow_time.shape != (batch.batch_size,) or not bool(((flow_time >= 0) & (flow_time <= 1)).all()):
            raise ValueError('Flow progress must be [B] in [0,1]')
        lag = (query_time_ps - batch.time_ps[:, -1:]).index_select(0, batch.abid).transpose(0, 1)
        if condition.task == 'dynamic' and not bool((lag > 0).all()):
            raise ValueError('Dynamic queries must follow observed history')
        t = flow_time.index_select(0, batch.abid)[:, None]
        features = []
        for frame in range(frames):
            result = self._decode_frame(state_A[frame], source_A[frame], t, batch)
            features.append(result)
        h, v = torch.stack([a for a, _ in features]), torch.stack([b for _, b in features])
        if condition.task == 'dynamic':
            h, v = self.conditioner(h, v, condition.history_h, condition.history_v,
                condition.history_displacement_A, lag, condition.history_lag_ps)
        width = h.shape[-1]
        h, v = h.reshape(frames * atoms, width), v.reshape(frames * atoms, 3, width)
        _, velocity = self.geometry.vel_ffn(h, v)
        _, drift = self.geometry.drf_ffn(h, v)
        return velocity.sum(-1).reshape(frames, atoms, 3), drift.sum(-1).reshape(frames, atoms, 3)

    @torch.no_grad()
    def sample(self, condition, query_time_ps, *, steps=10, generator=None):
        """Released SDE grid/final-noise omission, extended to joint queries."""
        if steps < 2:
            raise ValueError('At least two SDE steps are required')
        source = self.draw_source(condition, query_time_ps.shape[1], generator=generator)
        state = source.clone()
        dt = 1. / steps
        for step in range(steps):
            time = state.new_full((condition.observed.batch_size,), step * dt)
            _, drift = self(state, source, condition=condition,
                            query_time_ps=query_time_ps, flow_time=time)
            state = state + drift * dt
            if step != steps - 1:
                state = state + self.geometry.sigma * math.sqrt(dt) * torch.randn(
                    state.shape, dtype=state.dtype, device=state.device, generator=generator)
        return state
