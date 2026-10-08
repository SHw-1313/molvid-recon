"""Paired reference-coordinate rectified flow and displacement EDM models.

The only sampled/integrated state is XYZ [Q,N,3], in Angstrom. Hidden scalar
and vector features are recomputed work representations, never future targets.
Physical query times are ps, unrelated to flow progress or diffusion sigma.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..codec.bridge_history import BridgeHistoryConditioner, physical_features
from ..codec.collective_feedback import CollectiveFeedback
from ..codec.heads import EquivariantCoordinateHead
from ..codec.stable_norm import OverflowSafeLayerNorm
from ..data.batch import ClipBatch
from ..data.chemistry import NUM_ATOM_TYPE, NUM_BLOCK_TYPE
from ..geometry.neighbors import CudaRadiusNeighborList
from ..spatial.torchmd import TorchMDEncoder


@dataclass(frozen=True)
class CoordinateStateCondition:
    observed: ClipBatch
    reference_A: torch.Tensor
    history_h: torch.Tensor
    history_v: torch.Tensor
    history_lag_ps: torch.Tensor
    history_displacement_A: torch.Tensor
    bonds: torch.Tensor
    token_groups: tuple
    task: str


class CoordinateStateDenoiser(nn.Module):
    """Same trainable architecture for A/B; distinct objective/output semantics."""
    def __init__(self, settings):
        super().__init__()
        if settings['schema'] != 'molvid.coordinate_state.recipe.v1':
            raise ValueError('Coordinate-state recipe required; latent/bridge checkpoints are incompatible')
        self.settings = settings
        self.scheme = settings['scheme']
        if self.scheme not in ('flow', 'edm'):
            raise ValueError('Choose flow or edm')
        scales = settings['scales']
        self.sigma_data = float(scales['sigma_data_A'])
        self.rho_A = float(scales['flow_rho_A'])
        self.sigma_min, self.sigma_max = float(scales['sigma_min_A']), float(scales['sigma_max_A'])
        if not (0 < self.sigma_min < self.sigma_max and self.sigma_data > 0 and self.rho_A > 0):
            raise ValueError('Finite positive coordinate scales required')
        width = settings['width']
        self.spatial = TorchMDEncoder(hidden_channels=width, num_layers=settings['spatial_layers'],
            num_rbf=settings['num_rbf'], num_heads=settings['temporal_heads'],
            cutoff_upper=settings['radius_A'], max_z=NUM_ATOM_TYPE, max_b=NUM_BLOCK_TYPE)
        self.spatial.out_norm = OverflowSafeLayerNorm.from_layer(self.spatial.out_norm)
        self.neighbors = CudaRadiusNeighborList(cutoff_lower=0., cutoff_upper=settings['radius_A'],
            max_num_neighbors=settings['max_neighbors'], loop=True)
        self.temporal = BridgeHistoryConditioner(width, settings['temporal_heads'])
        self.tokens = CollectiveFeedback(width)
        self.progress = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.physical_time = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.displacement = nn.Linear(1, width, bias=False)
        self.head = EquivariantCoordinateHead(width)
        # Small nonzero readout: all trainable pathways can receive gradients
        # on the first update. No existing generative weights are inherited.
        nn.init.normal_(self.head.out.weight, std=.01/math.sqrt(width))
        self.forward_calls = 0
        self.geometry_calls = 0

    def contract(self):
        return dict(schema='molvid.coordinate_state.model.v1', scheme=self.scheme,
            state='XYZ_or_reversible_reference_displacement_Q_N_3_Angstrom', physical_time_unit='ps',
            weights=('owned_coordinate_state_weights_only_no_external_weights'
                if self.settings.get('initialization')=='coordinate_state_weights_only'
                else 'random_initialization_no_AF3_Protenix_PVB_weights'), frozen_modules=[],
            current_geometry='rebuild_current_radius_neighbors_plus_all_covalent_edges_each_call',
            hierarchy='atom_to_chemical_unit_global_attention_within_sample_then_atom_lift',
            covalent_cutoff='bonded_edges_exempt_from_cutoff_envelope_actual_distance_RBF_retained',
            source='last_observed_reference_plus_iid_coordinate_noise_no_clipping',
            scales=self.settings['scales'], generated_target='coordinates_not_future_hv',
            output='flow_vector_field_A_per_unit_s' if self.scheme=='flow' else 'clean_reference_displacement_A',
            sigma_training='log_uniform_between_frozen_TRAIN_based_bounds',
            static='same_current_geometry_temporal_bypass_tokens_and_head_on_corrupted_observation',
            parameters=sum(p.numel() for p in self.parameters()))

    def _graph(self, xyz, batch, bonds):
        self.geometry_calls += 1
        spatial = self.neighbors(xyz.float(), batch.abid, batch_size=batch.batch_size).edge_index
        edges = torch.unique(torch.cat((spatial, bonds), 1), dim=1)
        code = edges[0]*len(xyz)+edges[1]
        bond_codes = bonds[0]*len(xyz)+bonds[1]
        kind = torch.isin(code, bond_codes).long()
        delta = xyz.float()[edges[0]]-xyz.float()[edges[1]]
        distance = delta.norm(dim=-1)
        direction = delta/distance.clamp_min(1e-8)[:, None]
        rbf = self.spatial.distance_expansion(distance)
        attributes = torch.cat((rbf, self.spatial.bond_embedding(kind)), -1)
        # This is an attention envelope argument, not substituted geometry.
        # Actual distances/directions above remain those of the candidate XYZ.
        envelope_distance = torch.where(kind.bool(), torch.zeros_like(distance), distance)
        return edges, envelope_distance, attributes, direction

    def _layers(self, h, v, graph, start, stop):
        for layer in self.spatial.attention_layers[start:stop]:
            if self.training and torch.is_grad_enabled():
                dh, dv = checkpoint(layer, h, v, *graph, dim_size=len(h), use_reentrant=False)
            else:
                dh, dv = layer(h, v, *graph, dim_size=len(h))
            h, v = h+dh, v+dv
        return h, v

    def _begin_geometry(self, xyz, batch, bonds):
        graph = self._graph(xyz, batch, bonds)
        h = self.spatial.atom_embedding(batch.atype)+self.spatial.block_embedding(batch.btype)
        h = self.spatial.neighbor_embedding(batch.atype, h, graph[0], graph[1], graph[2])
        v = h.new_zeros(len(h), 3, h.shape[-1])
        return h, v, graph

    def _observed_frame(self, xyz, observed, bonds):
        h, v, graph = self._begin_geometry(xyz, observed, bonds)
        h, v = self._layers(h, v, graph, 0, len(self.spatial.attention_layers))
        return self.spatial.out_norm(h), v

    def prepare_condition(self, observed, *, task='dynamic'):
        if task not in ('dynamic', 'static') or (task=='static' and observed.frames!=1):
            raise ValueError('Static is a single corrupted structure, not a zero-motion trajectory')
        if not bool(observed.frame_mask.all()):
            raise ValueError('Initial experiment requires uniform valid H within a packed batch')
        bonds = torch.unique(torch.cat((observed.bond_index, observed.bond_index.flip(0)), 1), dim=1)
        if bonds.numel() and bool((observed.abid[bonds[0]]!=observed.abid[bonds[1]]).any()):
            raise ValueError('Cross-system covalent edge')
        encoded = []
        for xyz in observed.x:
            if self.training and torch.is_grad_enabled():
                encoded.append(checkpoint(self._observed_frame, xyz, observed, bonds, use_reentrant=False))
            else:
                encoded.append(self._observed_frame(xyz, observed, bonds))
        times = observed.time_ps[observed.abid].t()
        return CoordinateStateCondition(observed, observed.x[-1],
            torch.stack([x[0] for x in encoded]), torch.stack([x[1] for x in encoded]),
            times-times[-1:], observed.x-observed.x[-1:], bonds, self.tokens.grouping(observed), task)

    def forward(self, current_A, *, condition, query_time_ps, progress):
        """A returns u(X,s,C); B returns clean delta D(Z,sigma,C), NOT next Z."""
        self.forward_calls += 1
        batch = condition.observed
        if current_A.ndim!=3 or current_A.shape[1:]!=condition.reference_A.shape:
            raise ValueError('The model state must be [Q,N,3] Cartesian coordinates')
        if query_time_ps.shape != (batch.batch_size, len(current_A)) or progress.shape != (batch.batch_size,):
            raise ValueError('query times[B,Q] and flow progress/noise[B] are distinct inputs')
        if self.scheme=='flow':
            if not bool(((progress>=0)&(progress<=1)).all()):
                raise ValueError('Flow s must lie in[0,1]')
            c_in = torch.full_like(progress, 1./self.sigma_data)
            noise_embedding = 2.*progress-1.
        else:
            if not bool((progress>0).all()):
                raise ValueError('Denoiser never receives sigma=0')
            c_in = (progress.square()+self.sigma_data**2).rsqrt()
            noise_embedding = torch.log(progress/self.sigma_data)/4.
        z = current_A.float()-condition.reference_A[None]
        scaled_z = z*c_in[batch.abid][None, :, None]
        lag = (query_time_ps-batch.time_ps[:, -1:])[batch.abid].t()
        if condition.task=='dynamic' and not bool((lag>0).all()):
            raise ValueError('Dynamic queries must follow observed history')
        embedding = self.progress(noise_embedding[:, None])[batch.abid]
        physical = self.physical_time(physical_features(lag)) if condition.task=='dynamic' else 0.
        prefixes = []
        split = len(self.spatial.attention_layers)-2
        for frame, xyz in enumerate(current_A):
            h, v, graph = self._begin_geometry(xyz, batch, condition.bonds)
            h = h+embedding+(physical[frame] if condition.task=='dynamic' else 0.)
            v = v+self.displacement(scaled_z[frame, :, :, None])
            h, v = self._layers(h, v, graph, 0, split)
            prefixes.append((h, v, graph))
        h, v = torch.stack([x[0] for x in prefixes]), torch.stack([x[1] for x in prefixes])
        if condition.task=='dynamic':
            h, v = self.temporal(h, v, condition.history_h, condition.history_v,
                condition.history_displacement_A/self.sigma_data, lag, condition.history_lag_ps)
        outputs = []
        for frame, (_, _, graph) in enumerate(prefixes):
            hf, vf = self.tokens(h[frame], v[frame], current_A[frame], condition.token_groups)
            hf, vf = self._layers(hf, vf, graph, split, len(self.spatial.attention_layers))
            outputs.append((self.spatial.out_norm(hf), vf))
        result = self.head(torch.stack([x[0] for x in outputs]), torch.stack([x[1] for x in outputs]))
        if self.scheme=='flow':
            return self.sigma_data*result
        sigma = progress[batch.abid][None, :, None]
        c_skip = self.sigma_data**2/(sigma.square()+self.sigma_data**2)
        c_out = sigma*self.sigma_data/(sigma.square()+self.sigma_data**2).sqrt()
        return c_skip*z+c_out*result

    @torch.no_grad()
    def sample(self, condition, query_time_ps, *, steps=None, generator=None):
        intervals = self.settings['sampler_intervals'] if steps is None else steps
        if intervals<2:
            raise ValueError('At least two sampler intervals')
        shape = (query_time_ps.shape[1], *condition.reference_A.shape)
        noise = torch.randn(shape, device=condition.reference_A.device, generator=generator)
        batch_size = condition.observed.batch_size
        if self.scheme=='flow':
            state = condition.reference_A[None]+self.rho_A*noise
            dt = 1./intervals
            for index in range(intervals):
                s = state.new_full((batch_size,), index/intervals)
                u = self(state, condition=condition, query_time_ps=query_time_ps, progress=s)
                proposal = state+dt*u
                if index<intervals-1:
                    u_next = self(proposal, condition=condition, query_time_ps=query_time_ps, progress=s+dt)
                    state = state+.5*dt*(u+u_next)
                else:
                    state = proposal
            return state
        power = self.settings['sigma_schedule_power']
        ramp = torch.linspace(0., 1., intervals, device=noise.device, dtype=torch.float64)
        sigmas = (self.sigma_max**(1./power)+ramp*(self.sigma_min**(1./power)-self.sigma_max**(1./power)))**power
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1))).float()
        z = self.sigma_max*noise
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            clean = self(condition.reference_A[None]+z, condition=condition,
                query_time_ps=query_time_ps, progress=sigma.expand(batch_size))
            direction = (z-clean)/sigma
            proposal = z+(next_sigma-sigma)*direction
            if bool(next_sigma>0):
                clean_next = self(condition.reference_A[None]+proposal, condition=condition,
                    query_time_ps=query_time_ps, progress=next_sigma.expand(batch_size))
                z = z+.5*(next_sigma-sigma)*(direction+(proposal-clean_next)/next_sigma)
            else:
                # Algebraically the Euler endpoint is the clean prediction;
                # assign it directly to avoid subtractive cancellation.
                z = clean
        return condition.reference_A[None]+z


def coordinate_state_objective(model, condition, target_A, query_time_ps, *, generator):
    """Valid-frame/atom means within each packed system, then equal systems."""
    batch = condition.observed
    device = target_A.device
    uniform = torch.rand(batch.batch_size, device=device, generator=generator)
    noise = torch.randn(target_A.shape, device=device, generator=generator)
    if model.scheme=='flow':
        progress = uniform
        source = condition.reference_A[None]+model.rho_A*noise
        s = progress[batch.abid][None, :, None]
        current = (1.-s)*source+s*target_A
        target = target_A-source
        prediction = model(current, condition=condition, query_time_ps=query_time_ps, progress=progress)
        weight = torch.full_like(progress, 1./model.sigma_data**2)
    else:
        progress = torch.exp(math.log(model.sigma_min)+uniform*math.log(model.sigma_max/model.sigma_min))
        sigma = progress[batch.abid][None, :, None]
        target = target_A-condition.reference_A[None]
        current = target_A+sigma*noise
        prediction = model(current, condition=condition, query_time_ps=query_time_ps, progress=progress)
        weight = (progress.square()+model.sigma_data**2)/(progress*model.sigma_data).square()
    error = (prediction.float()-target.float()).square().mean(-1)
    losses = []
    for sample in range(batch.batch_size):
        valid = batch.loss_mask & (batch.abid==sample)
        if not bool(valid.any()):
            raise ValueError('Every system needs valid target atoms')
        losses.append(error[:, valid].mean()*weight[sample])
    loss = torch.stack(losses).mean()
    return loss, dict(progress_mean=float(progress.mean().detach()),
        raw_coordinate_MSE_A2=float(error[:, batch.loss_mask].mean().detach()),
        valid_atoms=int(batch.loss_mask.sum()), query_frames=len(target_A))
