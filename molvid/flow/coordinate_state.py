"""Paired reference-coordinate rectified flow and displacement EDM models.

The only sampled/integrated state is XYZ [Q,N,3], in Angstrom. Hidden scalar
and vector features are recomputed work representations, never future targets.
Physical query times are ps, unrelated to flow progress or diffusion sigma.
"""
from dataclasses import dataclass
from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..codec.bridge_history import BridgeHistoryConditioner, future_query_grouping, physical_features
from ..codec.collective_feedback import CollectiveFeedback
from ..codec.heads import EquivariantCoordinateHead
from ..codec.stable_norm import OverflowSafeLayerNorm
from ..data.batch import ClipBatch
from ..data.chemistry import NUM_ATOM_TYPE, NUM_BLOCK_TYPE
from ..geometry.neighbors import CudaRadiusNeighborList
from ..spatial.torchmd import TorchMDEncoder
from .reference_concat import ReferenceConcat


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
    query_groups: tuple | None = None
    reference_prefix_h: torch.Tensor | None = None  # [N,D], unnormalized
    reference_prefix_v: torch.Tensor | None = None  # [N,3,D]


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
        self.query_qk_mode = settings.get('query_qk_mode', 'atom')
        self.temporal = BridgeHistoryConditioner(width, settings['temporal_heads'],
            query_qk_mode=self.query_qk_mode)
        self.tokens = CollectiveFeedback(width)
        self.progress = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.physical_time = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.displacement = nn.Linear(1, width, bias=False)
        self.head = EquivariantCoordinateHead(width)
        # Small nonzero readout: all trainable pathways can receive gradients
        # on the first update. No existing generative weights are inherited.
        nn.init.normal_(self.head.out.weight, std=.01/math.sqrt(width))
        self.checkpoint_query_blocks = settings.get('checkpoint_query_blocks', False)
        if not isinstance(self.checkpoint_query_blocks, bool):
            raise ValueError('checkpoint_query_blocks must be boolean')
        self.capacity = {}
        for name in ('obs_extra_layers', 'query_pre_extra_layers',
                     'temporal_extra_layers', 'collective_extra_layers',
                     'query_post_extra_layers'):
            value = settings.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
            self.capacity[name] = value
        channels = settings.get('collective_extra_channels', 512)
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 4 or channels % 4:
            raise ValueError('collective_extra_channels must be positive and divisible by four')
        self.capacity['collective_extra_channels'] = channels
        # All original modules are constructed first, in their original order.
        # Only the added modules use the saved CPU RNG scope; never reseed CUDA.
        self.obs_extra = nn.ModuleList()
        self.query_pre_extra = nn.ModuleList()
        self.temporal_extra = nn.ModuleList()
        self.collective_extra = nn.ModuleList()
        self.query_post_extra = nn.ModuleList()
        if any(self.capacity[name] for name in self.capacity if name.endswith('_layers')):
            with torch.random.fork_rng(devices=[]):
                for name in ('obs', 'query_pre', 'query_post'):
                    modules = getattr(self, name+'_extra')
                    for _ in range(self.capacity[name+'_extra_layers']):
                        layer = deepcopy(self.spatial.attention_layers[0])
                        layer.reset_parameters()
                        # ET has both an o_proj residual and a direct vector
                        # message. Zero both residual outputs, while retaining
                        # a nonzero scalar Value so o_proj can learn immediately.
                        # The two vector Value slices themselves get gradients;
                        # there is no additional zero gate in series with them.
                        with torch.no_grad():
                            layer.o_proj.weight.zero_()
                            layer.o_proj.bias.zero_()
                            layer.v_proj.weight.view(
                                layer.num_heads, 3, layer.head_dim, width)[:, 1:].zero_()
                            layer.v_proj.bias.view(layer.num_heads, 3, layer.head_dim)[:, 1:].zero_()
                        modules.append(layer)
                self.temporal_extra.extend(
                    BridgeHistoryConditioner(width, settings['temporal_heads'],
                        query_qk_mode=self.query_qk_mode)
                    for _ in range(self.capacity['temporal_extra_layers']))
                self.collective_extra.extend(
                    CollectiveFeedback(width, channels=channels)
                    for _ in range(self.capacity['collective_extra_layers']))
        self.reference_concat_enabled = settings.get('reference_concat', False)
        if not isinstance(self.reference_concat_enabled, bool):
            raise ValueError('reference_concat must be boolean')
        self.reference_fusion = None
        if self.reference_concat_enabled:
            # No change to shared parameters or CPU/CUDA generator progression.
            with torch.random.fork_rng(devices=[]):
                self.reference_fusion = ReferenceConcat(width)
        self.forward_calls = 0
        self.geometry_calls = 0

    def contract(self):
        result = dict(schema='molvid.coordinate_state.model.v1', scheme=self.scheme,
            state='XYZ_or_reversible_reference_displacement_Q_N_3_Angstrom', physical_time_unit='ps',
            weights=('owned_coordinate_state_weights_only_no_external_weights'
                if self.settings.get('initialization')=='coordinate_state_weights_only'
                else 'random_initialization_no_AF3_Protenix_PVB_weights'), frozen_modules=[],
            current_geometry='rebuild_current_radius_neighbors_plus_all_covalent_edges_each_call',
            hierarchy='atom_to_chemical_unit_global_attention_within_sample_then_atom_lift',
            query_qk_mode=self.query_qk_mode,
            future_query_routing=('sample_component_block_loss_masked_scalar_mean_before_QK_ligand_edge_mask_atom_fallback'
                if self.query_qk_mode == 'group' else 'original_atom_QK'),
            future_query_values='unpooled_per_atom_h_v_with_original_output_maps_and_residuals',
            future_query_time='one_attention_row_per_requested_physical_time_ps',
            history_attention='original_atom_QK_and_atom_h_v_unchanged',
            covalent_cutoff='bonded_edges_exempt_from_cutoff_envelope_actual_distance_RBF_retained',
            source='last_observed_reference_plus_iid_coordinate_noise_no_clipping',
            scales=self.settings['scales'], generated_target='coordinates_not_future_hv',
            output='flow_vector_field_A_per_unit_s' if self.scheme=='flow' else 'clean_reference_displacement_A',
            sigma_training='log_uniform_between_frozen_TRAIN_based_bounds',
            static='same_current_geometry_temporal_bypass_tokens_and_head_on_corrupted_observation',
            parameters=sum(p.numel() for p in self.parameters()))
        if any(self.capacity[name] for name in self.capacity if name.endswith('_layers')):
            result['capacity'] = dict(self.capacity,
                obs_extra_dynamic='after_observed_spatial_before_normalization',
                obs_extra_static='same_weights_after_current_query_pre_spatial_before_temporal_bypass',
                query_pre_extra='after_original_pre_spatial_before_temporal',
                temporal_extra='after_original_temporal_same_observed_context_static_bypass',
                collective_extra='after_original_tokens_before_original_last_two_spatial',
                query_post_extra='after_original_last_two_spatial_before_normalization_and_head',
                extra_initialization='independent_CPU_fork_rng_identity_residuals_no_added_gate',
                extra_checkpoint='each_extra_spatial_layer_temporal_module_and_collective_module')
        if self.checkpoint_query_blocks:
            result['training_checkpoint'] = 'whole_query_prefix_temporal_suffix_recompute_current_graph'
        if self.reference_concat_enabled:
            result['reference_concat'] = dict(
                source='last_observed_or_corrupted_frame_only_never_clean_target',
                cache='same_observation_encoder_pass_prefix_before_original_last_two_layers',
                spatial_layers=len(self.spatial.attention_layers)-2,
                scalar_normalization='none_on_either_prefix',
                location='current_spatial_prefix_before_extra_layers_temporal_and_collective',
                operation='atomwise_channel_concat_linear_scalar_and_XYZ_shared_vector_no_bias',
                initialization='current_identity_reference_zero_no_gate_CPU_fork_rng',
                arithmetic='current_plus_(weight_minus_identity)_times_concat_preserves_autocast_identity',
                tasks=['static', 'dynamic'])
        return result

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
        # A self edge has the identically zero direction, including its
        # derivative. Avoid cancelling artificial +/-1e8 coordinate gradients
        # from delta/clamp at zero when checkpointing reconstructs the graph.
        direction = torch.where((edges[0] == edges[1])[:, None],
                                torch.zeros_like(direction), direction)
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

    def _extra_layers(self, modules, h, v, graph):
        for layer in modules:
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

    def _observed_frame(self, xyz, observed, bonds, apply_obs_extra=True, cache_reference=False):
        h, v, graph = self._begin_geometry(xyz, observed, bonds)
        if cache_reference:
            split = len(self.spatial.attention_layers)-2
            h, v = self._layers(h, v, graph, 0, split)
            reference_h, reference_v = h, v
            h, v = self._layers(h, v, graph, split, len(self.spatial.attention_layers))
        else:
            h, v = self._layers(h, v, graph, 0, len(self.spatial.attention_layers))
        if apply_obs_extra:
            h, v = self._extra_layers(self.obs_extra, h, v, graph)
        if cache_reference:
            return self.spatial.out_norm(h), v, reference_h, reference_v
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
        for frame, xyz in enumerate(observed.x):
            cache_reference = self.reference_concat_enabled and frame == observed.frames-1
            if self.training and torch.is_grad_enabled():
                encoded.append(checkpoint(self._observed_frame, xyz, observed, bonds,
                    task == 'dynamic', cache_reference, use_reentrant=False))
            else:
                encoded.append(self._observed_frame(xyz, observed, bonds, task == 'dynamic', cache_reference))
        times = observed.time_ps[observed.abid].t()
        return CoordinateStateCondition(observed, observed.x[-1],
            torch.stack([x[0] for x in encoded]), torch.stack([x[1] for x in encoded]),
            times-times[-1:], observed.x-observed.x[-1:], bonds, self.tokens.grouping(observed), task,
            future_query_grouping(observed) if self.query_qk_mode == 'group' and task == 'dynamic' else None,
            encoded[-1][2] if self.reference_concat_enabled else None,
            encoded[-1][3] if self.reference_concat_enabled else None)

    def _query_prefix_frame(self, xyz, condition, embedding, physical, scaled_z, *, keep_graph):
        h, v, graph = self._begin_geometry(xyz, condition.observed, condition.bonds)
        h = h+embedding+physical
        v = v+self.displacement(scaled_z[:, :, None])
        split = len(self.spatial.attention_layers)-2
        h, v = self._layers(h, v, graph, 0, split)
        if self.reference_fusion is not None:
            h, v = self.reference_fusion(h, v, condition.reference_prefix_h,
                                       condition.reference_prefix_v)
        if condition.task == 'static':
            # Train the observation capacity on current corrupted geometry
            # during static pretraining, where temporal conditioning is bypassed.
            h, v = self._extra_layers(self.obs_extra, h, v, graph)
        h, v = self._extra_layers(self.query_pre_extra, h, v, graph)
        return (h, v, graph) if keep_graph else (h, v)

    def _query_suffix_frame(self, h, v, xyz, condition, *, graph=None, decode=False):
        # With whole-query checkpointing, do not retain Q separate graph
        # attributes/neighbor-embedding autograd graphs between prefix/suffix.
        if graph is None:
            graph = self._graph(xyz, condition.observed, condition.bonds)
        h, v = self.tokens(h, v, xyz, condition.token_groups)
        for collective in self.collective_extra:
            if self.training and torch.is_grad_enabled():
                h, v = checkpoint(collective, h, v, xyz, condition.token_groups, use_reentrant=False)
            else:
                h, v = collective(h, v, xyz, condition.token_groups)
        split = len(self.spatial.attention_layers)-2
        h, v = self._layers(h, v, graph, split, len(self.spatial.attention_layers))
        h, v = self._extra_layers(self.query_post_extra, h, v, graph)
        h = self.spatial.out_norm(h)
        if decode:
            # The coordinate head acts independently on each frame and atom.
            return self.head(h[None], v[None])[0]
        return h, v

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
        whole_query = self.checkpoint_query_blocks and self.training and torch.is_grad_enabled()
        prefixes, graphs = [], []
        for frame, xyz in enumerate(current_A):
            args = (xyz, condition, embedding,
                    physical[frame] if condition.task == 'dynamic' else 0., scaled_z[frame])
            if whole_query:
                prefixes.append(checkpoint(self._query_prefix_frame, *args,
                    keep_graph=False, use_reentrant=False))
            else:
                hf, vf, graph = self._query_prefix_frame(*args, keep_graph=True)
                prefixes.append((hf, vf))
                graphs.append(graph)
        h, v = torch.stack([x[0] for x in prefixes]), torch.stack([x[1] for x in prefixes])
        del prefixes
        if condition.task=='dynamic':
            args = (h, v, condition.history_h, condition.history_v,
                    condition.history_displacement_A/self.sigma_data, lag, condition.history_lag_ps)
            if whole_query:
                h, v = checkpoint(self.temporal, *args, query_grouping=condition.query_groups,
                    use_reentrant=False)
            else:
                h, v = self.temporal(*args, query_grouping=condition.query_groups)
            for temporal in self.temporal_extra:
                args = (h, v, condition.history_h, condition.history_v,
                        condition.history_displacement_A/self.sigma_data, lag, condition.history_lag_ps)
                if self.training and torch.is_grad_enabled():
                    h, v = checkpoint(temporal, *args, query_grouping=condition.query_groups,
                        use_reentrant=False)
                else:
                    h, v = temporal(*args, query_grouping=condition.query_groups)
        outputs = []
        for frame, xyz in enumerate(current_A):
            if whole_query:
                outputs.append(checkpoint(self._query_suffix_frame, h[frame], v[frame], xyz,
                    condition, decode=True, use_reentrant=False))
            else:
                outputs.append(self._query_suffix_frame(h[frame], v[frame], xyz, condition,
                    graph=graphs[frame]))
        result = (torch.stack(outputs) if whole_query else
                  self.head(torch.stack([x[0] for x in outputs]), torch.stack([x[1] for x in outputs])))
        if self.scheme=='flow':
            return self.sigma_data*result
        sigma = progress[batch.abid][None, :, None]
        c_skip = self.sigma_data**2/(sigma.square()+self.sigma_data**2)
        c_out = sigma*self.sigma_data/(sigma.square()+self.sigma_data**2).sqrt()
        return c_skip*z+c_out*result

    def _flow_interval(self, state, condition, query_time_ps, index, intervals, *, recompute=False):
        """One native Heun interval; the final interval uses the native Euler end."""
        dt = 1. / intervals
        s = state.new_full((condition.observed.batch_size,), index / intervals)
        def velocity(current, progress):
            if recompute:
                return checkpoint(self, current, condition=condition, query_time_ps=query_time_ps,
                                  progress=progress, use_reentrant=False, preserve_rng_state=True)
            return self(current, condition=condition, query_time_ps=query_time_ps, progress=progress)
        u = velocity(state, s)
        proposal = state + dt * u
        if index < intervals - 1:
            u_next = velocity(proposal, s + dt)
            return state + .5 * dt * (u + u_next)
        return proposal

    def sample_truncated(self, observed, query_time_ps, *, tail_intervals, generator):
        """Native source sample with only the final K integration intervals differentiated.

        The detached prefix state omits its parameter dependence. The observed
        condition is recomputed with gradients for the suffix; no future target
        enters either condition. Whole-model-call recomputation bounds activation
        memory. This is a truncated gradient, not differentiation of all 35 NFE.
        """
        intervals = self.settings['sampler_intervals']
        if self.scheme != 'flow' or not 0 < tail_intervals <= intervals:
            raise ValueError('Valid flow tail interval count required')
        # Do not cache no-grad cast weights for the subsequent grad-enabled tail.
        with torch.no_grad(), torch.autocast(device_type=observed.x.device.type,
                enabled=torch.is_autocast_enabled(observed.x.device.type),
                dtype=torch.get_autocast_dtype(observed.x.device.type), cache_enabled=False):
            condition = self.prepare_condition(observed)
            shape = (query_time_ps.shape[1], *condition.reference_A.shape)
            noise = torch.randn(shape, device=condition.reference_A.device, generator=generator)
            state = condition.reference_A[None] + self.rho_A * noise
            for index in range(intervals - tail_intervals):
                state = self._flow_interval(state, condition, query_time_ps, index, intervals)
        state = state.detach()
        condition = self.prepare_condition(observed)
        for index in range(intervals - tail_intervals, intervals):
            state = self._flow_interval(state, condition, query_time_ps, index, intervals, recompute=True)
        return state

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
