"""Same-weight RF sampling protocols, without future-coordinate inputs.

Physical query times are ps; RF progress is dimensionless. Coarse/fine is
inference-time inpainting for a checkpoint without endpoint-conditioning
training, not the endpoint-trained BioKinema sampler. All unknown fine frames
are integrated through the RF model; only model-generated anchors are clamped.
"""
import torch

from time_sampling import flow_heun, sample_short_blocks, history_with_coordinates


def _validate_flow_control(model, condition, times, intervals):
    if getattr(model, "scheme", None) != "flow":
        raise NotImplementedError("short/coarse_fine require RF (model.scheme == 'flow'); EDM is unsupported")
    if model.settings.get("temporal_experiment", {}).get("mode", "control") != "control":
        raise NotImplementedError("short/coarse_fine currently support TIME-compatible RF control only")
    intervals = model.settings["sampler_intervals"] if intervals is None else intervals
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 2:
        raise ValueError("At least two integer RF integration intervals are required")
    if times.ndim != 2 or times.shape[0] != condition.observed.batch_size or times.shape[1] < 1:
        raise ValueError("query_time_ps must have shape [packed systems, positive future frames]")
    if times.device != condition.reference_A.device:
        raise ValueError("query_time_ps and reference coordinates must be on the same device")
    if not bool(torch.isfinite(times).all()) or not bool((times[:, 1:] > times[:, :-1]).all()):
        raise ValueError("Finite, strictly increasing physical query times are required")
    if not bool((times[:, 0] > condition.observed.time_ps[:, -1]).all()):
        raise ValueError("All query times must follow the observed history")
    return intervals


def _coarse_fine(model, condition, times, *, intervals, generator, stride, clean_context=False):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 2:
        raise ValueError("coarse_fine stride must be an integer >= 2")
    frames, atoms = times.shape[1], len(condition.reference_A)
    # For Q50/stride5 these are exactly 4,9,...,49. A shorter final interval
    # retains the last query as an anchor for other finite query lengths.
    indices = list(range(stride-1, frames, stride))
    if not indices or indices[-1] != frames-1:
        indices.append(frames-1)
    anchor_index = torch.tensor(indices, device=times.device, dtype=torch.long)
    coarse_times = times.index_select(1, anchor_index)
    # Draw once in the original full physical-frame order, exactly as joint.
    noise = torch.randn((frames, *condition.reference_A.shape),
                        device=condition.reference_A.device, generator=generator)
    anchors = flow_heun(model, condition, coarse_times,
                        noise.index_select(0, anchor_index), intervals)
    fine_condition = condition
    if clean_context:
        if not model.settings.get('endpoint_conditioning', False):
            raise ValueError('Clean anchor context requires explicit endpoint-conditioning architecture')
        known_mask = torch.zeros_like(times, dtype=torch.bool)
        known_mask[:, anchor_index] = True
        fine_condition = model.prepare_condition(condition.observed,
            known_coords_A=anchors, known_time_ps=coarse_times,
            query_time_ps=times, query_known_mask=known_mask)
        if not torch.equal(fine_condition.reference_A, condition.reference_A):
            raise RuntimeError('Known future conditions must not move the observed reference')
    source = condition.reference_A[None] + model.rho_A*noise
    anchor_source = source.index_select(0, anchor_index)
    state = source.clone()
    dt = 1./intervals

    def clamp_anchor(candidate, progress):
        # All packed systems share this scalar RF schedule. No atom/sample
        # aggregation or alignment is introduced by the sampling protocol.
        path = (1.-progress)*anchor_source + progress*anchors
        candidate.index_copy_(0, anchor_index, path)
        return candidate

    for index in range(intervals):
        s = state.new_full((condition.observed.batch_size,), index/intervals)
        clamp_anchor(state, s[0])
        u = model(state, condition=fine_condition, query_time_ps=times, progress=s)
        next_s = s+dt
        proposal = clamp_anchor(state+dt*u, next_s[0])
        if index < intervals-1:
            u_next = model(proposal, condition=fine_condition, query_time_ps=times, progress=next_s)
            state = clamp_anchor(state+.5*dt*(u+u_next), next_s[0])
        else:
            state = proposal
    # Exact endpoint copy avoids rounding in 1-s and in the final Euler step.
    state.index_copy_(0, anchor_index, anchors)
    endpoint_error = float((state.index_select(0, anchor_index)-anchors).abs().max())
    nfe = 2*intervals-1
    anchor_set = set(indices)
    coarse_effective = [nfe if i in anchor_set else 0 for i in range(frames)]
    fine_effective = [0 if i in anchor_set else nfe for i in range(frames)]
    model.last_sampling_info = dict(
        mode="coarse_fine_clean" if clean_context else "coarse_fine",
        protocol="coarse_fine_clean" if clean_context else "coarse_fine", stride_frames=stride,
        history_frames=condition.observed.frames,
        history_time_ps=condition.observed.time_ps.detach().cpu().tolist(),
        query_time_ps=times.detach().cpu().tolist(),
        anchor_indices=indices, coarse_query_time_ps=coarse_times.detach().cpu().tolist(),
        coarse_frames=len(indices), fine_frames=frames,
        coarse_NFE=nfe, fine_NFE=nfe, total_NFE=2*nfe,
        effective_intervals_per_frame=[intervals]*frames,
        effective_evaluations_per_frame=[nfe]*frames,
        coarse_effective_intervals_per_frame=[intervals if i in anchor_set else 0 for i in range(frames)],
        fine_effective_intervals_per_frame=[0 if i in anchor_set else intervals for i in range(frames)],
        computed_evaluations_per_frame=[nfe+v for v in coarse_effective],
        coarse_effective_evaluations_per_frame=coarse_effective,
        fine_effective_evaluations_per_frame=fine_effective,
        fine_clamped_evaluations_per_frame=coarse_effective,
        coarse_atom_frame_evaluations=nfe*len(indices)*atoms,
        fine_atom_frame_evaluations=nfe*frames*atoms,
        atom_frame_evaluations=nfe*(len(indices)+frames)*atoms,
        effective_atom_frame_evaluations=nfe*frames*atoms,
        fine_discarded_anchor_atom_frame_evaluations=nfe*len(indices)*atoms,
        source_noise_draws=1, source_noise_shape=list(noise.shape),
        source_policy="one_full_query_normal_draw; coarse_reuses_physical_frame_subset; fine_reuses_full_draw",
        anchor_path="(1-s)*original_RF_source + s*model_generated_coarse_anchor",
        anchor_final_max_difference_A=endpoint_error,
        anchor_fine_policy="field_computed_for_context_but_anchor_update_discarded_and_reclamped",
        condition_policy="original_reference_and_history_unchanged; fine appends explicit clean generated anchors" if clean_context else "original_observed_history_and_reference_unchanged_in_both_stages",
        clean_anchor_context=clean_context,
        endpoint_conditioning_trained=model.settings.get('endpoint_training', {}).get('mode') == 'endpoint_mixture',
        interpretation="explicit_clean_anchor_context_with_RF_path_clamping" if clean_context else "RF_inference_inpainting_approximation_not_BioKinema_endpoint_conditioning",
    )
    model.last_sampling_artifacts = dict(anchor_coordinates_A=anchors.detach(),
        anchor_indices=anchor_index.detach(), coarse_query_time_ps=coarse_times.detach())
    return state


@torch.no_grad()
def sample_with_protocol(model, condition, query_time_ps, protocol="joint",
                         intervals=None, generator=None, stride=5):
    """Return generated XYZ[Q,N,3] in Angstrom using observed-only conditions.

    ``joint`` delegates unchanged to the native sampler (including non-RF
    models). The additional protocols currently require TIME-compatible RF
    control and never change its checkpoint settings or ``temporal_mode``.
    ``short`` retains native H4/H8 generated-history blocks. ``coarse_fine``
    retains the original observed history while inpainting model-made anchors.
    Metadata is in ``last_sampling_info``; coarse anchor tensors alone are in
    ``last_sampling_artifacts`` and must not be embedded into JSON metadata.
    """
    if protocol == "q4":
        protocol = "short"
    if protocol == "joint":
        model.last_sampling_artifacts = {}
        count = _validate_flow_control(model, condition, query_time_ps, intervals)
        noise = torch.randn((query_time_ps.shape[1], *condition.reference_A.shape), device=condition.reference_A.device, generator=generator)
        value = flow_heun(model, condition, query_time_ps, noise, count)
        frames=query_time_ps.shape[1]; nfe=2*count-1
        model.last_sampling_info=dict(mode='joint', protocol='joint', total_NFE=nfe,
            effective_evaluations_per_frame=[nfe]*frames, effective_intervals_per_frame=[count]*frames,
            atom_frame_evaluations=nfe*frames*len(condition.reference_A), source_noise_shape=list(noise.shape))
        return value
    if protocol not in ("short", "q12", "coarse_fine", "coarse_fine_clean"):
        raise ValueError(f"Unknown sampling protocol: {protocol!r}")
    count = _validate_flow_control(model, condition, query_time_ps, intervals)
    model.last_sampling_artifacts = {}
    if protocol == "q12":
        return _fixed_q12(model, condition, query_time_ps, intervals=count, generator=generator)
    if protocol == "short":
        coordinates = sample_short_blocks(model, condition, query_time_ps,
                                         intervals=count, generator=generator)
        model.last_sampling_info.update(protocol="short",
            query_time_ps=query_time_ps.detach().cpu().tolist(),
            source_noise_draws=1,
            source_policy="one_full_query_normal_draw; each_block_reference_is_generated_history_endpoint")
        return coordinates
    return _coarse_fine(model, condition, query_time_ps,
                        intervals=count, generator=generator, stride=stride,
                        clean_context=protocol == "coarse_fine_clean")


@torch.no_grad()
def _fixed_q12(model, condition, times, *, intervals, generator):
    """Five native H4Q12 blocks; save all60, return the prespecified first50."""
    observed = condition.observed
    if observed.frames != 4 or times.shape[1] != 50:
        raise ValueError('This finite protocol is exactly H4, 5xQ12, first Q50 scored')
    delta = times[:, 0]-observed.time_ps[:, -1]
    full_times = observed.time_ps[:, -1:] + delta[:, None]*torch.arange(1, 61, device=times.device)[None]
    if not torch.allclose(times, full_times[:, :50], atol=1e-4, rtol=1e-6):
        raise ValueError('Q12 rollout requires the native uniform physical clock')
    noise = torch.randn((60, *condition.reference_A.shape), device=condition.reference_A.device, generator=generator)
    history_x, history_t = observed.x, observed.time_ps
    pieces = []
    for begin in range(0, 60, 12):
        current = condition if begin == 0 else model.prepare_condition(
            history_with_coordinates(observed, history_x[-4:], history_t[:, -4:]))
        part = flow_heun(model, current, full_times[:, begin:begin+12], noise[begin:begin+12], intervals)
        pieces.append(part)
        history_x = torch.cat((history_x, part))[-4:]
        history_t = torch.cat((history_t, full_times[:, begin:begin+12]), 1)[:, -4:]
    full = torch.cat(pieces)
    nfe = 2*intervals-1
    model.last_sampling_info = dict(mode='q12', protocol='q12', block_size=12,
        block_lengths=[12]*5, history_frames=4, generated_frames=60, scored_frames=50,
        total_NFE=5*nfe, effective_evaluations_per_frame=[nfe]*60,
        effective_intervals_per_frame=[intervals]*60,
        atom_frame_evaluations=nfe*60*len(condition.reference_A),
        query_time_ps=times.detach().cpu().tolist(), full_query_time_ps=full_times.detach().cpu().tolist(),
        source_noise_shape=list(noise.shape), source_noise_draws=1,
        condition_policy='previous_generated_H4_no_MD_reset_no_block_alignment',
        last_scored_block='first2_of12; excluded from full-block trend comparison')
    model.last_sampling_artifacts = dict(full_coordinates_A=full.detach(), full_query_time_ps=full_times.detach())
    return full[:50]
