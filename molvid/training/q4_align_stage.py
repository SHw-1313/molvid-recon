"""Finite Q4 teacher-history alignment on the frozen T joint/Q1 slot plan.

512 slots are not always 512 optimizer updates: shared_joint_only advances but
does not optimize the 256 Q1 slots. All LR positions refer to the original slot.
Only task adapters train; chunks use real visible history, never model rollout.
"""
import copy
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from ..codec.frame import slice_clip_frames
from ..data.bridge_mixture import BridgeMixture
from ..flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from ..flow.local_geometry_loss import ligand_geometry_band_loss
from ..runtime import atomic_write_json
from .coordinate_state import (LigandSourceGeometry, continuation_learning_rate,
                               load_coordinate_weights, _global_rng_state,
                               _set_global_rng_state)
from .kernel_tasks import KernelTasks
from .predictive_data import atomic_torch_save

ARMS = ('shared_joint_only', 'shared_q4_q1', 'routed_q4_q1')


def joint_q4_chunks(batch, history, queries):
    """Yield (observed H4, target Q4, physical times, target slice, metadata).

    Original H4Q12 -> windows [0:8,4:12,8:16]; H8Q8 -> [4:12,8:16].
    The target slice indexes the ORIGINAL target/normal array, not the clip.
    """
    if (history not in (4, 8) or batch.frames != 16 or
            list(queries) != list(range(history, 16)) or not bool(batch.frame_mask.all())):
        raise ValueError('Q4 splitting requires complete original H4Q12 or H8Q8')
    for offset in range(0, len(queries), 4):
        begin = history + offset
        observed = slice_clip_frames(batch, begin - 4, begin)
        yield (observed, batch.x[begin:begin + 4], batch.time_ps[:, begin:begin + 4],
               slice(offset, offset + 4),
               dict(history_indices=list(range(begin - 4, begin)),
                    target_indices=list(range(begin, begin + 4)),
                    reference_time_ps=observed.time_ps[:, -1].tolist()))


def _rf_with_draw(model, condition, target_A, query_time_ps, progress, noise):
    """Exact original flow expression, with an explicitly shared slot draw."""
    batch = condition.observed
    source = condition.reference_A[None] + model.rho_A * noise
    s = progress[batch.abid][None, :, None]
    current = (1. - s) * source + s * target_A
    target = target_A - source
    prediction = model(current, condition=condition, query_time_ps=query_time_ps, progress=progress)
    weight = torch.full_like(progress, 1. / model.sigma_data**2)
    error = (prediction.float() - target.float()).square().mean(-1)
    losses = []
    for sample in range(batch.batch_size):
        valid = batch.loss_mask & (batch.abid == sample)
        if not bool(valid.any()):
            raise ValueError('Every system needs valid target atoms')
        losses.append(error[:, valid].mean() * weight[sample])
    return torch.stack(losses).mean(), float(error[:, batch.loss_mask].mean().detach())


def backward_joint_q4(model, batch, history, queries, *, generator, bf16):
    """One original full-target progress/normal draw; accumulate weighted Q4 RF."""
    target = batch.x[queries]
    progress = torch.rand(batch.batch_size, device=target.device, generator=generator)
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    total = raw = 0.
    chunks = []
    for observed, part, times, target_slice, metadata in joint_q4_chunks(batch, history, queries):
        fraction = len(part) / len(target)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            condition = model.prepare_condition(observed, task='dynamic', route_task='joint')
            loss, mse = _rf_with_draw(model, condition, part, times, progress, noise[target_slice])
            weighted = fraction * loss
        weighted.backward()
        total += float(weighted.detach()); raw += fraction * mse
        chunks.append(dict(metadata, weight=fraction, RF_loss=float(loss.detach()),
                           source_epsilon_sum=float(noise[target_slice].double().sum()),
                           source_epsilon_square_sum=float(noise[target_slice].double().square().sum())))
        del condition, loss, weighted
    return dict(RF_loss=total, raw_coordinate_MSE_A2=raw,
                progress_mean=float(progress.mean()), progress_values=progress.tolist(),
                source_epsilon_sum=float(noise.double().sum()),
                source_epsilon_square_sum=float(noise.double().square().sum()),
                valid_atoms=int(batch.loss_mask.sum()), query_frames=len(target),
                chunks=chunks, RF_backward_calls=len(chunks))


def _tail_with_noise(model, observed, times, noise, tail_intervals):
    """Original truncated sampler math with a slice of one full-window normal.

    This private entry is needed only to preserve full-shape CUDA RNG consumption
    while splitting auxiliary generation into Q4; it changes no model method.
    """
    intervals = model.settings['sampler_intervals']
    if model.scheme != 'flow' or (intervals, tail_intervals) != (18, 4):
        raise ValueError('Frozen original 18-interval / tail4 RF required')
    with torch.no_grad(), torch.autocast(device_type=observed.x.device.type,
            enabled=torch.is_autocast_enabled(observed.x.device.type),
            dtype=torch.get_autocast_dtype(observed.x.device.type), cache_enabled=False):
        condition = model.prepare_condition(observed, route_task='joint')
        state = condition.reference_A[None] + model.rho_A * noise
        for index in range(intervals - tail_intervals):
            state = model._flow_interval(state, condition, times, index, intervals)
    state = state.detach()
    condition = model.prepare_condition(observed, route_task='joint')
    for index in range(intervals - tail_intervals, intervals):
        state = model._flow_interval(state, condition, times, index, intervals, recompute=True)
    return state


def backward_q4_auxiliary(model, batch, history, queries, auxiliary, *, bf16):
    """One due event, full original band and normal draw, weighted Q4 tails."""
    began = time.monotonic()
    target = batch.x[queries]
    if not bool((batch.loss_mask & (batch.component_id > 0)).any()):
        return dict(ligand_geometry_applied=False, ligand_geometry_weighted_loss=0.,
                    ligand_geometry_reason='no_valid_ligand')
    total = 0.; chunks = []
    with auxiliary.isolated_rng():
        noise = torch.randn(target.shape, device=target.device, generator=auxiliary.generator)
        for observed, part, times, target_slice, metadata in joint_q4_chunks(batch, history, queries):
            fraction = len(part) / len(target)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                endpoint = _tail_with_noise(model, observed, times, noise[target_slice],
                                            auxiliary.config['tail_intervals'])
                penalty, metrics = ligand_geometry_band_loss(endpoint, part, observed,
                    auxiliary.config['band'], band_reference_A=target)
                weighted = fraction * auxiliary.config['weight'] * penalty
            weighted.backward()
            value = float(weighted.detach()); total += value
            chunks.append(dict(metadata, weight=fraction, weighted_loss=value, metrics=metrics))
            del endpoint, penalty, weighted
        auxiliary.events += 1
    return dict(ligand_geometry_applied=True, ligand_geometry_weighted_loss=total,
                ligand_geometry_weight=auxiliary.config['weight'], ligand_geometry_events=auxiliary.events,
                ligand_geometry_counts=dict(auxiliary.counts), ligand_geometry_chunks=chunks,
                geometry_band_frames=len(target), ligand_geometry_seconds=time.monotonic()-began)


def backward_slot(model, batch, history, queries, *, arm, kind, identity,
                  band_reference, generator, auxiliary, bf16):
    """RF and optional ligand gradients for one ACTIVE slot; no clip/step here."""
    if arm not in ARMS or kind not in ('joint', 'kernel'):
        raise ValueError('Unknown Q4 arm or stream')
    if arm == 'shared_joint_only' and kind == 'kernel':
        raise ValueError('Inactive Q1 slots advance data only; do not call backward_slot')
    route_task = 'joint' if kind == 'joint' else 'q1'
    target, times = batch.x[queries], batch.time_ps[:, queries]
    split = kind == 'joint' and arm != 'shared_joint_only'
    if split:
        metrics = backward_joint_q4(model, batch, history, queries, generator=generator, bf16=bf16)
    else:
        observed = slice_clip_frames(batch, 0, history)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            condition = model.prepare_condition(observed, task='dynamic', route_task=route_task)
            loss, metrics = coordinate_state_objective(model, condition, target, times, generator=generator)
        loss.backward()
        metrics.update(RF_loss=float(loss.detach()), RF_backward_calls=1)
        del condition, loss
    # Same original slot/H visit counter, exactly once even for three Q4 chunks.
    triggered = auxiliary.due(identity, history)
    metrics.update(ligand_geometry_triggered=triggered, ligand_geometry_counts=dict(auxiliary.counts))
    if triggered:
        if split:
            metrics.update(backward_q4_auxiliary(model, batch, history, queries, auxiliary, bf16=bf16))
        else:
            metrics.update(auxiliary.backward(model, observed, target, times, bf16=bf16,
                band_reference=band_reference, route_task=route_task))
    metrics.update(loss=metrics['RF_loss']+metrics.get('ligand_geometry_weighted_loss', 0.),
                   route_task=route_task, task_adapter_mixing_weights=list(model.task_adapters.mixing_weights(route_task)),
                   geometry_band_frames=len(target) if band_reference is None else len(band_reference))
    return metrics


def validate_settings(settings, seed):
    arm = settings['q4_align']['arm']
    if arm not in ARMS or seed != 71 or settings['updates'] != 512 or settings['warmup_updates'] != 128:
        raise ValueError('Frozen three-arm seed71 / 512-slot / warmup128 recipe required')
    expected_mode = 'routed' if arm == 'routed_q4_q1' else 'shared'
    if settings['task_adapter'] != dict(mode=expected_mode, bottleneck=64):
        raise ValueError('Only frozen T 64-channel shared/routed adapters train')
    if (settings['scheme'] != 'flow' or settings['kernel_training']['arm'] != 'history_mix'
            or settings.get('static_replay') or settings.get('reference_concat', False)):
        raise ValueError('Use original history_mix RF without additional mechanisms')
    if settings['ligand_geometry']['weight_status'] != 'calibrated':
        raise ValueError('Original calibrated ligand auxiliary required')
    return arm


def train_q4_align(settings, output, seed, *, warm_start=None, resume=None, stop_after=None):
    """CUDA training entry. stop_after and checkpoint step refer to SLOT, not update."""
    arm = validate_settings(settings, seed)
    if (warm_start is None) == (resume is None):
        raise ValueError('Choose weights-only parent or exact resume')
    if stop_after is not None and not 0 < stop_after <= 512:
        raise ValueError('stop_after must lie within the unchanged 512-slot schedule')
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; no CPU model fallback')
    output = Path(output); output.mkdir(parents=True, exist_ok=resume is not None)
    torch.set_num_threads(settings['cpu_threads'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda:0')
    model = CoordinateStateDenoiser(settings).to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad != n.startswith('task_adapters.') for n,p in model.named_parameters()):
        raise ValueError('Only task adapters may train')
    if sum(p.numel() for p in parameters) != 166528:
        raise ValueError('Frozen T adapter capacity differs')
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed+20261008)
    mixture = BridgeMixture(settings['data'], seed)
    kernel = KernelTasks(settings, seed, device, generator)
    ligand = LigandSourceGeometry(settings['ligand_geometry'], seed, device)
    kernel_ligand = LigandSourceGeometry(settings['ligand_geometry'], seed+20261027, device)
    start = 0; prior_seconds = 0.; last_optimizer_lr = None
    exposure = dict(updates=0, dynamic_examples=0, static_examples=0, target_frames=0,
                    atom_query_pairs=0, joint_updates=0, kernel_updates=0, RF_backward_calls=0,
                    skipped_kernel_slots=0)
    if warm_start is not None:
        initialization = load_coordinate_weights(model, warm_start)
    else:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if (saved['schema'] != 'molvid.coordinate_state.checkpoint.v1' or
                saved.get('stage_schema') != 'molvid.q4_align.v1' or
                saved['settings'] != settings or saved['seed'] != seed or saved['slot'] != saved['step']):
            raise ValueError('Exact Q4 resume changes schema/recipe/seed/slot')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        mixture.load_state_dict(copy.deepcopy(saved['mixture']))
        generator.set_state(saved['generator_rng']); kernel.load_state_dict(saved['kernel_tasks'])
        ligand.load_state_dict(saved['ligand_geometry']); kernel_ligand.load_state_dict(saved['kernel_ligand_geometry'])
        _set_global_rng_state(saved['global_rng'])
        start = saved['slot']; exposure = copy.deepcopy(saved['exposure'])
        if saved['optimizer_updates'] != exposure['updates']:
            raise ValueError('Saved optimizer-update count differs')
        prior_seconds = saved['elapsed_seconds']; initialization = saved['initialization_report']
        last_optimizer_lr = saved['last_optimizer_lr']
        del saved
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    atomic_write_json(output/'recipe.json', dict(settings=settings, seed=seed, model=model.contract(),
        source_commit=source_commit, initialization_report=initialization, resume=str(resume),
        CUDA_device=torch.cuda.get_device_name(0), schedule_unit='slot', planned_slots=512,
        planned_optimizer_updates=256 if arm == 'shared_joint_only' else 512,
        chunk_history='real teacher history only; no generated training prefix',
        geometry_band='original full joint target8/12 or original kernel support16'))
    began = time.monotonic()

    def save(slot, final=False):
        elapsed = prior_seconds + time.monotonic()-began
        rng = _global_rng_state()
        state = dict(schema='molvid.coordinate_state.checkpoint.v1', stage_schema='molvid.q4_align.v1',
            model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings, seed=seed,
            step=slot, slot=slot, optimizer_updates=exposure['updates'], last_optimizer_lr=last_optimizer_lr,
            contract=model.contract(), mixture=mixture.state_dict(), kernel_tasks=kernel.state_dict(),
            exposure=copy.deepcopy(exposure), elapsed_seconds=elapsed, initialization_report=initialization,
            source_commit=source_commit, generator_rng=generator.get_state(), global_rng=rng,
            torch_rng=rng['torch_cpu'], cuda_rng=rng['cuda'], python_rng=rng['python'], numpy_rng=rng['numpy'],
            ligand_geometry=ligand.state_dict(), kernel_ligand_geometry=kernel_ligand.state_dict())
        path = output/('coordinate_final.pt' if final else 'checkpoint_latest.pt')
        atomic_torch_save(path, state)
        if slot in (32, 256, 512):
            snapshot = output/f'checkpoint_step{slot}.pt'
            if not snapshot.exists():
                atomic_torch_save(snapshot, state)
        atomic_write_json(output/'status.json', dict(status='completed' if final else 'training',
            step=slot, slot=slot, optimizer_updates=exposure['updates'], checkpoint=str(path),
            elapsed_seconds=elapsed, GPU_hours=elapsed/3600, exposure=dict(exposure),
            last_optimizer_lr=last_optimizer_lr, peak_cuda_bytes=torch.cuda.max_memory_allocated()))

    try:
        with (output/'updates.jsonl').open('a', buffering=1) as log:
            for step in range(start, 512):
                batch, task, history, queries, identity, band, kind = kernel.next(step, mixture)
                if task != 'dynamic' or not bool(batch.frame_mask.all()):
                    raise ValueError('Only complete dynamic TRAIN windows are allowed')
                scheduled_lr = continuation_learning_rate(settings, step)
                optimizer.param_groups[0]['lr'] = scheduled_lr
                active = not (arm == 'shared_joint_only' and kind == 'kernel')
                # Data-only skips have no forward, auxiliary, clip or AdamW step.
                if not active:
                    exposure['skipped_kernel_slots'] += 1
                    metrics = dict(loss=None, RF_loss=None, grad_norm=None, RF_backward_calls=0,
                                   ligand_geometry_triggered=False)
                else:
                    with kernel.isolated_rng(kind) as active_generator:
                        batch = batch.to(device)
                        if band is not None:
                            band = band.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        auxiliary = kernel_ligand if kind == 'kernel' else ligand
                        metrics = backward_slot(model, batch, history, queries, arm=arm, kind=kind,
                            identity=identity, band_reference=band, generator=active_generator,
                            auxiliary=auxiliary, bf16=settings['bf16'])
                        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['clip_grad'], error_if_nonfinite=True)
                        if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('task_adapters.')):
                            raise RuntimeError('Frozen base received a gradient')
                        metrics['grad_norm'] = float(norm)
                        metrics['task_adapter_gradient_norms_after_clip'] = {
                            f'branch{i}':sum(float(p.grad.float().square().sum()) for p in branch.parameters()
                                            if p.grad is not None)**.5
                            for i,branch in enumerate(model.task_adapters.branches)}
                        optimizer.step(); last_optimizer_lr = scheduled_lr
                        exposure['updates'] += 1; exposure[f'{kind}_updates'] += 1
                        exposure['dynamic_examples'] += batch.batch_size
                        exposure['target_frames'] += len(queries)*batch.batch_size
                        exposure['atom_query_pairs'] += len(queries)*int(batch.loss_mask.sum())
                        exposure['RF_backward_calls'] += metrics['RF_backward_calls']
                # Merge mappings explicitly: diagnostics may contain query_frames;
                # no duplicate-key expansion can fail after a successful backward.
                row = dict(identity)
                row.update(metrics)
                row.update(step=step+1, slot=step+1, optimizer_updates=exposure['updates'],
                    optimized=active, task=task, sample=batch.sample_id[0], history=history,
                    queries=list(queries), query_frames=len(queries),
                    learning_rate=scheduled_lr, last_optimizer_lr=last_optimizer_lr,
                    kernel_counts=dict(kernel.counts), kernel_crossings=dict(kernel.crossings),
                    exposure=dict(exposure), elapsed_seconds=prior_seconds+time.monotonic()-began)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                if step < start+4 or (step+1) % 32 == 0:
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step+1 in (32, 256) or (step+1) % settings['save_every'] == 0:
                    save(step+1)
                if stop_after is not None and step+1 >= stop_after and step+1 < 512:
                    save(step+1)
                    return
            expected = 256 if arm == 'shared_joint_only' else 512
            if exposure['updates'] != expected or kernel.counts != dict(joint=256, kernel=256):
                raise RuntimeError('Final slot/optimizer exposure differs from the frozen plan')
            save(512, final=True)
    except Exception as error:
        atomic_write_json(output/'failure.json', dict(error=repr(error), exposure=exposure,
            elapsed_seconds=prior_seconds+time.monotonic()-began))
        raise
    finally:
        mixture.close(); kernel.close()
