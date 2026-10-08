"""Actual mixed-pretraining loop; checkpoints are recovery and fixed-final only."""
from dataclasses import replace
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from ..codec.frame import slice_clip_frames
from ..data.bridge_mixture import BridgeMixture
from ..flow.pretrained_bridge import PretrainedGeometryBridge
from ..runtime import atomic_write_json
from .predictive_data import atomic_torch_save
from .pretrained_bridge import bridge_objective


def load_geometry(settings, device):
    root = Path(settings['pvb_root']).resolve()
    sys.path.insert(0, str(root))
    import module.model as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Original PVB import namespace required')
    geometry = torch.load(settings['checkpoint'], map_location='cpu', weights_only=False)
    return geometry.to(device).requires_grad_(True)


def training_example(model, clean, task, history, queries, settings, generator):
    """Observed-only conditioning; clean static and future positions are labels."""
    if task == 'static':
        noise = torch.randn(clean.x.shape, device=clean.x.device, generator=generator)
        corrupted = clean.x + settings['static_sigma_A'] * noise
        # Recompute block centers as well: no clean-coordinate metadata bypass.
        _, group = torch.unique(torch.stack((clean.abid, clean.block_id), -1),
                                dim=0, return_inverse=True)
        sums = corrupted.new_zeros(1, int(group.max()) + 1, 3).index_add_(1, group, corrupted)
        centers = sums / torch.bincount(group)[None, :, None]
        observed = replace(clean, x=corrupted, bpos=centers[:, group])
        condition = model.prepare_condition(observed, task='static', generator=generator)
        target, times = clean.x, clean.time_ps
    else:
        observed = slice_clip_frames(clean, 0, history)
        condition = model.prepare_condition(observed, source_perturbation_A=settings['dynamic_source_sigma_A'],
                                            generator=generator)
        target, times = clean.x[queries], clean.time_ps[:, queries]
    return bridge_objective(model, condition, target, times, generator=generator)


def train(settings, *, output, seed, resume=None, diagnose_step=None, warmstart=None):
    if resume is not None and warmstart is not None:
        raise ValueError('Exact resume and changed-architecture continuation are distinct')
    if settings.get('warmstart_step') and resume is None and warmstart is None:
        raise ValueError('This experiment requires its same-seed trained parent')
    output = Path(output)
    if output.exists() and resume is None:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA training required; no CPU fallback')
    device = torch.device('cuda:0')
    torch.set_num_threads(settings['cpu_threads'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    geometry = load_geometry(settings, device)
    inherited = sum(p.numel() for p in geometry.parameters())
    model = PretrainedGeometryBridge(geometry, recompute_frames=True,
        temporal_insertion_layer=settings.get('temporal_insertion_layer'),
        collective_feedback=settings.get('collective_feedback', False)).to(device).train()
    initialization = dict(checkpoint=settings['checkpoint'], loaded_geometry_parameters=inherited,
        mode='released geometry with new temporal modules')
    if warmstart is not None:
        saved = torch.load(warmstart, map_location='cpu', weights_only=False)
        if (saved['seed'] != seed or saved['step'] != settings['warmstart_step'] or
                saved['step'] != saved['settings']['updates'] or
                saved['settings'].get('temporal_insertion_layer') != settings['temporal_insertion_layer']):
            raise ValueError('Continuation requires the same-seed final placement parent')
        expected_new = {k for k in model.state_dict() if k.startswith('collective.')}
        if set(model.state_dict()) - set(saved['model']) != expected_new or set(saved['model']) - set(model.state_dict()):
            raise ValueError('Only collective feedback parameters may be new')
        loaded = model.load_state_dict(saved['model'], strict=False)
        if set(loaded.missing_keys) != expected_new or loaded.unexpected_keys:
            raise ValueError('Unexpected inherited-weight mismatch')
        initialization = dict(checkpoint=str(Path(warmstart).resolve()), parent_step=saved['step'],
            parent_settings=saved['settings'], parent_contract=saved['contract'],
            mode='placement continuation; inherited geometry and time weights; fresh optimizer and sampling schedule',
            missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys),
            new_collective_parameters=sum(p.numel() for p in model.collective.parameters()) if model.collective else 0)
        del saved
    parameter_groups = [
        dict(params=model.geometry.parameters(), lr=settings['geometry_lr'], name='geometry'),
        dict(params=model.conditioner.parameters(), lr=settings['temporal_lr'], name='temporal')]
    if model.collective is not None:
        parameter_groups.append(dict(params=model.collective.parameters(), lr=settings['collective_lr'], name='collective'))
    optimizer = torch.optim.AdamW(parameter_groups,
        weight_decay=settings['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 81007)
    mixture = BridgeMixture(settings['data'], seed)
    start = 0
    resume_arithmetic_unchanged = None
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['settings'] != settings or saved['seed'] != seed:
            raise ValueError('Exact resume requires the same settings and seed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        mixture.load_state_dict(saved['mixture'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['generator_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        start = saved['step']
        resume_arithmetic_unchanged = (saved['contract'].get('scalar_output_normalization') ==
                                      model.contract().get('scalar_output_normalization'))
        initialization = saved.get('initialization', initialization)
        del saved
    atomic_write_json(output/'recipe.json', dict(settings=settings, seed=seed, contract=model.contract(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        inheritance=initialization,
        device=torch.cuda.get_device_name(0), resume=None if resume is None else str(resume),
        resume_arithmetic_unchanged=resume_arithmetic_unchanged,
        resume_state_policy='restore model optimizer sampler and all RNG; numerical-kernel repair is not a bitwise-identical continuation'))
    started = time.monotonic()
    try:
        with (output/'updates.jsonl').open('a', buffering=1) as log:
            for step in range(start, settings['updates']):
                if step + 1 == diagnose_step:
                    # Exact pre-update state for one failed optimizer update;
                    # no sample skipping, altered precision, or loss changes.
                    atomic_torch_save(output/'checkpoint_pre_failure.pt', dict(
                        model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings, initialization=initialization,
                        contract=model.contract(), step=step, seed=seed, mixture=mixture.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        generator_rng=generator.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state()))
                batch, task, history, queries, identity = mixture.next()
                batch = batch.to(device)
                warmup = settings['warmup_updates']
                progress = max(0., (step-warmup) / max(1, settings['updates']-warmup-1))
                factor = ((step+1)/warmup if step < warmup else .1 + .45*(1 + math.cos(math.pi*progress)))
                for group in optimizer.param_groups:
                    group['lr'] = settings[group['name']+'_lr'] * factor
                optimizer.zero_grad(set_to_none=True)
                try:
                    context = torch.autograd.detect_anomaly(check_nan=True) if step + 1 == diagnose_step else nullcontext()
                    with context:
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings['bf16']):
                            loss, metrics = training_example(model, batch, task, history, queries, settings, generator)
                        loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad'], error_if_nonfinite=True)
                except (RuntimeError, FloatingPointError) as error:
                    atomic_write_json(output/'failure.json', dict(status='failed', attempted_step=step+1,
                        completed_step=step, task=task, sample=batch.sample_id[0], history=history,
                        query_indices=queries, identity=identity, error=str(error),
                        nonfinite_gradient_parameters=[name for name, p in model.named_parameters()
                            if p.grad is not None and not bool(torch.isfinite(p.grad).all())]))
                    raise
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite bridge objective')
                optimizer.step()
                row = dict(step=step+1, loss=float(loss.detach()), grad_norm=float(norm), task=task,
                    sample=batch.sample_id[0], atoms=batch.atom_count, history=history, query_indices=queries,
                    query_time_ps=batch.time_ps[:, queries].tolist(), elapsed_seconds=time.monotonic()-started,
                    output_norm_float64_calls_since_process_start=model.geometry.decoder.out_norm.high_precision_calls,
                    **identity, **{k:float(v) for k,v in metrics.items()})
                log.write(json.dumps(row)+'\n')
                if step < 5 or (step+1) % 16 == 0:
                    print(json.dumps(row), flush=True)
                if (step+1) % settings['save_every'] == 0 or step+1 == settings['updates']:
                    if shutil.disk_usage(output).free < settings['reserve_bytes']:
                        raise OSError('Training checkpoint disk reserve exhausted')
                    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings, initialization=initialization,
                        contract=model.contract(), step=step+1, seed=seed, mixture=mixture.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        generator_rng=generator.get_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state())
                    path = output/('bridge_final.pt' if step+1 == settings['updates'] else 'checkpoint_latest.pt')
                    atomic_torch_save(path, payload)
                    del payload
                    atomic_write_json(output/'status.json', dict(status='completed' if step+1 == settings['updates'] else 'training',
                        step=step+1, elapsed_seconds=time.monotonic()-started, counts=dict(mixture.counts),
                        checkpoint=str(path), peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                del loss, metrics, batch
    finally:
        mixture.close()
