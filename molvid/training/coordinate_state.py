"""Actual paired coordinate-state training; one shared loop for RF and EDM."""
from dataclasses import replace
import math
from pathlib import Path
import random
import subprocess
import time
import json

import numpy as np
import torch

from ..codec.frame import slice_clip_frames
from ..data.bridge_mixture import BridgeMixture
from ..flow.coordinate_state import CoordinateStateDenoiser, coordinate_state_objective
from ..runtime import atomic_write_json
from .predictive_data import atomic_torch_save


def load_coordinate_weights(model, path):
    """Weights-only stage transition; never restore optimizer, RNG or data visits.

    Coordinate units, noise preconditioning, output meaning and architecture
    must match. Data mixtures and optimization budgets may change explicitly.
    Only a completed fixed-budget parent is accepted, not a selected partial fit.
    """
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') != 'molvid.coordinate_state.checkpoint.v1':
        raise ValueError('Warm start requires an owned coordinate-state checkpoint, not h/v or external weights')
    parent = saved['settings']
    semantic_keys = ('schema', 'scheme', 'scales', 'width', 'spatial_layers',
                     'temporal_heads', 'num_rbf', 'radius_A', 'max_neighbors')
    changed = [key for key in semantic_keys if parent[key] != model.settings[key]]
    if changed:
        raise ValueError(f'Incompatible coordinate warm start: {changed}')
    if saved['step'] != parent['updates']:
        raise ValueError('Use a completed fixed-budget parent for a new training stage')
    model.load_state_dict(saved['model'], strict=True)
    return dict(kind='coordinate_state_weights_only', parent_checkpoint=str(Path(path).resolve()),
        parent_step=saved['step'], parent_seed=saved['seed'], parent_exposure=saved['exposure'],
        parent_source_commit=saved.get('source_commit'),
        parent_GPU_hours=saved['elapsed_seconds']/3600,
        parent_initialization=saved.get('initialization_report', {'kind': parent['initialization']}),
        inherited_parameters=sum(p.numel() for p in model.parameters()),
        inherited_modules=['spatial', 'temporal', 'tokens', 'progress', 'physical_time', 'displacement', 'head'],
        frozen_modules=[], newly_initialized_modules=[],
        reset=['optimizer', 'schedule', 'RNG', 'data_visits', 'stage_exposure', 'stage_GPU_hours'])


def train(settings, output, seed, resume=None, warm_start=None):
    if resume is not None and warm_start is not None:
        raise ValueError('Exact resume and weights-only warm start are mutually exclusive')
    if resume is None and (warm_start is not None) != (settings['initialization']=='coordinate_state_weights_only'):
        raise ValueError('Weights-only initialization must be explicit in both recipe and CLI')
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; no CPU model fallback')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume is not None)
    torch.set_num_threads(settings['cpu_threads'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda:0')
    model = CoordinateStateDenoiser(settings).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed+20261008)
    mixture = BridgeMixture(settings['data'], seed)
    start, prior_seconds = 0, 0.
    exposure = dict(updates=0, dynamic_examples=0, static_examples=0, target_frames=0, atom_query_pairs=0)
    initialization_report = dict(kind=settings['initialization'], inherited_parameters=0,
        frozen_modules=[], newly_initialized_modules=['all'])
    if warm_start is not None:
        initialization_report = load_coordinate_weights(model, warm_start)
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['schema']!='molvid.coordinate_state.checkpoint.v1' or saved['settings']!=settings or saved['seed']!=seed:
            raise ValueError('Wrong objective, scales, seed or checkpoint schema')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        mixture.load_state_dict(saved['mixture']); generator.set_state(saved['generator_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        start=saved['step']; exposure=saved['exposure']; prior_seconds=saved['elapsed_seconds']
        initialization_report=saved.get('initialization_report', initialization_report)
        del saved
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    atomic_write_json(output/'recipe.json', dict(settings=settings, seed=seed, model=model.contract(),
        source_commit=source_commit,
        CUDA_device=torch.cuda.get_device_name(0), resume=str(resume), initialization=settings['initialization'],
        initialization_report=initialization_report))
    began=time.monotonic()

    def save(step, final=False):
        elapsed=prior_seconds+time.monotonic()-began
        path=output/('coordinate_final.pt' if final else 'checkpoint_latest.pt')
        atomic_torch_save(path, dict(schema='molvid.coordinate_state.checkpoint.v1', model=model.state_dict(),
            optimizer=optimizer.state_dict(), settings=settings, seed=seed, step=step, contract=model.contract(),
            mixture=mixture.state_dict(), exposure=exposure, elapsed_seconds=elapsed,
            initialization_report=initialization_report, source_commit=source_commit,
            generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state()))
        atomic_write_json(output/'status.json', dict(status='completed' if final else 'training', step=step,
            checkpoint=str(path), elapsed_seconds=elapsed, GPU_hours=elapsed/3600,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), exposure=exposure))

    try:
        with (output/'updates.jsonl').open('a', buffering=1) as log:
            for step in range(start, settings['updates']):
                batch, task, history, queries, identity=mixture.next()
                if not bool(batch.frame_mask.all()):
                    raise ValueError('Pilot uses complete valid windows; no hidden padding')
                batch=batch.to(device)
                if task=='dynamic':
                    observed=slice_clip_frames(batch,0,history)
                    target, query_times=batch.x[queries],batch.time_ps[:,queries]
                else:
                    # Future static stages use corrupted input only. No clean
                    # target enters conditions or graph construction directly.
                    sigma=settings['static_corruption_A']
                    corrupted=batch.x+sigma*torch.randn(batch.x.shape,device=device,generator=generator)
                    _,group=torch.unique(torch.stack((batch.abid,batch.block_id),-1),dim=0,return_inverse=True)
                    centers=corrupted.new_zeros(1,int(group.max())+1,3).index_add_(1,group,corrupted)
                    centers=centers/torch.bincount(group)[None,:,None]
                    observed=replace(batch,x=corrupted,bpos=centers[:,group])
                    target,query_times=batch.x,batch.time_ps
                warmup=settings['warmup_updates']
                fraction=max(0.,(step-warmup)/max(1,settings['updates']-warmup-1))
                factor=(step+1)/warmup if step<warmup else .1+.45*(1+math.cos(math.pi*fraction))
                optimizer.param_groups[0]['lr']=settings['learning_rate']*factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings['bf16']):
                    condition=model.prepare_condition(observed,task=task)
                    loss,metrics=coordinate_state_objective(model,condition,target,query_times,generator=generator)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad'],error_if_nonfinite=True)
                # Report real first-update supervision of the static path;
                # no separate probe model, synthetic optimization or extra run.
                if step==start and task=='static':
                    metrics['static_shared_gradient_norms_after_clip']={
                        name:sum(float(p.grad.float().square().sum()) for p in module.parameters()
                                 if p.grad is not None)**.5
                        for name,module in [('current_geometry',model.spatial),
                                            ('shared_tokens',model.tokens),('coordinate_head',model.head)]}
                    if any(value==0 or not math.isfinite(value) for value in metrics['static_shared_gradient_norms_after_clip'].values()):
                        raise RuntimeError('Static supervision did not reach every shared geometry path')
                optimizer.step()
                exposure['updates']+=1
                exposure['dynamic_examples' if task=='dynamic' else 'static_examples']+=batch.batch_size
                exposure['target_frames']+=len(target)*batch.batch_size
                exposure['atom_query_pairs']+=len(target)*int(batch.loss_mask.sum())
                row=dict(step=step+1,loss=float(loss.detach()),grad_norm=float(norm),task=task,
                    sample=batch.sample_id[0],**identity,history=history,queries=queries,
                    elapsed_seconds=prior_seconds+time.monotonic()-began,**metrics)
                log.write(json.dumps(row)+'\n')
                if step<4 or (step+1)%64==0:
                    print(json.dumps(row),flush=True)
                if (step+1)%settings['save_every']==0:
                    save(step+1)
            save(settings['updates'],final=True)
    except Exception as error:
        atomic_write_json(output/'failure.json',dict(error=repr(error),exposure=exposure,
            elapsed_seconds=prior_seconds+time.monotonic()-began))
        raise
    finally:
        mixture.close()
