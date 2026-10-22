"""Actual paired coordinate-state training; one shared loop for RF and EDM."""
from dataclasses import replace
from contextlib import contextmanager
from collections import Counter
import copy
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


def _global_rng_state():
    """All default streams used by data preparation, dropout and checkpoints."""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def _set_global_rng_state(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['cuda'])


def _static_observation(batch, sigma_A, generator):
    """The existing corrupted-reference Q1 recipe; clean is only the target."""
    corrupted=batch.x+sigma_A*torch.randn(batch.x.shape,device=batch.x.device,generator=generator)
    _,group=torch.unique(torch.stack((batch.abid,batch.block_id),-1),dim=0,return_inverse=True)
    centers=corrupted.new_zeros(1,int(group.max())+1,3).index_add_(1,group,corrupted)
    centers=centers/torch.bincount(group)[None,:,None]
    return replace(batch,x=corrupted,bpos=centers[:,group])


class StaticReplay:
    """One Q1 backward on its own saved random streams; optimizer stays outside.

    Enter the private streams for the entire load/condition/forward/backward,
    including activation-checkpoint recomputation. Dynamic data, source,
    progress and default/dropout RNG states are restored before returning.
    """
    def __init__(self, config, seed, device):
        required={'enabled','every','weight','corruption_A','seed_offset','data'}
        if set(config)!=required or config['enabled'] is not True:
            raise ValueError('Replay requires the complete explicit enabled recipe')
        if (config['every'],config['weight'],config['corruption_A'])!=(4,.25,.2):
            raise ValueError('This finite replay recipe is every4, weight.25, corruption.2A')
        tasks=config['data']['sixteen_step_tasks']
        if Counter(tasks)!=Counter({'static:protein':8,'static:atlas':4,'static:misato':4}):
            raise ValueError('Replay requires the existing 50/25/25 TRAIN snapshot pool')
        self.config=copy.deepcopy(config)
        self.device=torch.device(device)
        private_seed=int(seed)+int(config['seed_offset'])
        self.generator=torch.Generator(device=self.device).manual_seed(private_seed+1)
        # Never torch.manual_seed: it changes every visible CUDA default stream.
        self.rng=dict(python=random.Random(private_seed).getstate(),
            numpy=np.random.RandomState(private_seed % 2**32).get_state(),
            torch_cpu=torch.Generator(device='cpu').manual_seed(private_seed+2).get_state(),
            cuda=[torch.Generator(device=f'cuda:{i}').manual_seed(private_seed+3+i).get_state()
                  for i in range(torch.cuda.device_count())])
        self.exposure=dict(backward_calls=0,static_examples=0,target_frames=0,atom_query_pairs=0)
        with self.isolated_rng():
            self.mixture=BridgeMixture(config['data'],private_seed)

    @contextmanager
    def isolated_rng(self):
        dynamic=_global_rng_state()
        _set_global_rng_state(self.rng)
        try:
            yield
        finally:
            self.rng=_global_rng_state()
            _set_global_rng_state(dynamic)

    def due(self, optimizer_step):
        return optimizer_step % self.config['every']==0

    def backward(self, model, *, bf16):
        began=time.monotonic()
        with self.isolated_rng():
            batch,task,history,queries,identity=self.mixture.next()
            if task!='static' or batch.frames!=1 or not bool(batch.frame_mask.all()):
                raise ValueError('Replay must contain exactly one valid TRAIN snapshot')
            batch=batch.to(self.device)
            observed=_static_observation(batch,self.config['corruption_A'],self.generator)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                condition=model.prepare_condition(observed,task='static')
                loss,metrics=coordinate_state_objective(model,condition,batch.x,batch.time_ps,
                                                        generator=self.generator)
                weighted=self.config['weight']*loss
            weighted.backward()
            self.exposure['backward_calls']+=1
            self.exposure['static_examples']+=batch.batch_size
            self.exposure['target_frames']+=batch.batch_size
            self.exposure['atom_query_pairs']+=int(batch.loss_mask.sum())
            return dict(sample=batch.sample_id[0],**identity,history=history,queries=queries,
                RF_loss=float(loss.detach()),weight=self.config['weight'],
                weighted_loss=float(weighted.detach()),corruption_A=self.config['corruption_A'],
                wall_seconds=time.monotonic()-began,exposure=dict(self.exposure),**metrics)

    def state_dict(self):
        return dict(schema='molvid.static_replay.v1',config=self.config,
            mixture=self.mixture.state_dict(),generator_rng=self.generator.get_state(),
            isolated_rng=self.rng,exposure=self.exposure)

    def load_state_dict(self, saved):
        if saved['schema']!='molvid.static_replay.v1' or saved['config']!=self.config:
            raise ValueError('Static replay recipe differs on exact resume')
        self.mixture.load_state_dict(saved['mixture'])
        self.generator.set_state(saved['generator_rng'])
        if len(saved['isolated_rng']['cuda'])!=torch.cuda.device_count():
            raise ValueError('Exact replay resume requires the same visible CUDA devices')
        self.rng=copy.deepcopy(saved['isolated_rng'])
        self.exposure=copy.deepcopy(saved['exposure'])

    def close(self):
        self.mixture.close()


def static_replay_from_settings(settings, seed, device):
    config=settings.get('static_replay') or {}
    if not config.get('enabled',False):
        return None  # Legacy path: no additional sampler, seeds or RNG work.
    if settings['scheme']!='flow' or any(not t.startswith('dynamic:')
                                        for t in settings['data']['sixteen_step_tasks']):
        raise ValueError('This replay is additional to an entirely dynamic RF recipe')
    return StaticReplay(config,seed,device)


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
    capacity_defaults = dict(obs_extra_layers=0, query_pre_extra_layers=0,
        temporal_extra_layers=0, collective_extra_layers=0,
        collective_extra_channels=512, query_post_extra_layers=0)
    changed += [key for key, default in capacity_defaults.items()
                if parent.get(key, default) != model.settings.get(key, default)]
    if parent.get('reference_concat',False)!=model.settings.get('reference_concat',False):
        changed.append('reference_concat')
    if changed:
        raise ValueError(f'Incompatible coordinate warm start: {changed}')
    if saved['step'] != parent['updates']:
        raise ValueError('Use a completed fixed-budget parent for a new training stage')
    model.load_state_dict(saved['model'], strict=True)
    return dict(kind='coordinate_state_weights_only', parent_checkpoint=str(Path(path).resolve()),
        parent_step=saved['step'], parent_seed=saved['seed'], parent_exposure=saved['exposure'],
        query_qk_transition=dict(parent=parent.get('query_qk_mode', 'atom'),
                                 current=model.settings.get('query_qk_mode', 'atom')), 
        capacity=dict(model.capacity),
        parent_source_commit=saved.get('source_commit'),
        parent_GPU_hours=saved['elapsed_seconds']/3600,
        parent_initialization=saved.get('initialization_report', {'kind': parent['initialization']}),
        inherited_parameters=sum(p.numel() for p in model.parameters()),
        inherited_modules=['spatial', 'temporal', 'tokens', 'progress', 'physical_time', 'displacement', 'head'],
        frozen_modules=[], newly_initialized_modules=[],
        reset=['optimizer', 'schedule', 'RNG', 'data_visits', 'stage_exposure', 'stage_GPU_hours'])


def train(settings, output, seed, resume=None, warm_start=None, stop_after=None):
    # Operational save/exit preserves the configured optimization schedule.
    if stop_after is not None and not 0 < stop_after <= settings['updates']:
        raise ValueError('stop_after must be within the unchanged configured updates')
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
    replay = static_replay_from_settings(settings, seed, device)
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
        if replay is not None:
            replay.load_state_dict(saved['static_replay'])
        elif 'static_replay' in saved:
            raise ValueError('Cannot discard replay state on an exact resume')
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
        replay_state=({} if replay is None else dict(static_replay=replay.state_dict(),
                                                   replay_exposure=dict(replay.exposure)))
        atomic_torch_save(path, dict(schema='molvid.coordinate_state.checkpoint.v1', model=model.state_dict(),
            optimizer=optimizer.state_dict(), settings=settings, seed=seed, step=step, contract=model.contract(),
            mixture=mixture.state_dict(), exposure=exposure, elapsed_seconds=elapsed,
            initialization_report=initialization_report, source_commit=source_commit,
            generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), **replay_state))
        atomic_write_json(output/'status.json', dict(status='completed' if final else 'training', step=step,
            checkpoint=str(path), elapsed_seconds=elapsed, GPU_hours=elapsed/3600,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), exposure=exposure,
            **({} if replay is None else dict(replay_exposure=dict(replay.exposure)))))

    try:
        with (output/'updates.jsonl').open('a', buffering=1) as log:
            for step in range(start, settings['updates']):
                batch, task, history, queries, identity=mixture.next()
                if task=='dynamic' and settings.get('history_sampling')=='alternating_optimizer_updates':
                    # Identical to the TIME continuation: the optimizer step,
                    # not a newly visited system, selects H4Q12 versus H8Q8.
                    history=(4,8)[step % 2]
                    queries=list(range(history,batch.frames))
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
                    observed=_static_observation(batch,sigma,generator)
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
                logged_loss=float(loss.detach())
                if replay is not None:
                    metrics.update(dynamic_RF_loss=logged_loss,replay_applied=replay.due(step+1))
                    if replay.due(step+1):
                        metrics['replay']=replay.backward(model,bf16=settings['bf16'])
                        logged_loss+=metrics['replay']['weighted_loss']
                # One clip and one optimizer update for the sum of dynamic RF
                # and (on every fourth step) separately normalized .25*Q1 RF.
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad'],error_if_nonfinite=True)
                # Report real first-update supervision of the static path;
                # no separate probe model, synthetic optimization or extra run.
                if step==start and task=='static':
                    metrics['static_shared_gradient_norms_after_clip']={
                        name:sum(float(p.grad.float().square().sum()) for p in module.parameters()
                                 if p.grad is not None)**.5
                        for name,module in [('current_geometry',model.spatial),
                                            ('shared_tokens',model.tokens),('coordinate_head',model.head)]}
                    metrics['static_capacity_gradient_norms_after_clip'] = {
                        name: sum(float(p.grad.float().square().sum())
                                  for p in module.parameters() if p.grad is not None)**.5
                        for name, module in [('obs_extra', model.obs_extra),
                            ('query_pre_extra', model.query_pre_extra),
                            ('collective_extra', model.collective_extra),
                            ('query_post_extra', model.query_post_extra)] if len(module)}
                    if any(value == 0 or not math.isfinite(value)
                           for value in metrics['static_capacity_gradient_norms_after_clip'].values()):
                        raise RuntimeError('Static supervision did not reach the active capacity path')
                    if any(value==0 or not math.isfinite(value) for value in metrics['static_shared_gradient_norms_after_clip'].values()):
                        raise RuntimeError('Static supervision did not reach every shared geometry path')
                optimizer.step()
                exposure['updates']+=1
                exposure['dynamic_examples' if task=='dynamic' else 'static_examples']+=batch.batch_size
                exposure['target_frames']+=len(target)*batch.batch_size
                exposure['atom_query_pairs']+=len(target)*int(batch.loss_mask.sum())
                row=dict(step=step+1,loss=logged_loss,grad_norm=float(norm),task=task,
                    sample=batch.sample_id[0],**identity,history=history,queries=queries,
                    elapsed_seconds=prior_seconds+time.monotonic()-began,**metrics)
                log.write(json.dumps(row)+'\n')
                if step<4 or (step+1)%64==0:
                    print(json.dumps(row),flush=True)
                if (step+1)%settings['save_every']==0:
                    save(step+1)
                if stop_after is not None and step+1 >= stop_after and step+1 < settings['updates']:
                    save(step+1)
                    atomic_write_json(output/'requested_stop.json', dict(
                        step=step+1, configured_updates=settings['updates'],
                        checkpoint=str(output/'checkpoint_latest.pt'),
                        reason='requested_save_and_exit_without_schedule_change'))
                    return
            save(settings['updates'],final=True)
    except Exception as error:
        atomic_write_json(output/'failure.json',dict(error=repr(error),exposure=exposure,
            elapsed_seconds=prior_seconds+time.monotonic()-began))
        raise
    finally:
        mixture.close()
        if replay is not None:
            replay.close()
