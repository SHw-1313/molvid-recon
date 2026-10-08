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


def train(settings, output, seed, resume=None):
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
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['schema']!='molvid.coordinate_state.checkpoint.v1' or saved['settings']!=settings or saved['seed']!=seed:
            raise ValueError('Wrong objective, scales, seed or checkpoint schema')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        mixture.load_state_dict(saved['mixture']); generator.set_state(saved['generator_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        start=saved['step']; exposure=saved['exposure']; prior_seconds=saved['elapsed_seconds']
        del saved
    atomic_write_json(output/'recipe.json', dict(settings=settings, seed=seed, model=model.contract(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        CUDA_device=torch.cuda.get_device_name(0), resume=str(resume), initialization=settings['initialization']))
    began=time.monotonic()

    def save(step, final=False):
        elapsed=prior_seconds+time.monotonic()-began
        path=output/('coordinate_final.pt' if final else 'checkpoint_latest.pt')
        atomic_torch_save(path, dict(schema='molvid.coordinate_state.checkpoint.v1', model=model.state_dict(),
            optimizer=optimizer.state_dict(), settings=settings, seed=seed, step=step, contract=model.contract(),
            mixture=mixture.state_dict(), exposure=exposure, elapsed_seconds=elapsed,
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
