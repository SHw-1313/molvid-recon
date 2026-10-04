"""K0 predictors and K1 diagnostics, without a K2 generator loss."""
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from ..finite_time import FiniteTimeResponse,fit_response_regions,response_kernels
from ..evaluation.predictive import HORIZONS,system_report,paired_report,kernel_calibration
from ..runtime import configure_device,atomic_write_json,sha256_file
from .predictive_data import load_cache,atomic_torch_save


@torch.no_grad()
def predict(model,payload,*,chunk=2048):
    x=payload['x'];c,a,_=x.shape;outputs=[]
    for horizon in HORIZONS:
        values=[model(v,v.new_full((len(v),),horizon)) for v in x.flatten(0,1).split(chunk)]
        outputs.append(torch.cat(values).reshape(c,a,-1))
    return torch.stack(outputs,2)


def train_k(cache,output,*,steps=2000,seeds=(101,102,103),generation=True):
    output=Path(output)
    if output.exists() or steps<1: raise ValueError('fresh K output and positive steps required')
    device=configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.25)
    data,stats,manifest=load_cache(cache,device)
    output.mkdir(parents=True)
    regions=fit_response_regions(data['fit']['y'])
    atomic_torch_save(output/'regions.pt',regions)
    targets={split:response_kernels(p['y'],regions) for split,p in data.items()}
    protocol=dict(schema='molvid.k.probe.v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cache=str(cache),cache_manifest_sha256=sha256_file(Path(cache)/'complete.json'),regions_sha256=sha256_file(output/'regions.pt'),
        steps=steps,seeds=list(seeds),arms=['K_last','K_history'],lr=3e-4,batch_size=512,weight_decay=1e-4,
        horizons_ps=list(HORIZONS),generator_updates=0,K2_enabled=False,test_opened=False,pid=os.getpid(),
        target='conditional expectation of 16 Gaussian response kernels; not categorical probabilities',
        loss='squared error on bounded real-MD kernel targets',generation_diagnostic=generation)
    atomic_write_json(output/'protocol.json',protocol)
    x=data['fit']['x'];target=targets['fit'];last_width=data['fit']['last_width']
    horizons=x.new_tensor(HORIZONS);valid=data['valid'];truth=targets['valid'];metadata=valid['metadata']
    marginal=target.mean((0,1));persistent=response_kernels(torch.zeros_like(valid['y']),regions)
    baseline_values={'lag_only':(truth-marginal).square().mean(-1),'persistence':(truth-persistent).square().mean(-1)}
    atomic_write_json(output/'baselines.json',{k:system_report(v,metadata) for k,v in baseline_values.items()})
    comparisons={};started=time.monotonic()
    try:
        for seed in seeds:
            errors={}
            for arm,history in [('K_last',False),('K_history',True)]:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                config=dict(input_width=x.shape[-1],regions=truth.shape[-1],last_width=last_width,history=history)
                model=FiniteTimeResponse(**config).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                rng=torch.Generator(device=device).manual_seed(seed+10000);rows=[];tick=time.monotonic()
                for step in range(steps):
                    ci=torch.randint(len(x),(512,),device=device,generator=rng)
                    ai=torch.randint(x.shape[1],(512,),device=device,generator=rng)
                    hi=torch.randint(3,(512,),device=device,generator=rng)
                    prediction=model(x[ci,ai],horizons[hi]);loss=(prediction-target[ci,ai,hi]).square().mean()
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite K loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    if (step+1)%100==0 or step+1==steps:
                        row=dict(arm=arm,seed=seed,step=step+1,loss=float(loss),grad_norm=float(norm),seconds=time.monotonic()-tick)
                        rows.append(row);atomic_write_json(output/f'{arm}_{seed}_steps.json',rows);print(json.dumps(row),flush=True)
                model.eval();pred=predict(model,valid);error=(pred-truth).square().mean(-1);errors[arm]=error
                cal=predict(model,data['calibration'])
                report=dict(valid_mse=system_report(error,metadata),
                    calibration_mse=system_report((cal-targets['calibration']).square().mean(-1),data['calibration']['metadata']),
                    calibration_bins=kernel_calibration(cal,targets['calibration']),valid_bins=kernel_calibration(pred,truth),
                    note='Bins are descriptive pooled counts, not independent sample CIs; no calibration fitting on valid.')
                atomic_write_json(output/f'{arm}_{seed}_metrics.json',report)
                path=output/f'{arm}_{seed}.pt'
                atomic_torch_save(path,dict(schema='molvid.k.probe.checkpoint.v1',model=model.state_dict(),model_config=config,
                    optimizer=optimizer.state_dict(),rng=rng.get_state().cpu(),step=steps,seed=seed,arm=arm,
                    regions=regions,statistics=stats,protocol=protocol))
                saved=torch.load(path,map_location=device,weights_only=False)
                check=FiniteTimeResponse(**saved['model_config']).to(device).eval();check.load_state_dict(saved['model'],strict=True)
                with torch.no_grad():
                    if not torch.equal(check(x[0,:4],horizons[0].expand(4)),model(x[0,:4],horizons[0].expand(4))):
                        raise RuntimeError('K strict reload mismatch')
                del check,saved,model,optimizer
            comparisons[str(seed)]={
                'history_minus_last':paired_report(errors['K_history'],errors['K_last'],metadata),
                'history_minus_lag_only':paired_report(errors['K_history'],baseline_values['lag_only'],metadata),
                'history_minus_persistence':paired_report(errors['K_history'],baseline_values['persistence'],metadata)}
            atomic_write_json(output/'paired_comparisons.json',comparisons)
        atomic_write_json(output/'training_summary.json',dict(status='completed',steps_per_arm=steps,seeds=list(seeds),
            seconds=time.monotonic()-started,checkpoints=2*len(seeds),test_opened=False))
        if generation:
            from ..evaluation.finite_time import diagnose_generator
            # Fixed first declared seed, NOT the validation-best head.
            diagnose_generator(Path(cache),output/f'K_history_{seeds[0]}.pt',output/'generation_diagnostic')
        atomic_write_json(output/'summary.json',dict(status='completed',test_opened=False,K2_enabled=False,
            generation_diagnostic=generation,promotion='none; probe results and generator diagnostics are not training gains'))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
