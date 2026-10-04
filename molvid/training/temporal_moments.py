"""Independent U-probe training on frozen, hash-verified real-MD features."""
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from ..temporal_moments import FutureMoments,gaussian_nll
from ..evaluation.predictive import HORIZONS,gaussian_metrics,variance_temperature,system_report,paired_report
from ..runtime import configure_device,atomic_write_json,sha256_file
from .predictive_data import load_cache,atomic_torch_save
from ..provenance import freeze_run_identity


@torch.no_grad()
def predict(model,payload,*,chunk=2048):
    x=payload['x'];c,a,_=x.shape
    outputs=[]
    for horizon in HORIZONS:
        means=[];variances=[]
        for values in x.flatten(0,1).split(chunk):
            m,v=model(values,values.new_full((len(values),),horizon))
            means.append(m);variances.append(v)
        outputs.append((torch.cat(means).reshape(c,a,-1),torch.cat(variances).reshape(c,a,-1)))
    return torch.stack([o[0] for o in outputs],2),torch.stack([o[1] for o in outputs],2)


def train_u(cache,output,*,steps=2000,seeds=(101,102,103),arms=None,feature_modes=None,comparisons=None):
    output=Path(output)
    if output.exists() or steps<1: raise ValueError('fresh U output and positive steps required')
    device=configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.20)
    data,stats,manifest=load_cache(cache,device)
    output.mkdir(parents=True)
    arms=arms or {'U_last':(False,True),'U_history_mu':(True,False),'U_history_muvar':(True,True)}
    feature_modes=feature_modes or {}
    comparisons=comparisons or [('U_last','U_history_muvar'),('U_history_mu','U_history_muvar')]
    protocol=dict(schema='molvid.u.probe.v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cache=str(cache),cache_manifest_sha256=sha256_file(Path(cache)/'complete.json'),steps=steps,seeds=list(seeds),
        arms=arms,feature_modes=feature_modes,lr=3e-4,batch_size=512,weight_decay=1e-4,grad_clip=1.,
        horizons_ps=list(HORIZONS),generator_updates=0,teacher_updates=0,history_updates=0,
        variance='diagonal Gaussian, log variance in [-6,4]; not full covariance or multimodality',
        calibration='one temperature per horizon on 24 head-held-out training systems, clipped [0.05,20]',
        test_opened=False,pid=os.getpid())
    atomic_write_json(output/'protocol.json',protocol)
    freeze_run_identity(output,inputs=[Path(cache)/'complete.json',Path(cache)/'protocol.json'],purpose='U/head controls',protocol=protocol)
    x=data['fit']['x'];y=data['fit']['y'];last_width=data['fit']['last_width']
    horizons=x.new_tensor(HORIZONS);reports={};paired={};started=time.monotonic()
    fit_mean=y.mean((0,1));fit_var=y.var((0,1),unbiased=False).clamp_min(1e-4)
    valid=data['valid'];truth=valid['y'];metadata=valid['metadata']
    baselines={}
    for name,mu,var in [('persistence',torch.zeros_like(truth),y.square().mean((0,1)).clamp_min(1e-4).expand_as(truth)),
                        ('lag_only',fit_mean.expand_as(truth),fit_var.expand_as(truth))]:
        baselines[name]={k:system_report(v,metadata) for k,v in gaussian_metrics(mu,var,truth).items()}
    atomic_write_json(output/'baselines.json',baselines)
    try:
        for seed in seeds:
            metrics_by_arm={}
            for arm,(history,hetero) in arms.items():
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                mode=feature_modes.get(arm,'all')
                model=FutureMoments(x.shape[-1],y.shape[-1],last_width,history=history,heteroscedastic=hetero,feature_mode=mode).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                rng=torch.Generator(device=device).manual_seed(seed+10000)
                rows=[];tick=time.monotonic();model.train()
                for step in range(steps):
                    ci=torch.randint(len(x),(512,),device=device,generator=rng)
                    ai=torch.randint(x.shape[1],(512,),device=device,generator=rng)
                    hi=torch.randint(3,(512,),device=device,generator=rng)
                    mu,var=model(x[ci,ai],horizons[hi]);loss=gaussian_nll(mu,var,y[ci,ai,hi])
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite U loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    if (step+1)%100==0 or step+1==steps:
                        row=dict(arm=arm,seed=seed,step=step+1,loss=float(loss),grad_norm=float(norm),seconds=time.monotonic()-tick)
                        rows.append(row);atomic_write_json(output/f'{arm}_{seed}_steps.json',rows);print(json.dumps(row),flush=True)
                model.eval();cal=data['calibration']
                cm,cv=predict(model,cal);temperature=variance_temperature(cm,cv,cal['y'])
                mu,var=predict(model,valid)
                raw=gaussian_metrics(mu,var,truth)
                calibrated=gaussian_metrics(mu,var*temperature[None,None,:,None],truth)
                metrics_by_arm[arm]=calibrated
                key=f'{arm}_{seed}'
                report=dict(temperature=temperature.cpu().tolist(),
                    raw={k:system_report(v,metadata) for k,v in raw.items()},
                    calibrated={k:system_report(v,metadata) for k,v in calibrated.items()})
                reports[key]=report;atomic_write_json(output/f'{key}_metrics.json',report)
                path=output/f'{key}.pt'
                atomic_torch_save(path,dict(schema='molvid.u.probe.checkpoint.v1',model=model.state_dict(),
                    optimizer=optimizer.state_dict(),rng=rng.get_state().cpu(),step=steps,seed=seed,arm=arm,
                    model_config=dict(input_width=x.shape[-1],target_width=y.shape[-1],last_width=last_width,
                                      history=history,heteroscedastic=hetero,feature_mode=mode),
                    temperature=temperature,statistics=stats,protocol=protocol))
                # Model-only reload parity; no generator weight is copied.
                payload=torch.load(path,map_location=device,weights_only=False)
                check=FutureMoments(**payload['model_config']).to(device).eval();check.load_state_dict(payload['model'],strict=True)
                with torch.no_grad():
                    a=model(x[0,:4],horizons[0].expand(4));b=check(x[0,:4],horizons[0].expand(4))
                    if not all(torch.equal(u,v) for u,v in zip(a,b)): raise RuntimeError('U reload mismatch')
                del check,payload,model,optimizer
            paired[str(seed)]={f'{b}_minus_{a}':{metric:paired_report(metrics_by_arm[b][metric],metrics_by_arm[a][metric],metadata)
                for metric in ('mse','nll','coverage90','interval_width90')}
                for a,b in comparisons}
            atomic_write_json(output/'paired_comparisons.json',paired)
        atomic_write_json(output/'summary.json',dict(status='completed',steps_per_arm=steps,seeds=list(seeds),
            arms=list(arms),seconds=time.monotonic()-started,checkpoints=len(arms)*len(seeds),test_opened=False,
            promotion='none; diagnostic only. Individual-seed system CIs do not estimate between-seed uncertainty.'))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
