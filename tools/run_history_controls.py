"""N3 fixed-cache controls; no generator updates or K1 regeneration."""
import argparse
from pathlib import Path
import torch
from molvid.training.temporal_moments import train_u
from molvid.training.finite_time import train_k
from molvid.training.predictive_data import load_cache, atomic_torch_save
from molvid.evaluation.predictive import gaussian_metrics, variance_temperature, system_report
from molvid.runtime import configure_device, atomic_write_json
from molvid.provenance import freeze_run_identity


@torch.no_grad()
def ridge(cache,output):
    if output.exists(): raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True);data,stats,_=load_cache(cache,device)
    freeze_run_identity(output,inputs=[cache/'complete.json',cache/'protocol.json'],purpose='N3 fixed ridge',
                        protocol={'alpha':.001,'tuning':'none','per_horizon':True,'generator_updates':0})
    reports={}
    for name,width in [('last',128),('delta',256),('history',512)]:
        def design(split):
            x=data[split]['x'][...,:width].flatten(0,1).double()
            return torch.cat((x,torch.ones_like(x[:,:1])),1)
        x=design('fit');y=data['fit']['y'].flatten(0,1).double()
        reg=torch.eye(x.shape[-1],device=device,dtype=x.dtype)*.001;reg[-1,-1]=0
        weights=torch.linalg.solve(x.T@x/len(x)+reg,x.T@y.flatten(1)/len(x))
        fit_mu=(x@weights).reshape_as(y)
        variance=(y-fit_mu).square().mean(0).clamp_min(1e-4).float()
        predictions={s:(design(s)@weights).reshape_as(data[s]['y']).float() for s in ('calibration','valid')}
        temperature=variance_temperature(predictions['calibration'],variance.expand_as(data['calibration']['y']),data['calibration']['y'])
        metrics=gaussian_metrics(predictions['valid'],(variance*temperature[:,None]).expand_as(data['valid']['y']),data['valid']['y'])
        reports[name]={k:system_report(v,data['valid']['metadata']) for k,v in metrics.items()}
        atomic_torch_save(output/f'{name}.pt',dict(weights=weights.cpu(),variance=variance.cpu(),temperature=temperature.cpu(),statistics=stats))
    atomic_write_json(output/'summary.json',dict(status='completed',reports=reports,test_opened=False,
        limitation='Separate horizon fits; ridge and MLP are different capacity classes, not parameter-matched.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=['u','k','ridge'],required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);a=p.parse_args()
    if a.lane=='u':
        arms={f'{mode}_{kind}':(True,kind=='muvar') for mode in ('last','delta','history') for kind in ('mu','muvar')}
        modes={name:('all' if name.startswith('history') else name.split('_')[0]) for name in arms}
        # Expand the paired list explicitly over both proper-score heads.
        pairs=[(f'{first}_{kind}',f'{second}_{kind}') for kind in ('mu','muvar') for first,second in [('last','delta'),('delta','history')]]
        train_u(a.cache,a.output,steps=a.steps,seeds=tuple(a.seeds),arms=arms,feature_modes=modes,comparisons=pairs)
    elif a.lane=='k': train_k(a.cache,a.output,steps=a.steps,seeds=tuple(a.seeds),generation=False,include_delta=True)
    else: ridge(a.cache,a.output)
