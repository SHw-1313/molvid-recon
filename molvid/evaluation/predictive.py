"""System-equal, horizon-separated reporting for frozen-feature probes."""
from collections import defaultdict
from statistics import mean
import math
import numpy as np
import torch


HORIZONS=(100,400,1200)


def system_report(values,metadata):
    """values [clips, atoms, horizons]; atoms/windows are not independent CIs."""
    if values.ndim!=3 or len(metadata)!=len(values) or values.shape[-1]!=3:
        raise ValueError('probe metric axes must be clips, atoms, three horizons')
    per_clip=values.detach().float().mean(1).cpu().numpy()
    groups=defaultdict(list)
    for row,value in zip(metadata,per_clip): groups[row['system']].append(value)
    per_system={key:np.mean(group,axis=0) for key,group in groups.items()}
    result={}
    for lag_index,lag in enumerate(HORIZONS):
        data=np.array([v[lag_index] for _,v in sorted(per_system.items())])
        rng=np.random.default_rng(1704)
        boot=data[rng.integers(0,len(data),size=(2000,len(data)))].mean(1)
        result[str(lag)]=dict(mean=float(data.mean()),ci95_system_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
            systems=len(data),per_system={k:float(v[lag_index]) for k,v in per_system.items()})
    return result


def gaussian_metrics(mu,var,target):
    if mu.shape!=target.shape or var.shape!=target.shape or not bool((var>0).all()):
        raise ValueError('Gaussian prediction dimensions/variance invalid')
    error=target-mu;sd=var.sqrt()
    return dict(mse=error.square().mean(-1),
        nll=(.5*(math.log(2*math.pi)+var.log()+error.square()/var)).mean(-1),
        coverage50=(error.abs()<=.67448975*sd).float().mean(-1),
        coverage90=(error.abs()<=1.64485363*sd).float().mean(-1),
        coverage95=(error.abs()<=1.95996398*sd).float().mean(-1),
        interval_width90=(2*1.64485363*sd).mean(-1))


def variance_temperature(mu,var,target):
    """Scalar per horizon, fit only on head-calibration systems; floor/cap explicit."""
    return ((target-mu).square()/var).mean((0,1,3)).clamp(.05,20.)


def kernel_calibration(prediction,target,bins=10):
    """Calibration of a bounded conditional kernel expectation, not event probability."""
    result=[]
    for lag,ps in enumerate(HORIZONS):
        p=prediction[:,:,lag].flatten();y=target[:,:,lag].flatten();rows=[]
        for i in range(bins):
            mask=(p>=i/bins)&(p<((i+1)/bins) if i<bins-1 else p<=1.)
            if bool(mask.any()):
                rows.append(dict(bin=i,count=int(mask.sum()),prediction=float(p[mask].mean()),target=float(y[mask].mean())))
        result.append(dict(horizon_ps=ps,bins=rows))
    return result


def paired_report(second,first,metadata):
    return system_report(second-first,metadata)
