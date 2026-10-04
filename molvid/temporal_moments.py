"""Small observed-feature probes; not a replacement history encoder or flow."""
import torch
from torch import nn
from .observed_features import select_observed_features


def time_features(horizon_ps):
    if not bool(torch.isfinite(horizon_ps).all()) or bool((horizon_ps<=0).any()):
        raise ValueError('finite positive physical horizons in ps required')
    t=horizon_ps.float()/100.
    return torch.stack((torch.log1p(t),t/12.,torch.sin(t/12.),torch.cos(t/12.)),dim=-1)


class FutureMoments(nn.Module):
    """Diagonal Gaussian in fixed projected scalar-change space [N, channels].

    Last-only and history arms use identical parameter shapes. Removed history
    channels are zeroed AFTER fit-only normalization, never filled by targets.
    """
    def __init__(self,input_width,target_width,last_width,*,history=True,heteroscedastic=True,feature_mode='all'):
        super().__init__()
        self.last_width=last_width;self.history=history;self.heteroscedastic=heteroscedastic
        self.target_width=target_width
        self.feature_mode=feature_mode
        self.network=nn.Sequential(nn.Linear(input_width+4,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2*target_width))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,observed_features,horizon_ps):
        x=select_observed_features(observed_features,self.last_width,self.feature_mode if self.history else 'last')
        output=self.network(torch.cat((x,time_features(horizon_ps)),dim=-1))
        mu,log_variance=output.chunk(2,dim=-1)
        log_variance=log_variance.clamp(-6.,4.) if self.heteroscedastic else torch.zeros_like(log_variance)
        return mu,log_variance.exp()


def gaussian_nll(mu,variance,target):
    return .5*(variance.log()+(target-mu).square()/variance).mean()
