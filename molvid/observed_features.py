"""Explicit ablations of the immutable [last, endpoint delta, memory] cache."""
import torch


def select_observed_features(x, last_width, mode):
    keep = {'last': last_width, 'delta': 2*last_width, 'all': x.shape[-1]}
    if mode not in keep or keep[mode] > x.shape[-1]:
        raise ValueError('invalid observed-feature control')
    width = keep[mode]
    return x if width == x.shape[-1] else torch.cat((x[...,:width],torch.zeros_like(x[...,width:])),dim=-1)
