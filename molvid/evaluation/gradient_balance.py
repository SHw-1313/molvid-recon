"""Read-only task-gradient diagnostics; no optimizer or gradient surgery."""
import math

import torch


def gradient_snapshot(model):
    """Copy populated parameter gradients to CPU; absent means unused."""
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float().cpu().clone()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"nonfinite gradient: {name}")
            result[name] = gradient
    return result


def gradient_comparison(dynamic, static):
    """Full module norms plus alignment restricted to static-active parameters.

    A missing gradient is a zero in the complete module vector, not grounds
    for dropping a dynamic-only parameter from its reported norm.
    """
    result = {}
    for group, prefix in (("dit", "dit."), ("readout", "readout.")):
        d = {n: g for n, g in dynamic.items() if n.startswith(prefix)}
        s = {n: g for n, g in static.items() if n.startswith(prefix)}
        if not d or not s:
            raise ValueError(f"missing task gradients in {group}")
        norm = lambda values: math.sqrt(sum(float(g.double().square().sum()) for g in values))
        dn, sn = norm(d.values()), norm(s.values())
        active_dn = norm(g for n, g in d.items() if n in s)
        dot = sum(float((g.double() * s[n].double()).sum()) for n, g in d.items() if n in s)
        result[group] = dict(dynamic_norm=dn, static_norm=sn,
            cosine=None if dn * sn == 0 else dot / (dn * sn),
            static_active_dynamic_norm=active_dn,
            static_active_cosine=None if active_dn * sn == 0 else dot / (active_dn * sn),
            static_to_dynamic_norm=None if dn == 0 else sn / dn,
            static_parameter_tensors=len(s), dynamic_parameter_tensors=len(d))
    return result


def require_static_boundaries(gradients):
    """The static task must never enter history, motion or physical time."""
    forbidden = ("history_encoder.", "motion_encoder.", "target_teacher.", "dit.query_time.")
    for name in gradients:
        if name.startswith(forbidden) or any(token in name for token in (".temporal.", ".temporal_adaln.", ".history.", ".history_adaln.", ".history_time", ".temporal_time")):
            raise RuntimeError(f"static gradient entered temporal/frozen module: {name}")
