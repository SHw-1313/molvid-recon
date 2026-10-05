"""Coordinate-RF training step and explicit fresh/resume state boundaries."""
from dataclasses import dataclass
import random
import numpy as np
import torch
from ..codec.frame import slice_clip_frames
from ..flow.coordinate import coordinate_flow_loss
from ..latent.types import QuerySpec


@dataclass(frozen=True)
class CoordinateTrainConfig:
    dit_lr: float = 5e-5
    history_lr: float = 2.5e-5
    readout_lr: float = 5e-4
    weight_decay: float = .01
    clip_grad: float = 1.
    total_updates: int = 2048


def make_coordinate_optimizer(model, config):
    groups = [dict(params=[p for p in model.dit.parameters() if p.requires_grad], lr=config.dit_lr, name="dit"),
              dict(params=[p for p in model.history_encoder.parameters() if p.requires_grad], lr=config.history_lr, name="history"),
              dict(params=list(model.readout.parameters()), lr=config.readout_lr, name="readout")]
    if model.motion_encoder is not None:
        groups.append(dict(params=[p for p in model.motion_encoder.parameters() if p.requires_grad], lr=config.history_lr, name="motion"))
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def coordinate_update(model, optimizer, cpu_batch, *, history_frames, generator, step, config):
    """One FP32 system-equal CFM update; no posthoc repair or geometry loss.

    All data extraction is on CPU before transfer. Only the observed prefix
    reaches condition construction; future XYZ is used for RF interpolation
    and velocity supervision, not passed as a generation condition.
    """
    device = next(model.parameters()).device
    observed_cpu = slice_clip_frames(cpu_batch, 0, history_frames)
    model.prepare_batch(observed_cpu)
    observed = observed_cpu.to(device)
    target = cpu_batch.x[history_frames:].to(device)
    query = QuerySpec(cpu_batch.time_ps[:, history_frames:].to(device),
                      cpu_batch.frame_mask[:, history_frames:].to(device))
    warmup = max(1, int(np.ceil(.05 * config.total_updates)))
    factor = ((step + 1) / warmup if step < warmup else
              .1 + .45 * (1 + np.cos(np.pi * min((step - warmup) / max(1, config.total_updates - warmup - 1), 1))))
    rates = dict(dit=config.dit_lr, history=config.history_lr, readout=config.readout_lr, motion=config.history_lr)
    for group in optimizer.param_groups:
        group["lr"] = rates[group["name"]] * factor
    optimizer.zero_grad(set_to_none=True)
    condition = model.prepare_condition(observed)
    source = model.source(condition=condition, query=query, generator=generator)
    progress = torch.rand(observed.batch_size, generator=generator, device=device)
    atom_progress = progress.index_select(0, observed.abid)[None, :, None]
    noisy = torch.lerp(source, target, atom_progress)
    prediction = model(noisy, condition=condition, query=query, flow_time=progress)
    loss = coordinate_flow_loss(prediction, target - source, batch=observed, query=query)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
    if not torch.isfinite(norm) or not torch.isfinite(loss):
        raise FloatingPointError("nonfinite coordinate RF update; cursor not advanced")
    if any(p.grad is not None for p in model.target_teacher.parameters()):
        raise RuntimeError("frozen teacher acquired parameter gradients")
    optimizer.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm), flow_time_mean=float(progress.mean()),
                learning_rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                valid_atom_queries=int((query.frame_mask.index_select(0, observed.abid).T & observed.loss_mask.bool()[None]).sum()),
                history_frames=history_frames, samples=list(cpu_batch.sample_id))


def capture_coordinate_rng(generator):
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                flow=generator.get_state(), python=random.getstate(), numpy=np.random.get_state())


def restore_coordinate_rng(state, generator):
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    generator.set_state(state["flow"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
