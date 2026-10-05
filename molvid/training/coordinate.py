"""Coordinate-RF training step and explicit fresh/resume state boundaries."""
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from ..codec.frame import slice_clip_frames
from ..flow.coordinate import coordinate_flow_loss
from ..latent.types import QuerySpec
from ..losses.stereochemistry import observed_ca_reference, handedness_margin_loss
from .coordinate_rollout import checkpointed_coordinate_rollout


@dataclass(frozen=True)
class CoordinateTrainConfig:
    dit_lr: float = 5e-5
    history_lr: float = 2.5e-5
    readout_lr: float = 5e-4
    weight_decay: float = .01
    clip_grad: float = 1.
    total_updates: int = 2048
    checkpoint_blocks: bool = False
    handedness_weight: float = 0.
    handedness_every: int = 8
    handedness_steps: int = 16
    handedness_margin: float = .1
    handedness_minimum_volume_A3: float = .2

    def __post_init__(self):
        if not math.isfinite(self.handedness_weight) or self.handedness_weight < 0:
            raise ValueError('handedness weight must be finite and nonnegative')
        if (type(self.handedness_every) is not int or self.handedness_every < 1
                or type(self.handedness_steps) is not int or self.handedness_steps < 1):
            raise ValueError('positive integer handedness cadence and steps required')
        if not math.isfinite(self.handedness_margin) or not 0 < self.handedness_margin <= 1:
            raise ValueError('handedness margin must lie in (0,1]')
        if not math.isfinite(self.handedness_minimum_volume_A3) or self.handedness_minimum_volume_A3 <= 0:
            raise ValueError('positive finite reference-volume threshold required')


def make_coordinate_optimizer(model, config):
    groups = [dict(params=[p for p in model.dit.parameters() if p.requires_grad], lr=config.dit_lr, name="dit"),
              dict(params=[p for p in model.history_encoder.parameters() if p.requires_grad], lr=config.history_lr, name="history"),
              dict(params=list(model.readout.parameters()), lr=config.readout_lr, name="readout")]
    if model.motion_encoder is not None:
        groups.append(dict(params=[p for p in model.motion_encoder.parameters() if p.requires_grad], lr=config.history_lr, name="motion"))
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def coordinate_update(model, optimizer, cpu_batch, *, history_frames, generator, step, config,
                      atom_identity=None):
    """One FP32 CFM update with optional observed-only sampled chirality.

    All data extraction is on CPU before transfer. Only the observed prefix
    reaches condition construction; future XYZ is used for RF interpolation
    and velocity supervision, not passed as a generation condition.
    """
    device = next(model.parameters()).device
    observed_cpu = slice_clip_frames(cpu_batch, 0, history_frames)
    model.prepare_batch(observed_cpu)
    observed = observed_cpu.to(device)
    target = cpu_batch.x[history_frames:].detach().to(device)
    query = QuerySpec(cpu_batch.time_ps[:, history_frames:].to(device),
                      cpu_batch.frame_mask[:, history_frames:].to(device))
    use_handedness = config.handedness_weight > 0 and (step+1) % config.handedness_every == 0
    reference = None
    if use_handedness:
        if atom_identity is None:
            raise ValueError('sampled handedness requires explicit packed atom identities')
        reference = observed_ca_reference(observed=observed.x, atom_identity=atom_identity,
            block_id=observed.block_id, abid=observed.abid, frame_mask=observed.frame_mask,
            loss_mask=observed.loss_mask, bond_index=observed.bond_index,
            minimum_volume_A3=config.handedness_minimum_volume_A3)
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
    prediction = model(noisy, condition=condition, query=query, flow_time=progress,
                       checkpoint_blocks=config.checkpoint_blocks)
    loss = coordinate_flow_loss(prediction, target - source, batch=observed, query=query)
    loss.backward()
    rf_loss = float(loss.detach())
    auxiliary_value = 0.
    # Release CFM activations before differentiable generation. Source depends
    # only on observations and the same main RNG in both paired arms. No future
    # coordinates enter the auxiliary trajectory or handedness reference.
    del prediction, noisy, condition, loss
    if use_handedness and reference.indices.numel():
        # Preserve main dropout/global RNG across an extra auxiliary trajectory.
        # The explicit flow generator is never consumed by the auxiliary pass.
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            condition = model.prepare_condition(observed)
            generated = checkpointed_coordinate_rollout(model, source, condition=condition,
                query=query, steps=config.handedness_steps)
            auxiliary = handedness_margin_loss(generated, reference, frame_mask=query.frame_mask,
                margin=config.handedness_margin)
            if not bool(torch.isfinite(auxiliary)):
                raise FloatingPointError('nonfinite sampled handedness; cursor not advanced')
            (config.handedness_weight*auxiliary).backward()
            auxiliary_value = float(auxiliary.detach())
            del condition, generated, auxiliary
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
    if not torch.isfinite(norm) or not math.isfinite(rf_loss):
        raise FloatingPointError("nonfinite coordinate RF update; cursor not advanced")
    if any(p.grad is not None for p in model.target_teacher.parameters()):
        raise RuntimeError("frozen teacher acquired parameter gradients")
    optimizer.step()
    return dict(loss=rf_loss+config.handedness_weight*auxiliary_value,
                rf_loss_A2=rf_loss, handedness_loss=auxiliary_value,
                handedness_applied=bool(use_handedness and reference.indices.numel()),
                handedness_centers=0 if reference is None else reference.indices.shape[0],
                handedness_weight=config.handedness_weight,
                grad_norm=float(norm), flow_time_mean=float(progress.mean()),
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
