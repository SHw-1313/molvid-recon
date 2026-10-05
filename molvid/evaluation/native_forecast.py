"""Observed-only inference adapters; no change to external PVB algorithms."""
from pathlib import Path
import sys
import torch

from ..codec.frame import slice_clip_frames
from ..latent.types import ObservedContext, QuerySpec
from ..training.batches import prepare_batch_then_to_device
from ..flow.sampling import euler_sample_frames


@torch.no_grad()
def molvid_forecast(model, observed_cpu, query_times_ps, *, device, seed):
    """H observed frames only, query clock [1,Q]; return [Q,N,3] Angstrom."""
    observed = prepare_batch_then_to_device(model.target_teacher, observed_cpu, device)
    latent, origin = model.target_teacher(observed)
    context = ObservedContext(latent=latent, coordinates=observed.x,
        sample_origin=origin, loss_mask=observed.loss_mask.bool())
    times = query_times_ps.to(device)
    query = QuerySpec(time_ps=times, frame_mask=torch.ones_like(times, dtype=torch.bool))
    generated, _ = euler_sample_frames(model, context=context, query=query, steps=16, seed=seed)
    return model.decoder(context, generated, query).coordinates.float()


def load_pvb(checkpoint, root, device):
    # Pickled official model needs its original import namespace, not modified PVB.
    sys.path.insert(0, str(root))
    import module.model as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('PVB namespace resolved outside original repository')
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return model.to(device).eval().requires_grad_(False)


def pvb_observation(pdb, observed_cpu, atom_source_index, device):
    """Use native topology builder; replace PDB coordinates by exact observed MD.

    Atom order, types and covalent graph must agree before any prediction.
    PVB consumes only the last observed structure (its native H1 contract).
    """
    from data.data_init import make_batch
    batch, (indices,) = make_batch(str(pdb), 1)
    if not torch.equal(torch.as_tensor(indices).long(), torch.as_tensor(atom_source_index).long()):
        raise ValueError('PVB/Molvid atom mapping differs')
    for key in ('atype', 'btype'):
        if not torch.equal(batch[key], getattr(observed_cpu, key)):
            raise ValueError(f'PVB/Molvid {key} differs')
    def edges(value):
        return {tuple(sorted(pair)) for pair in value.T.tolist()}
    if edges(batch['bond_index']) != edges(observed_cpu.bond_index):
        raise ValueError('PVB/Molvid covalent graph differs')
    last = observed_cpu.x[-1].float()
    origin = last.mean(0)
    batch['x0'] = last-origin
    batch['b0'] = observed_cpu.bpos[-1].float()-origin
    return {k:v.to(device) for k,v in batch.items()}, origin.to(device)


@torch.no_grad()
def pvb_forecast(model, batch, origin, *, seed, frames=12):
    from utils.random_seed import setup_seed
    setup_seed(seed)
    batch = dict(batch)
    frames_out = []
    for _ in range(frames):
        x = model.inference(batch, sde_step=10)
        frames_out.append(x+origin)
        batch['x0'] = x
    return torch.stack(frames_out).float()
