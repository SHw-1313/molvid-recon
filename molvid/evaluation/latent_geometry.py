"""Train-only latent spectra and decoder sensitivity, not a generation benchmark."""
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from ..data.batch import collate_clip_records
from ..flow.sampling import euler_sample_frames
from ..flow.source import repeat_last_frame_center, sample_frame_source
from ..geometry.ca_detail import condition_from_record
from ..losses.sampled_physical import PhysicalAuxiliaryConfig
from ..provenance import freeze_run_identity
from ..runtime import atomic_write_json, configure_device, sha256_file
from ..training.batches import prepare_frame_joint_batch
from ..training.ca_detail_data import load_ca_cache
from ..training.execution_evidence import ExecutionEvidence
from ..training.predictive_data import load_parent
from .ca_detail import score


def channel_spectrum(values, *, vector):
    """Channel covariance: xyz pooled for SO(3)-invariant vector channel Gram.

    Scalars are [frames,atoms,C]; vectors [frames,atoms,3,C]. This is a
    128-channel spectrum, not the intrinsic dimension of the full molecule.
    """
    x = torch.cat([v.reshape(-1, v.shape[-1]) for v in values]).double()
    if not vector:
        x = x - x.mean(0)
    gram = x.T @ x / len(x)
    eigenvalues, basis = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    basis = basis.flip(1).float()
    fraction = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(fraction * fraction.clamp_min(1e-30).log()).sum()
    # Only the 128-value reporting reduction is CPU; model/eigensolve stay CUDA.
    cumulative = fraction.cpu().cumsum(0)
    return dict(eigenvalues=eigenvalues.tolist(), energy_fraction=fraction.tolist(),
                effective_rank=float(entropy.exp()),
                rank90=int(torch.searchsorted(cumulative, .9).item() + 1),
                rank99=int(torch.searchsorted(cumulative, .99).item() + 1),
                rows=len(x), centered=not vector), basis


def norm_stats(value):
    norm = value.float().flatten(2).norm(dim=-1)
    return dict(mean=float(norm.mean()), rms=float(norm.square().mean().sqrt()),
                median=float(norm.median()), coefficient_rms=float(value.float().square().mean().sqrt()))


def matched_norm(value, magnitude):
    """Fix total feature L2 per atom-frame, preserving vector xyz covariance."""
    norm = value.float().flatten(2).norm(dim=-1).clamp_min(1e-12)
    return value * (magnitude / norm).reshape(*value.shape[:2], *([1] * (value.ndim - 2)))


@torch.no_grad()
def diagnose(cache, output, systems=8):
    output, cache = Path(output), Path(cache)
    if output.exists():
        raise FileExistsError(output)
    if systems != 8:
        raise ValueError('first registered diagnostic fixes eight training systems')
    items = load_ca_cache(cache, 'train')
    groups = defaultdict(list)
    for item in items:
        groups[item['metadata']['system']].append(item)
    items = [sorted(groups[s], key=lambda x:x['record']['sample_id'])[0] for s in sorted(groups)[:systems]]
    device = configure_device('cuda', deterministic=True)
    model, raw, release = load_parent(device)
    recipe = Path('/workspace/molvid-recon-gm-motion-v3') / release['auxiliary_recipe']['path']
    if sha256_file(recipe) != release['auxiliary_recipe']['sha256']:
        raise ValueError('physical recipe identity changed')
    physical = PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    protocol = dict(systems=[i['metadata']['system'] for i in items],
                    samples=[i['record']['sample_id'] for i in items],
                    split='train', future_informed_diagnostic=True, solver=16, seed=83,
                    perturbation_norms=[.1, 1.], test_opened=False,
                    purpose='representation diagnosis, not independent model selection')
    freeze_run_identity(output, inputs=[cache/'complete.json', release['selected_checkpoint']['path'],recipe],
                        purpose='Current A1 train-only latent geometry diagnostic', protocol=protocol)
    atomic_write_json(output/'protocol.json', protocol)
    with ExecutionEvidence(output):
        started = time.monotonic()
        moments = defaultdict(list)
        for item in items:
            cpu = collate_clip_records([item['record']])
            batch = prepare_frame_joint_batch(model.target_teacher,cpu,device=device,normalizer=model,history_frames=4)
            clean = model.normalize(batch.target_future)
            last = model.normalize(batch.observed_context.latent)
            # Bound storage for channel statistics; deterministic atom/frame sampling.
            for field in ('h','v'):
                value = getattr(clean,field)
                delta = value - getattr(last,field)[-1:]
                for kind, tensor in (('clean',value),('motion',delta)):
                    moments[f'{kind}_{field}'].append(tensor[:,::max(1,value.shape[1]//256)].cpu())
        spectra, bases = {}, {}
        for name, values in moments.items():
            spectra[name], bases[name] = channel_spectrum([v.to(device) for v in values], vector=name.endswith('_v'))
        atomic_write_json(output/'spectra.json', spectra)
        rows = []
        for item in items:
            if time.monotonic()-started > 7200:
                raise TimeoutError('two-hour diagnosis budget')
            record = item['record']
            c = condition_from_record(record).to(device)
            batch = prepare_frame_joint_batch(model.target_teacher,collate_clip_records([record]),
                                              device=device,normalizer=model,history_frames=4)
            context, query = batch.observed_context, batch.query
            target = model.normalize(batch.target_future)
            generated, _ = euler_sample_frames(model,context=context,query=query,steps=16,seed=83)
            generated_z = model.normalize(generated)
            clean_x = model.decoder(context,batch.target_future,query).coordinates.float()
            predicted_x = model.decoder(context,generated,query).coordinates.float()
            source_center = repeat_last_frame_center(model.normalize(context.latent), query)
            _, noise = sample_frame_source(source_center,generator=torch.Generator(device=device).manual_seed(83))
            row = dict(system=item['metadata']['system'],sample_id=record['sample_id'],
                       generated=score(predicted_x,record,c,physical),oracle=score(clean_x,record,c,physical),
                       norms={},error_energy_outside_motion_channels={},sensitivity=[])
            for field in ('h','v'):
                truth = getattr(target,field)
                err = getattr(generated_z,field)-truth
                movement = truth-getattr(source_center,field)
                row['norms'][field] = {k:norm_stats(v) for k,v in [('motion',movement),('source_noise',getattr(noise,field)),('generated_error',err)]}
                flat = err.reshape(-1,err.shape[-1]); basis = bases['motion_'+field]
                projected = flat @ basis
                energy = projected.square().sum().clamp_min(1e-20)
                row['error_energy_outside_motion_channels'][field] = {str(rank):float(projected[:,rank:].square().sum()/energy) for rank in (8,32,64)}
                rng = torch.Generator(device=device).manual_seed(917)
                random = torch.randn(err.shape,device=device,generator=rng)
                for direction, delta in [('actual_generated_error',err),('isotropic',random)]:
                    for magnitude in (.1,1.):
                        adjusted = truth + matched_norm(delta,magnitude)
                        perturbed = target.with_features(adjusted if field=='h' else target.h,
                                                       adjusted if field=='v' else target.v)
                        decoded = model.decoder(context,model.inverse(perturbed),query).coordinates.float()
                        row['sensitivity'].append(dict(field=field,direction=direction,norm=magnitude,
                            coordinate_change_rmse_A=float((decoded-clean_x).square().sum(-1).mean().sqrt())))
            rows.append(row)
            atomic_write_json(output/'rows.json',rows)
            print(json.dumps(dict(phase='latent_diagnostic',completed=len(rows),total=len(items),seconds=time.monotonic()-started)),flush=True)
        atomic_write_json(output/'summary.json',dict(status='completed',systems=len(items),draws=len(items),
            seconds=time.monotonic()-started,spectra=spectra,rows=rows,test_opened=False,
            limitations=['train-only','channel spectra are not full molecular intrinsic rank',
                         'actual-error perturbations use targets; not deployable forecasts',
                         'random generated vs single true path error is not a distribution score']))
