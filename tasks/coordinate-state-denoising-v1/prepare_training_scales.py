"""Freeze a paired TRAIN subset and coordinate scales; no model/validation calls."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from molvid.data.bridge_mixture import BridgeMixture
from molvid.runtime import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = json.loads(args.base_config.read_text())['data']
    config.update(query_frames='all', sixteen_step_tasks=['dynamic:protein']*8+['dynamic:complex']*8)
    inventory = BridgeMixture(config, 71)
    selected = {}
    for domain in ('protein', 'complex'):
        keys = sorted(inventory.groups[domain])
        random.Random(20261008).shuffle(keys)
        selected[domain] = sorted(keys[:128])
    inventory.close()
    config['selected_systems'] = selected
    atomic_write_json(args.output/'data.json', config)
    mixture = BridgeMixture(config, 20261008)
    moments, atom_quantiles, by_system = [], [], defaultdict(list)
    rows = []
    try:
        # Six visits per system: H4/H8 and all native clock buckets.
        for index in range(6*sum(map(len, selected.values()))):
            batch, task, history, queries, identity = mixture.next()
            delta = (batch.x[queries]-batch.x[history-1]).double()
            valid = delta[:, batch.loss_mask]
            mean_square = float(valid.square().mean())
            amplitude = valid.square().sum(-1).sqrt().flatten()
            q = torch.quantile(amplitude, torch.tensor([.5, .95, .99, 1.], dtype=torch.float64)).tolist()
            moments.append(mean_square); atom_quantiles.append(q)
            by_system[(identity['domain'], identity['system'])].append(mean_square)
            rows.append(dict(**identity, sample=batch.sample_id[0], history=history,
                queries=len(queries), atoms=batch.x.shape[1], coordinate_second_moment_A2=mean_square,
                displacement_norm_quantiles_A=q))
            if (index+1)%128 == 0:
                print(json.dumps(dict(examples=index+1, total=1536)), flush=True)
    finally:
        mixture.close()
    system_moments = [float(np.mean(v)) for v in by_system.values()]
    sigma_data = math.sqrt(float(np.mean(system_moments)))
    # Conservative per-coordinate signal/noise ratio <=0.01 in every sampled
    # TRAIN window. This is a moment check, not proof of high-D Gaussianity.
    sigma_max = 10*math.sqrt(max(moments))
    sigma_min = max(.001, sigma_data/100.)
    scales = dict(schema='molvid.coordinate_state.scales.v1', units='Angstrom',
        systems=len(system_moments), examples=len(rows), sigma_data_A=sigma_data,
        flow_rho_A=sigma_data, sigma_min_A=sigma_min, sigma_max_A=sigma_max,
        diffusion_log_sigma_distribution='uniform_between_log_sigma_min_and_log_sigma_max',
        high_noise_mean_coordinate_SNR=float(np.mean(system_moments))/sigma_max**2,
        high_noise_max_sampled_window_coordinate_SNR=max(moments)/sigma_max**2,
        high_noise_largest_observed_atom_displacement_over_sigma=max(q[-1] for q in atom_quantiles)/sigma_max,
        method='equal-system second moment of target minus last observed; no separate ligand centering/alignment; six visits per TRAIN system',
        high_noise_limitation='Per-coordinate moments only; not a guarantee of total-variation proximity in high dimensions or exhaustive TRAIN-tail coverage',
        model_calls=0, validation_coordinates_opened=False)
    atomic_write_json(args.output/'scales.json', scales)
    atomic_write_json(args.output/'rows.json', rows)
    common = dict(schema='molvid.coordinate_state.recipe.v1', data=config, scales=scales,
        width=256, spatial_layers=6, temporal_heads=4, num_rbf=50, radius_A=10., max_neighbors=32,
        updates=2048, warmup_updates=128, learning_rate=.0001, weight_decay=.01, clip_grad=1.,
        bf16=True, cpu_threads=8, save_every=256, sampler_intervals=18, sigma_schedule_power=7.,
        initialization='random_from_scratch_all_modules_trainable', static_exposure=0)
    for scheme in ('flow', 'edm'):
        recipe = copy.deepcopy(common); recipe['scheme'] = scheme
        atomic_write_json(args.output/f'{scheme}.json', recipe)
    print(json.dumps(scales), flush=True)


if __name__ == '__main__':
    main()
