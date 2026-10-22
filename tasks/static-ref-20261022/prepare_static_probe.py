"""Prepare twelve fixed MD snapshots for a paired static restoration probe.

These are development MD snapshots, not a held-out PDB generalization set.
The model input contains one corrupted structure; reference futures are only
referenced by the CPU evaluator for the existing geometry normal bands.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path('/workspace/MOLVID-MAIN-20261016')
MAIN_RUN = MAIN/'runs/main_integration_20261016'
RUN = ROOT/'runs/static_ref_20261022'
CAPACITY = Path('/workspace/MOLVID-FLOW-CAPACITY-20261020/runs/flow_capacity_20261020/query_post')
TIME = Path('/workspace/MOLVID-FLOW-TIME-20261021/runs/flow_time_modeling_20261021')
RHO = 1.1397942064485225
CORR = 0.2
STAGES = {
    'old_static4096': (CAPACITY/'static/coordinate_final.pt', 4096),
    'control4096': (RUN/'control/checkpoint_step4096.pt', 4096),
    'control8192': (RUN/'control/coordinate_final.pt', 8192),
    'concat4096': (RUN/'concat/checkpoint_step4096.pt', 4096),
    'concat8192': (RUN/'concat/coordinate_final.pt', 8192),
}


def standalone(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(output):
    if os.environ.get('CONDA_DEFAULT_ENV') != 'torch-ito' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use existing container torch-ito and hide CUDA for input preparation')
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from molvid.data.static import trajectory_record_to_static
    from molvid.data.batch import collate_clip_records
    panel = standalone('probe_common_panel', MAIN/'molvid/evaluation/common_panel.py')
    base = MAIN_RUN/'short50_generated_20261017'
    source = json.loads((base/'manifest_ready.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for case in source['cases']:
        prepared = panel.prepare_case(case, base)
        if prepared.history_frames != 4 or len(prepared.reference) != 50:
            raise ValueError('Only the existing six H4Q50 development references are supported')
        for future_index in (0, 25):
            record_index = prepared.history_frames + future_index
            identity = f"{case['id']}_f{future_index:03d}"
            record = trajectory_record_to_static(prepared.record, frame_index=record_index, sample_id=identity)
            clean = torch.as_tensor(np.asarray(record['x'])).float()
            generator = torch.Generator(device='cpu').manual_seed(17)
            corruption = CORR*torch.randn(clean.shape, generator=generator)
            # Actual native source is CUDA-generated seed29, separately from this
            # fixed corruption; it is saved by the native runner for exact pairing.
            payload = dict(record=record, corrupted_A=clean+corruption,
                coordinate_unit='angstrom', corruption_seed=17, source_seed=29,
                corruption_sigma_A=CORR, source_rho_A=RHO,
                case=case['id'], reference_future_index=future_index, source_record_index=record_index,
                source_time_ps=float(prepared.record['time_ps'][record_index]),
                source_record=str(prepared.record_path), topology=prepared.topology)
            batch = collate_clip_records([record])
            if batch.frames != 1 or not torch.equal(batch.x, clean):
                raise ValueError('Static record conversion changed the selected real coordinates')
            path = output/(identity+'.pt')
            torch.save(payload, path)
            snapshots.append(dict(id=identity, case=case['id'], dataset='ATLAS' if case['id'].startswith('atlas_') else 'MISATO',
                reference_future_index=future_index, source_record_index=record_index,
                source_time_ps=payload['source_time_ps'], input=str(path), atoms=batch.atom_count,
                source_record=str(prepared.record_path), reference_case={k:v for k,v in case.items() if k!='models'}))
    manifest = dict(schema='molvid.static_ref.probe.v1', scope='Twelve development MD snapshots; not held-out PDB generalization',
        snapshots=snapshots, stages={name:dict(checkpoint=str(path), expected_step=step,
            available_at_preparation=path.is_file()) for name,(path,step) in STAGES.items()},
        corruption_seed=17, source_seed=29, corruption_sigma_A=CORR, source_rho_A=RHO,
        full_sampler=dict(task='static', queries=1, intervals=18, NFE=35),
        teacher_probe=dict(progress=.5, NFE=1, uses_clean_target=True,
            reason='Teacher-noised single-field diagnostic, reported separately from true-source restoration'),
        geometry_reference='Same existing Q50 MD normal bands, not a band fitted to one static frame',
        reference_manifest=str(base/'manifest_ready.json'), GPU_seconds=0, model_calls=0)
    panel.write_json(output/'manifest.json', manifest)
    print(json.dumps(dict(manifest=str(output/'manifest.json'), snapshots=len(snapshots),
        stages=manifest['stages'], model_calls=0, GPU_seconds=0)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RUN/'static_probe/inputs')
    prepare(p.parse_args().output.resolve())
