"""Strictly load retained B/C/D artifacts through the merged inference loader."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.runtime import atomic_write_json, configure_device, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('use a fresh artifact verification output')
    root = Path('/workspace')
    teacher = root / 'molvid-recon-gm-motion-v3/inputs/codec_best.pt'
    teacher_sha = 'ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df'
    droot = root / 'molvid-recon-static-decoder-probe-v1'
    dmanifest = json.loads((droot / 'agent/static_decoder_probe_v1/artifacts/train_manifest.json').read_text())
    cases = [('B_M-short', root / 'molvid-recon-learned-memory-v1/runs/learned_memory_v1/formal_r02/configs/formal/M-short/frame_joint_step_00009741.pt',
              'd20360a75e2c0d3110d4d3aaed5484fa9024635d67384d9f4a28510736257ea0')]
    cpath = root / 'molvid-recon-gm-capacity-v3/runs/gm_motion_memory_v3/C/r10/configs/formal/C4/frame_joint_step_00013575.pt'
    # C4 is completed; C8's live files are deliberately not inspected/loaded.
    cases.append(('C4_completed_not_selected', cpath, sha256_file(cpath)))
    for arm, row in dmanifest['arms'].items():
        cases.append(('D_' + arm, droot / f'runs/static_decoder_probe_v1/formal_500/{arm}/frame_joint_step_00000500.pt', row['checkpoint_sha256']))
    device = configure_device('cuda', deterministic=True)
    started = time.monotonic()
    results = []
    for label, checkpoint, expected in cases:
        loaded = load_frame_joint_inference(checkpoint, expected_sha256=expected,
            codec_path=teacher, codec_sha256=teacher_sha, device=device)
        model = loaded['model']
        results.append({'label': label, 'checkpoint': str(checkpoint), 'sha256': expected,
            'step': int(loaded['payload']['step']), 'history_backend': model.history_backend,
            'depth': model.dit.depth, 'state_tensors': len(model.state_dict()),
            'contract': dict(model.contract()), 'strict_load': True,
            'scope': 'contract/state reconstruction only; not evaluation or promotion'})
        print(json.dumps({'label': label, 'strict_load': True}), flush=True)
        del model, loaded
        gc.collect()
        torch.cuda.empty_cache()
    atomic_write_json(args.output, {'results': results, 'test_opened': False,
        'gpu_hours': (time.monotonic() - started) / 3600})


if __name__ == '__main__':
    main()
