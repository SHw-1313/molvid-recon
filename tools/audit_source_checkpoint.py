"""CPU reload/optimizer identity audit; does not evaluate generation quality."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from molvid.checkpoints import load_frame_joint_inference
from molvid.flow.source import FrameSourceConfig
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    config=json.loads(args.config.read_text());codec=config['codec']
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    step=payload['step'];scales=FrameSourceConfig.resolve(config['model'].get('frame_source'))
    if FrameSourceConfig.resolve(payload['contracts']['model'].get('frame_source'))!=scales:
        raise ValueError('checkpoint/config source discrepancy')
    loaded=load_frame_joint_inference(args.checkpoint,expected_sha256=sha256_file(args.checkpoint),
        codec_path=codec['checkpoint'],codec_sha256=codec['sha256'],device='cpu')
    model=loaded['model']
    if model.source_config!=scales: raise ValueError('inference source restored incorrectly')
    path=args.checkpoint.parent/'warm_start_report.json';report=json.loads(path.read_text())
    if any(report[k] for k in ('optimizer_restored','scheduler_restored','cursor_restored','rng_restored')):
        raise ValueError('source intervention mislabeled warm start')
    # A later training step may still be writing; only complete JSONL records
    # can enter the audit of this already-saved checkpoint prefix.
    rows=[json.loads(line) for line in (args.checkpoint.parent/'train_metrics.jsonl').read_text().splitlines(keepends=True)
          if line.endswith('\n')]
    rows=[r for r in rows if r['step']<=step]
    if [r['step'] for r in rows]!=list(range(1,step+1)): raise ValueError('training log update continuity')
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()): raise ValueError('nonfinite saved state')
    counts=Counter(int(s['step']) for s in payload['optimizer_state']['state'].values())
    if not counts or max(counts)!=step or min(counts)<1: raise ValueError('optimizer step evidence missing')
    exposure={(r['step'],r['history_frames'],r['sampled_branch_active'],tuple(r['sample_ids'])) for r in rows}
    freeze_run_identity(args.output,inputs=[args.checkpoint,args.config,path],
        purpose='CPU source checkpoint reload and training audit',protocol=dict(model_calls=0,training_updates=0,
            generation_quality_evaluated=False,scope=f'first{step}updates'))
    atomic_write_json(args.output/'summary.json',dict(status='completed',step=step,
        source_config=scales.contract(),inference_restore='passed',state_tensors=len(model.state_dict()),
        optimizer_step_histogram=dict(counts),logged_updates=len(rows),unique_exposure_entries=len(exposure),
        real_training_systems=len({s.split('_R')[0] for r in rows for s in r['sample_ids']}),
        model_calls=0,training_updates=0,compute='CPU',generation_quality_evaluated=False,
        parent_sha256=report['source_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
