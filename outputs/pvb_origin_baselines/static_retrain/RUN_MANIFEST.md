# Baseline C run manifest

This file records the full-run choice and the compatibility boundary before
launch.

## Inputs

- Reference code: `/workspace/PVB_origin` (host: `/data4/users/sihao/workspace/PVB_origin`), commit `c08e5e3cd49d45c6d748387e78224843bd356f50`.
- Static pretrained checkpoint: `/data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt`.
- Checkpoint selection: first/top row of the checkpoint directory's `topk_map.txt`, metric `0.26337328421718936`.
- Seed: `20260810`.
- Device: physical GPU 7, exposed as logical `cuda:0` through `CUDA_VISIBLE_DEVICES=7`.
- No PDB records are in the train or validation inputs.

## Split and mapping

The prepared native clip stores are retained as the source of truth:

| source | native bucket | train clips | valid clips | mapping |
|---|---:|---:|---:|---|
| ATLAS | 100 ps | 101,556 | 17,484 | frame 0 → frame 1 |
| MISATO | 80 ps | 41,910 | 4,740 | frame 0 → frame 1 |

The full run uses `streaming_origin_adapter_v2.py`, not the small
`smoke_pairs` materialization. Every train/valid clip is indexed and exposed
as one lazy original-PVB adjacent pair; each clip is decompressed only when a
batch requests it. This is the explicit boundary adopted to avoid creating
15 legacy records per T=16 clip and a much larger duplicate store. The
adapter supports all 15 adjacent pairs, but this run uses the native first
transition once per clip so the current pilot's clip count and system split
remain one-to-one.

The original `MMAPDataset` format was verified incompatible with the clip
`npz-v1` payload (`PK` ZIP magic versus gzip-JSON). `/data5/PVB` was checked;
its original-format stores are PDB EPT data only and were excluded. No
reference-repo edit or PDB fallback is used.

The old `ubound_per_batch=8000` would discard 558 ATLAS train, 186 ATLAS
valid, 1,794 MISATO train, and 372 MISATO valid records based on the original
wrapper's `item_len > ubound` rule. The full config therefore uses 24,000;
the largest indexed clip has 20,561 atoms. This is a config-level memory
accommodation, not a code change. If a larger record produces an actual CUDA
OOM, the original trainer's warning and the exact skipped record counts will
be reported as a runtime blocker.

## Original entrypoint and config

The unmodified original `/workspace/PVB_origin/train.py` is run through the
output-local `run_origin_train.py` wrapper. The wrapper only injects the
20260810 seed and replaces `UniDataset` with the lazy adapter; the original
`dyVAE._train`, `DynamicTrainer`, collator, optimizer, validation loop, and
checkpoint serialization are used unchanged.

```bash
cd /workspace/PVB
source /home/sihao/miniforge3/bin/activate torch-ito
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/workspace/PVB_origin \
  WANDB_MODE=offline \
  python -u outputs/pvb_origin_baselines/static_retrain/run_origin_train.py \
  --config outputs/pvb_origin_baselines/static_retrain/origin_train_config_full.yaml \
  --seed 20260810 --pairs-per-clip 1 --gpus 0 \
  > outputs/pvb_origin_baselines/static_retrain/train_full.log 2>&1
```

The config runs one full split pass (`max_epoch: 1`, `max_batches: 60000` per
source wrapper), with original PVB fine-tuning settings `lr=5e-5`, warmup
1,000, gradient clip 1.0, and `model_type: md`.

## Preflight evidence

The exact checkpoint loaded successfully. One real optimizer/backward preflight
completed on GPU 7 for ATLAS (887 atoms, peak 7,822,351,360 bytes) and MISATO
(4,460 atoms, peak 40,337,958,400 bytes). No quality metric is inferred from
that preflight.
