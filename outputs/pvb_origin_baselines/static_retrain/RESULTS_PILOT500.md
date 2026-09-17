# Baseline C result: pilot500 restart

Status: completed for the requested pilot-limited policy.

## Run

- Reference: `/data4/users/sihao/workspace/PVB_origin`, commit `c08e5e3cd49d45c6d748387e78224843bd356f50`; not edited.
- Static initialization: `/data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt`.
- Data: ATLAS `dt_100ps` and MISATO `dt_80ps`; no PDB data.
- Adapter: output-local `streaming_origin_adapter_v2.py`, one lazy adjacent pair (`frame 0 -> frame 1`) per T=16 clip; no 15x materialization.
- Seed: `20260810`.
- Device: host GPU 7 via `CUDA_VISIBLE_DEVICES=7`; process-local device `cuda:0`.
- Config: `origin_train_config_pilot500.yaml` (`ubound_per_batch=5000`, `max_batches=500`, `max_epoch=1`).
- Original components used: `train.py`, `DynamicBatchWrapper`, `collate_fn`, `DynamicTrainer`, and `dyVAE._train`.

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/data4/users/sihao/workspace/PVB_origin \
  WANDB_MODE=offline PYTHONUNBUFFERED=1 python -u \
  /data4/users/sihao/workspace/PVB/outputs/pvb_origin_baselines/static_retrain/run_origin_train.py \
  --config /data4/users/sihao/workspace/PVB/outputs/pvb_origin_baselines/static_retrain/origin_train_config_pilot500.yaml \
  --seed 20260810 --pairs-per-clip 1 --gpus 0
```

## Coverage and metrics

The reference wrapper formed 500 train and 500 valid dynamic batches per source: 1,000 train batches and 1,000 valid batches total. The selected pair counts were:

| split/source | clip-pairs available | selected pairs | pairs >5,000 atoms in full split | max atoms |
|---|---:|---:|---:|---:|
| train / ATLAS | 101,556 | 1,115 | 3,906 | 17,002 |
| train / MISATO | 41,910 | 748 | 5,982 | 20,561 |
| valid / ATLAS | 17,484 | 1,202 | 558 | 8,521 |
| valid / MISATO | 4,740 | 707 | 1,098 | 19,013 |

- Train loader completed 1,000 batches; 8 batches hit the original trainer's CUDA-OOM skip path, leaving `global_step=992` optimizer updates.
- Valid loader completed 1,000 batches with no CUDA-OOM skip.
- Original validation mean loss: `1.8790495992898941`.

## Outputs

- Training/validation log: `train_pilot500.log`.
- Checkpoint: `checkpoints_pilot500/version_0/checkpoint/epoch0_step992.ckpt` (39,965,095 bytes; SHA-256 `a7f5332152321f873224dd243697e5a191a0851cc5253af3161e269ac30d2954`).
- Validation ranking: `checkpoints_pilot500/version_0/checkpoint/topk_map.txt`.
- Exact split/batch audit: `pilot500_data_audit.log`.
- Logged run phase: 16:14:03 to 16:22:50 CST, approximately 8m47s; validation began at 16:20:13 CST.

The `>5,000` counts are full-split diagnostics; the reference `DynamicBatchWrapper` excludes those records from this pilot policy, which is the intentional memory boundary corresponding to the requested `ubound=5000`. The resulting metric is therefore the requested pilot500 validation metric, not a full-dataset metric.

The prior `24000/60000` run was not resumed: it stopped around `172/14797` with continuous OOM skips and produced no checkpoint. Its provenance is in `OLD_RUN_ABORTED.md` and `train_full.log`.
