# Baseline C retry

The old `24000`-bound run was not resumed. GPU 7 was observed free (`0 MiB`) before this retry was launched.

Command inside the runtime container:

```bash
cd /workspace/PVB
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/workspace/PVB_origin \
  WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup python -u \
  outputs/pvb_origin_baselines/static_retrain/run_origin_train_retry.py \
  --config /workspace/PVB/outputs/pvb_origin_baselines/static_retrain/origin_train_config_retry.yaml \
  --seed 20260810 --pairs-per-clip 1 --gpus 0 \
  > outputs/pvb_origin_baselines/static_retrain/train_retry.log 2>&1 &
```

The retry uses the same lazy clip adapter and full current train/valid paths. Its only runtime data-side adjustment is the output-local full-coverage dynamic wrapper with `ubound_per_batch=8000`: records above the bound are retained as singleton batches, so the mapping is not silently truncated. Memory feasibility of such singleton records is tracked separately; a GPU OOM is a blocker, not a fabricated metric.
