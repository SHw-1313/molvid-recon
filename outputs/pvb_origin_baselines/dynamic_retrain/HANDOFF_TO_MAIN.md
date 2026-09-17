# Baseline D execution handoff

The worker-side launch is stopped. Do not launch another D worker process.

At handoff verification, the only remaining D command was:

```text
python -u /workspace/PVB/outputs/pvb_origin_baselines/dynamic_retrain/run_baseline_d_train.py \
  --config /workspace/PVB/outputs/pvb_origin_baselines/dynamic_retrain/baseline_d_train_container.yaml \
  --ref-repo /workspace/PVB_origin \
  --output-dir /workspace/PVB/outputs/pvb_origin_baselines/dynamic_retrain \
  --seed 20260810 --gpus 0
```

It was running as PID 1504686 on host GPU4 (CUDA-visible remap is expected to
be `CUDA_VISIBLE_DEVICES=4`, process cuda:0), and was left untouched because
the main thread owns it. Host GPU7 remained occupied by C. The worker-side
GPU4 launch used a separate `baseline_d_train_gpu4.yaml`/log and was stopped;
that partial log is retained and is not a result.

Policy to audit in the main-thread run: `ubound_per_batch=5000`,
`max_batches=500` per source, `max_epoch=1`, seed `20260810`, streaming
all-adjacent-pairs adapter, no PDB, and the original dynamic checkpoint.
