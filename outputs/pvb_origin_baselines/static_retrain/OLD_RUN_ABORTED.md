# Baseline C old run (aborted)

- Run: `run_origin_train.py` with `origin_train_config_full.yaml`, stream adapter, `pairs_per_clip=1`, seed `20260810`.
- Device request: host GPU 7 (`CUDA_VISIBLE_DEVICES=7`; process-local `cuda:0`).
- The old run used `ubound_per_batch=24000` and was stopped after continuous original-trainer `CUDA out of memory, skip batch` messages at approximately `172/14797` training batches.
- No checkpoint was produced under `checkpoints_full/version_0/checkpoint/`; only the original trainer's `train_config.json` directory exists.
- This run is not resumed and its partial log is retained in `train_full.log` for provenance. A new run will use a new output checkpoint directory and will wait for GPU 7 to be free.
