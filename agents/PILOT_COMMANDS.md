# PVB codec pilot commands

These commands use the versioned half-data stores named in `HANDOFF.md`. They do not modify the legacy pair stores. Replace `<GPU_ID>` with an actually idle GPU and keep the seed fixed at `20260810` for the pilot.

```bash
source /home/sihao/miniforge3/bin/activate torch-ito
cd /workspace/PVB
export PROCESSED_DATASET_ROOT=/data4/users/sihao/data/pvb_cross_dataset_20260810
export ATLAS_TRAIN=$PROCESSED_DATASET_ROOT/clips/atlas/dt_100ps/train
export ATLAS_VALID=$PROCESSED_DATASET_ROOT/clips/atlas/dt_100ps/valid
export MISATO_TRAIN=$PROCESSED_DATASET_ROOT/clips/misato/dt_80ps/train
export MISATO_VALID=$PROCESSED_DATASET_ROOT/clips/misato/dt_80ps/valid
```

## G5 controls

Train each control into a separate directory so checkpoints cannot be mixed:

`config/codec.yaml` uses `max_tokens: 80000`, which admits the observed 71,360-token MISATO clips while excluding larger clips that exceed the pilot memory bound. Normalization is fit on a fixed 256-batch training sample so the 143k-clip stores do not require a full decompression pass before the 1000-step pilot.
Every non-dry-run training command appends per-step metrics to <save-dir>/train_metrics.jsonl by default. Use --log-every to thin the records; the visualization command below consumes these logs.

```bash
python train_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-steps 1000 --fit-normalization --normalization-batches 256 --temporal-ratio 1 --temporal-layers 0 --save-dir outputs/g5/ratio1_no_temporal --log-every 1 --train-root "$ATLAS_TRAIN" --train-root "$MISATO_TRAIN" --valid-root "$ATLAS_VALID" --valid-root "$MISATO_VALID"
python train_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-steps 1000 --fit-normalization --normalization-batches 256 --temporal-ratio 1 --temporal-layers 1 --save-dir outputs/g5/ratio1_temporal --log-every 1 --train-root "$ATLAS_TRAIN" --train-root "$MISATO_TRAIN" --valid-root "$ATLAS_VALID" --valid-root "$MISATO_VALID"
python train_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-steps 1000 --fit-normalization --normalization-batches 256 --temporal-ratio 4 --temporal-layers 1 --save-dir outputs/g5/ratio4_temporal --log-every 1 --train-root "$ATLAS_TRAIN" --train-root "$MISATO_TRAIN" --valid-root "$ATLAS_VALID" --valid-root "$MISATO_VALID"
```

Evaluate all three with native-bucket stratification. Substitute the actual checkpoint filenames emitted by each training command:

```bash
python eval_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-batches 32 --valid-root "$ATLAS_VALID" --valid-root "$MISATO_VALID" --ratio1-no-temporal-checkpoint outputs/g5/ratio1_no_temporal/codec_step_00001000.pt --ratio1-temporal-checkpoint outputs/g5/ratio1_temporal/codec_step_00001000.pt --ratio4-temporal-checkpoint outputs/g5/ratio4_temporal/codec_step_00001000.pt --json outputs/g5/codec_eval.json --markdown outputs/g5/codec_eval.md
```

Generate loss curves and native-time metric comparison plots; the script auto-discovers the three train_metrics.jsonl files under outputs/g5/:

```bash
python scripts/plot_codec_results.py --eval-json outputs/g5/codec_eval.json --output-dir outputs/g5/plots
```

## Per-bucket controls

Run separate commands when a go/no-go decision must isolate one native interval. The 100 ps and 80 ps rows are real half-data; a 1 ns row must use an explicitly supplied 1 ns store or synthetic smoke fixture and must not be relabeled from another interval.

```bash
python train_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-steps 1000 --temporal-ratio 4 --temporal-layers 1 --save-dir outputs/buckets/dt_100ps --train-root "$ATLAS_TRAIN" --valid-root "$ATLAS_VALID"
python train_codec.py --config config/codec.yaml --device cuda:<GPU_ID> --seed 20260810 --max-steps 1000 --temporal-ratio 4 --temporal-layers 1 --save-dir outputs/buckets/dt_80ps --train-root "$MISATO_TRAIN" --valid-root "$MISATO_VALID"
python smoke_codec.py --device cuda:<GPU_ID> --data-root "$ATLAS_VALID" --checkpoint outputs/smoke/atlas_t16.pt --output outputs/smoke/atlas_t16.json
```

## Smoke gate before pilot

```bash
python smoke_codec.py --device cuda:<GPU_ID> --data-root "$ATLAS_VALID" --output outputs/smoke/atlas_t16.json
python smoke_codec.py --ddp --ddp-gpus 0,5 --output outputs/smoke/ddp.json
```

If CUDA is unavailable, only this diagnostic is permitted; it is not a GPU result:

```bash
python smoke_codec.py --cpu-fallback --output outputs/smoke/cpu_t16.json
```

## Result table

| Native bucket | Clip span | Control | Ratio | Latent interval | Frame-0 RMSD | Future dRMSD | Bond RMSE | Contact error | Clash rate | Velocity RMSE | Acceleration RMSE | HF retention | Peak GPU | Samples/s | Go/no-go |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100 ps | 1.5 ns | ratio1_no_temporal | 1 | 100 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 100 ps | 1.5 ns | ratio1_temporal | 1 | 100 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 100 ps | 1.5 ns | ratio4_temporal | 4 | ~400 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 80 ps | 1.2 ns | ratio1_no_temporal | 1 | 80 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 80 ps | 1.2 ns | ratio1_temporal | 1 | 80 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 80 ps | 1.2 ns | ratio4_temporal | 4 | ~320 ps | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1 ns | 15 ns | ratio4_temporal | 4 | ~4 ns | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

No quality or go/no-go claim is valid until the corresponding bucket row has a completed JSON/Markdown report and an actual GPU checkpoint.

The executed pilot artifacts are under `outputs/g5/`: all three `codec_step_00001000.pt` checkpoints, `codec_eval.json`, and `codec_eval.md`. The recorded evaluation used 32 validation batches and reported 48 clips per control across `dt_100ps` and `dt_80ps`; it is a pilot observation, not a final go/no-go decision.
The logged rerun artifacts are under `outputs/g5_logged/`; `outputs/g5/ratio*/train_metrics.jsonl` contains 1,000 records per control, and `outputs/g5_logged/plots/` contains PNG/PDF loss curves, native-bucket metric comparisons, and `eval_metrics.csv`.
