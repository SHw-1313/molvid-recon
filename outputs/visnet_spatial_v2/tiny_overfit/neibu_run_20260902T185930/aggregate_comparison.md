# ViSNet spatial backbone v2 tiny comparison

Outcome classification: **IMPLEMENTED** — v2 parity passed but tiny quality gate failed.

| Backbone | Implementation | Tiny quality | Final holdout total | Future dRMSD | Future RMSD | Future bond RMSE | Velocity RMSE | Frequency | Train min | k tokens/s | Holdout eval s/epoch | Peak alloc GiB | Peak reserved GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| torchmd_et | reused_immutable_v1 | True | 0.306157 | 0.6448 | 0.873627 | 0.180768 | 0.00418078 | 0.670033 | 69.3 | 83.0 | 47.1 | 56.25 | 78.71 |
| visnet_radius | reused_immutable_v1 | False | 0.348693 | 0.682237 | 0.932993 | 0.177395 | 0.00445324 | 0.636193 | 58.9 | 97.5 | 45.4 | 36.09 | 78.33 |
| visnet_bonded | reused_immutable_v1 | False | 0.345613 | 0.679828 | 0.92835 | 0.174751 | 0.00444298 | 0.619523 | 62.0 | 92.7 | 46.0 | 36.11 | 78.62 |
| visnet_v2_bonded | new_v2 | False | 0.30961 | 0.646277 | 0.879702 | 0.160754 | 0.0042243 | 0.678905 | 170.9 | 33.6 | 57.7 | 35.51 | 37.15 |

## Locked protocol

- 441 train clips and 117 late-holdout clips; train/holdout intersection is zero.
- Lazy loading, replacement=false, exact no-replacement epoch coverage, FP32, max_tokens=80000, 30 epochs, and a 6000-step safety cap.
- temporal_layers=1, temporal_ratio=1, unchanged decoder/loss/optimizer/evaluator.
- v1 TorchMD and v1 bonded rows are immutable stored results; v2 has a separate result/checkpoint directory.

## Gap recovery

- {"future_drmsd_gap_recovery_vs_v1_bonded_to_torchmd": 0.9578133100411423, "v2_beats_v1_bonded_future_drmsd": true, "v2_beats_v1_bonded_future_rmsd": true, "v2_within_10_percent_of_torchmd_future_drmsd": true, "v2_within_10_percent_of_torchmd_future_rmsd": true}

## Artifacts

- JSON: outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/aggregate_comparison.json
- CSV: outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/aggregate_comparison.csv
- Plots: outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/loss_curves_comparison.png, outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/loss_curves_comparison.pdf, outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/eval_metrics_comparison.png, outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/eval_metrics_comparison.pdf, outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/performance_summary_comparison.png, outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/performance_summary_comparison.pdf
