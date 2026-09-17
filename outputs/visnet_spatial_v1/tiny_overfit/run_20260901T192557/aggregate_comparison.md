# ViSNet spatial backbone v1 tiny-overfit report

The report separates implementation correctness, the three-clip micro-overfit, and the final tiny-overfit quality gate.

Implementation status: **passed**.
Development-backend recommendation: **torchmd_et** (both ViSNet variants failed implementation or tiny-overfit quality gates).

| Backbone | Parameters | Implementation | Micro 500-step | Tiny quality | Train loss reduction | Holdout future dRMSD improvement | Final future dRMSD | Train seconds | Train tokens/s | Peak alloc (GiB) | Peak reserved (GiB) | Graph mode |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| torchmd_et | 1974904 | True | True | True | 66.776% | 49.836% | 0.6448 | 4157.3 | 82979.5 | 56.25 | 78.71 | external |
| visnet_radius | 2133316 | True | True | False | 49.392% | 35.579% | 0.682237 | 3536.9 | 97533.0 | 36.09 | 78.33 | native_radius |
| visnet_bonded | 2133316 | True | True | False | 49.552% | 35.288% | 0.679828 | 3721.7 | 92690.1 | 36.11 | 78.62 | external |

## Required protocol facts

- Manifest: 441 train and 117 late-holdout clips; 0 overlap; T=16; dt_100ps.
- Loader: lazy mmap, `replacement=false`, exact no-replacement epoch coverage, FP32, `max_tokens=80000`, 30 epochs, safety cap 6000 steps.
- Every epoch records all train sample IDs exactly once and evaluates all 117 holdout clips, with per-system metrics.

## Artifacts

- JSON: `outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/aggregate_comparison.json`
- CSV: `outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/aggregate_comparison.csv`
- Plots: `outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/loss_curves.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/loss_curves.pdf, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics.pdf, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_torchmd_et.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_torchmd_et.pdf, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_visnet_radius.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_visnet_radius.pdf, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_visnet_bonded.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/eval_metrics_visnet_bonded.pdf, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/performance_summary.png, outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/performance_summary.pdf`

## Limitations

- no protein isolation.
- one seed.
- natural rather than parameter-matched models.
- this is a development protocol, not a final scientific benchmark.
