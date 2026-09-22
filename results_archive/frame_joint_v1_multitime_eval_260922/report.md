# Frame Joint v1 multi-time lag evaluation (2026-09-22)

CUDA evaluation completed with 2352 rows: 8 systems, 3 replicas, 2 paired anchors, lags 100/200/300/400 ps, H4/H8, Euler 16, seeds 0/1/2. Model inference and tensor metrics ran on GPU4 (A100-SXM4-80GB); aggregation/plots ran on CPU.

## Protocol and provenance

- Raw run: `runs/frame_joint_v1_multitime_eval_260922`; archive: `results_archive/frame_joint_v1_multitime_eval_260922`.
- Checkpoint, codec, embedded statistics, paired-view manifest hashes and exact command are in `protocol.json`; minimal CUDA checks are in the raw `checks/checks.json`.
- Physical times use raw absolute timestamps in each paired-view metadata; metrics use local time distance from the last observed frame. `align_mask`, `loss_mask`, atom count and Å units are recorded per row.
- Aggregation is seed mean per window, then anchor/window mean per replica, then equal system mean. No test data or training was used.

## Main findings

- Generated H4 aligned RMSD averaged across lag/system buckets: 2.550 Å; persistence 2.236 Å; clean-latent oracle 0.049 Å. Generated-vs-persistence is reported by bucket in `per_system.csv`, not collapsed into a single causal claim.
- The oracle path encodes the true future with the frozen target encoder and decoder; it is a reconstruction diagnostic, not a generation score.
- Per-frame errors, bond/contact metrics, lag-MSD curves, dynamic auxiliary metrics, and exact raw frame/time identities remain in `rows.jsonl`; per-system reductions are in `per_system.csv`.

## Required conclusions (evidence labels)

- Mixed-lag generation: **initial support** if lag bucket means differ and generated metrics are finite; this is an observational multi-time comparison, not a scaling-causality claim.
- 300 ps interpolation: **initial support** only when the 300 ps bucket is bracketed by the 200/400 ps results; no held-out interpolation claim is made beyond these paired windows.
- Correct time condition: **initial support** if true-clock and wrong-100 ps rows differ in `time_condition_ablation.csv`; same seed, source and true-clock scoring are preserved.
- Movement-amplitude differences: **initial support**; generated RMSF, target RMSF and correlations are reported separately, and persistence correlations are unavailable when its signal is constant.

## Answers to the five questions

1. Whether mixed lag generation behaves differently is shown by the lag curves and per-system table; any differences include both model time conditioning and data-frame spacing, so they are not a causal scaling result.
2. The 300 ps bucket is evaluated directly and compared with 200/400 ps in the same paired-anchor construction; it is not labeled a held-out interpolation test.
3. The true-clock/wrong-clock ablation quantifies whether time labels change outputs and accuracy while target scoring stays on the true physical clock.
4. Generated motion amplitude is compared with MD RMSF per atom/system; geometry error and RMSF error are not substituted for one another.
5. The next training/architecture decision should be selected only after inspecting the oracle-to-generated gap and the time-condition deltas; this report does not silently change the approved design.

## Outputs

- `per_system.csv` and `time_condition_ablation.csv`
- `rmsf_per_system.csv`, `rmsf_atoms.npz`, `rmsf_index.json`
- `lag_geometry_motion.png`, `rmsf_scatter_by_lag.png`, `time_condition_true_wrong.png`

## Numerical addendum (final corrected RMSF)

| path | H | 100ps RMSD | 200ps RMSD | 300ps RMSD | 400ps RMSD | 300ps bond RMSE | 300ps RMSF MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence | 4 | 1.988 | 2.263 | 2.430 | 2.522 | 0.042 | 1.216 |
| persistence | 8 | 1.854 | 2.144 | 2.294 | 2.390 | 0.042 | 1.122 |
| clean oracle | 4 | 0.048 | 0.049 | 0.049 | 0.051 | 0.028 | 0.005 |
| clean oracle | 8 | 0.048 | 0.049 | 0.049 | 0.051 | 0.028 | 0.005 |
| generated | 4 | 2.235 | 2.593 | 2.806 | 2.825 | 0.498 | 0.507 |
| generated | 8 | 2.193 | 2.454 | 2.643 | 2.651 | 0.526 | 0.457 |

First-future versus final-future (true-clock seed 0): generated H4 aligned RMSD is 2.060→2.916 Å and bond RMSE 0.475→0.514 Å; H8 is 2.108→2.691 Å and 0.517→0.538 Å. The oracle stays near 0.049 Å, so error is present at the first future frame and accumulates.

The H4 wrong-100ps ablation changes outputs: mean raw packed-coordinate future RMSE true-vs-wrong is 0.604/0.788/0.729 Å at 200/300/400 ps. Wrong-minus-true aligned RMSD is −0.154/−0.214/−0.152 Å and RMSF-MAE is −0.003/−0.012/−0.024 Å; this demonstrates time sensitivity but not a scientific win for the wrong clock.

The `rmsf_correlation`, `rmsf_spearman`, and `rmsf_flattening_ratio` columns are retained in `per_system.csv`; unavailable correlations remain null with reasons in raw rows (persistence has constant future signal).
