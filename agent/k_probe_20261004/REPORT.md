# K: independent finite-time response-kernel probe

## Identity

Branch `exp/k-finite-time-20261004` starts from ABCD `946917b` and imports only
shared feature/reporting utilities (`f8c5204`, patch-equivalent to U `2cf532c`).
No S/D2, Mamba or U predictor code/weights are required. A1 generator and
teacher are frozen, using parent SHA `52d922f8…aa14a9`; no K2 training loss.

K0 learns conditional expectations of bounded Gaussian kernels in future
**local scalar-feature change** space. This deliberate first experiment avoids
calling chemical identity clusters conformational states. It is NOT global
state occupancy, event probability, first-passage probability or committor.
The chosen kernel dictionary cannot certify a full path distribution.

## Data and independently fitted targets

Read-only cache from `/workspace/molvid-u-probe-20261004/runs/uk_20261004/features_v1`:
168 fit / 24 calibration systems within the original 192-system train split;
8 original valid systems; no test. All replicas, two windows each, 64 atoms,
H4/dt100 and query horizons 100/400/1200 ps. Cache selection/hash contracts and
whole-system separation are checked before training. The original A1 did see
calibration systems in its pretraining; only these new heads did not.

Target is a fixed 16-dimensional scalar-teacher projection of future-minus-last
features. Global per-channel scales use fit responses only and are the SAME
across all horizons, preserving a single response-region definition over time.
Input standardization is fit-only. U/K share measurements, never predictions.

Fit 16 kernel centers by 20 deterministic k-means iterations on at most 8,192
fit responses; bandwidth is each region's 80th-percentile training RMS radius
with an explicit positive floor. Centers/scales never use calibration/valid.
Target `exp(-0.5 * RMS_distance² / bandwidth²)` is in [0,1]; its conditional
expectation is NOT a normalized categorical distribution or event probability.

## K0 arms and evidence

- **K_last**: last-frame scalar input plus time, historical channels masked.
- **K_history**: same-sized network with frozen history and observed change.
- Analytical **lag-only** kernel means and **persistence** (zero change)
  baselines are fitted/defined without validation labels.
- Three paired seeds 101/102/103; 2,000 steps/arm/seed; batch512, AdamW
  lr3e-4/decay1e-4/clip1; same minibatches across arms. Two hidden layers of128.
- Loss: MSE on real-MD kernel targets. Report system-equal, horizon-separated
  errors and paired system-bootstrap CIs. Raw calibration bins on the 24
  head-held-out systems and on valid are descriptive, not independent-atom CIs.
- No validation-selected seed or predictor. No predicted/generated pseudo-labels.

## K1 generator diagnostic

After K0, use the first PREDECLARED seed's K_history head (101), not the best
validation head. Frozen A1 generates Euler16/FP32 samples with seeds0/1/2/3
for one H4 anchor in each of8 validation systems. Re-encode the **decoded
coordinates**, not the internal generated latent, with the frozen teacher.
Generated bpos is rebuilt using observed-only atom anchors/group centroids;
no true future bpos is reused when encoding generated structures.

Keep four distinct comparisons: generated-vs-real kernel mean, predictor-vs-real,
generated-vs-predictor, and persistence-vs-real. Also retain a fixed-RBF MMD²
over pooled local responses and ordinary bond/clash/RMSF/MSD metrics.
MMD² is descriptive with correlated atoms, can be negative, and is not a
calibrated significance test. Four draws and one real successor per anchor
are a noisy bounded diagnostic, not comprehensive distribution validation.

If K_history does not beat last-state/lag-only controls, do not use it as a
training critic. Even if it passes, K2 requires independent real-trajectory
evaluation and a separate controlled experiment; K2 is disabled here.

## Checks and launch

Six CUDA boundary/model tests passed, plus one complete two-arm 8-step
numerical-fixture training/reload test. Tests cover bounded reproducible kernel
targets, history masking, system splits, future-input isolation and generated
bpos independence from hidden future data. Fixture tests are not real-data gains.

```bash
python -u -m tools.run_k_probe \
  --cache /workspace/molvid-u-probe-20261004/runs/uk_20261004/features_v1 \
  --output runs/uk_20261004/k2000 --steps 2000 --wait-cache
```

GPU4, at most25% memory. Shared extraction runs once, not redundantly in this
branch. Dependency state is `k2000.dependency.json` and times out after4h or
fails if the producer fails. `training_summary.json` is K0 completion;
`generation_diagnostic/summary.json` is K1 completion; root `summary.json`
requires both. No large parent-checkpoint copies or environment changes.
