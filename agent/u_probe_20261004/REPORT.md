# U: independent frozen-history future-moment probe

## Identity and limits

Branch `exp/u-future-probe-20261004` starts from ABCD `946917b`.
It contains no S/D2 or Mamba modifications. Common feature/reporting utilities
were fixed at `2cf532c` and separately cherry-picked to the K branch; no U/K
predictor weights are shared. This is a **probe**, not a new memory backend,
variance-conditioned flow, or demonstrated improvement to Molvid generation.

Parent: frozen A1 checkpoint SHA
`52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.
Teacher SHA `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
The complete parent is eval-only and has `requires_grad=False` during cache
extraction. All generator, decoder, history and teacher update counts are zero.

## Shared real-MD feature protocol

- 192 original train systems are hash-partitioned into **168 fit / 24 head
  calibration systems**; the original 8 validation systems remain separate.
  A1 was pretrained on the 24 calibration systems: these are held out from
  the new heads, NOT unseen by the full pretrained model. No sealed test.
- All three replicas, two deterministically spaced windows per replica:
  1,008 fit clips, 144 calibration clips, 48 validation clips. No random
  overlapping-window split or validation-driven sample selection.
- Training ordinary dt100 records and SHA-verified dt100/H4 validation views;
  H4 observation, query horizons **100/400/1200 ps**. The default dt300 valid
  store is not silently mixed into this matched-grid experiment.
- 64 evenly indexed supervised atoms per clip; retain system/trajectory/clip
  and selected-atom identity. Atoms and windows are correlated observations.
- Inputs: last normalized teacher scalar h, observed H4 endpoint h change,
  frozen A1 legacy-memory scalar h, plus physical query-time features. Raw
  observations are sliced BEFORE the teacher call; future x/bpos cannot
  influence these features.
- Target: future-minus-last normalized scalar h projected by a fixed seeded
  orthonormal 128→16 map. Only scalar channels are used: this does not assert
  vector covariance, full molecular-state uncertainty or joint path modeling.
- Inputs are standardized on fit systems only. Targets use a **single global
  per-channel fit RMS scale across horizons**, no valid/calibration statistics.
  Cache uses FP32 and one-frame teacher chunks, independent of training precision.
- Cache hashes, parent identity, projection and selections are persisted.
  U and K consume the same immutable cache; neither consumes the other's model.

## U arms and fixed budget

| Arm | Observed information | Distribution head |
|---|---|---|
| U_last | last-frame scalar h only | mean + diagonal variance |
| U_history_mu | last h + frozen history/change | mean, fixed unit variance |
| U_history_muvar | last h + frozen history/change | mean + diagonal variance |

Identical network shapes (two 128-wide hidden layers); unused history channels
are zeroed after fit normalization. All optimize Gaussian NLL, with variance
fixed in the mean-only arm; log variance is bounded [-6,4]. A single successor
is a supervised draw, **not an observed conditional variance label**. Any claim
about conditional dispersion depends on out-of-sample calibration.

Three paired seeds 101/102/103, 2,000 steps/arm/seed, batch 512 sampled equally
over clips, selected atoms and horizons, AdamW lr 3e-4 / decay 1e-4 / clip1.
Same minibatch RNG per seed across arms. There are also analytical persistence
and lag-only mean/variance baselines fit only on the fit split. No seed is
chosen by validation performance; all nine head checkpoints are retained.

Variance calibration: one scalar temperature per horizon on the 24 calibration
systems, bounded [0.05,20]. Report both raw and calibrated NLL, MSE, 50/90/95%
marginal coverage, and interval width. Increasing variance alone is not success.
Average atoms/windows within system before bootstrap; 2,000 resamples of
systems per seed. These CIs are not estimates of between-training-seed variance.

No automatic integration: require history benefit beyond last-state and
lag-only controls, useful sharpness with coverage, and seed consistency.
Failure is evidence to revisit representation/targets; it is not a reason
to feed a miscalibrated variance into the flow source.

## Checks and launch

Six targeted CUDA tests passed, including future-coordinate/bpos isolation,
frozen-parent gradients, system split/equal reporting, calibration axes,
history masking, positive variance, strict reload and complete 8-step
three-arm numerical-fixture training. Fixture runs are not scientific results.

Real shared-cache extraction is launched on GPU 4, container PID 393467,
from source `2cf532c`. It uses at most 25% of device memory and leaves the
existing foreign job, S on GPU 2, and C8 on GPU 5 untouched. No env upgrade.

```bash
python -u -m tools.run_u_probe \
  --cache runs/uk_20261004/features_v1 \
  --output runs/uk_20261004/u2000 --steps 2000 --wait-cache
```

The wait is a bounded dependency wait (4h), not hidden training: its status is
`u2000.dependency.json`. Cache failure fails the dependent job. Logs and
`summary.json` distinguish waiting, training, failure and completion. Only
small probe checkpoints are saved; no copies of the full A1 checkpoint.
