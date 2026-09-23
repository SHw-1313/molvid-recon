# Frame GM Calibration v2

Status: P1--P3 complete (2026-09-24). This is exploratory validation evidence,
not a production-parity claim. Sealed test data was never opened.

## Scope and provenance

The work ran only in the dedicated
`/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2` worktree on
`feat/frame-gm-calibration-v2`, based on
`f49cf27efa60a0cca172248c49913d70ff1516a5`. Python, CUDA, training and
evaluation commands ran inside `enter-container` with `torch-ito`.

The original Frame Joint parent was `frame_joint_step_00022272.pt` (SHA-256
`b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`);
the frozen codec SHA was
`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
Training used 192 systems and 576 physical trajectories. Formal evaluation
used eight validation systems, three replicas, two anchors and seeds 0/1/2.

## Actual model and one-step data flow

The implementation stays in the shallow `molvid/` package. For B systems,
N packed atoms, H observed frames, Q query frames and C=128 latent channels:

- observed scalar/vector latents are `[H,N,C]` and `[H,N,3,C]`; observed
  coordinates are `[H,N,3]`;
- physical query times and their `[B,Q]` mask are separate from scalar flow
  time `s in [0,1]`;
- the DiT uses scalar width 256, vector width 128, four blocks and eight heads;
  its future tensors are `[Q,N,C]` and `[Q,N,3,C]`;
- the frozen target teacher encodes training targets, while the trainable
  history encoder, DiT and coordinate decoder receive observed context,
  topology and query time. Generation never receives hidden future coordinates;
- packed-system and query masks exclude padding from graphs, attention, losses
  and metrics. The codec/teacher stays frozen; gradients in joint training end
  at those boundaries.

G inserts sparse covalent one-hop / 1--3 topology messages after the spatial
part of DiT blocks 2 and 4. It reads invariant information from the current
unnormalized vector state, uses observed last-frame distances as conditions,
and mixes vectors only with invariant gates. A one-sided zero gate preserves
the off/initial path while still allowing startup gradients.

M computes observed-only per-atom motion summaries from history increments and
last-frame-relative differences. It adds zero-preserving scalar/vector
conditioning plus explicit `query_horizon_ps`, `query_delta_ps`,
`history_span_ps` and observed intervals. These physical-time features are
separate from flow time; no future token enters the context.

For P3 J1, every eighth successful update starts K=2 samples from
`repeat(last_observed_normalized_latent) + N(0,I)`, takes four differentiable
Euler steps, decodes coordinates and applies condition-local energy score plus
an observed-bond term. Feature groups are residue RMSF, displacement-squared /
adjacent increments, and fixed-topology internal-distance increments/products.
The entire sampled branch is inside one outer forward/backward; an independent
auxiliary RNG leaves the paired main noise and data stream unchanged. J0 has
the same base objective, exposure and optimizer policy without this auxiliary.

## P1: metric and data repair

Schema-v3 metrics distinguish displacement increment RMSE in Å from finite-
difference velocity RMSE in Å/ps. Dynamic coordinate metrics align every frame
to the same last-observed reference; internal-distance and torsion increments
remain alignment-free. RMSF reports atom/residue errors and correlations plus
system-mean correlation, MAE and spread, with explicit unavailable reasons.
All dependent fields were recomputed from saved coordinates rather than
partially patching an old table.

The fixed-history family uses observed times `[-300,-200,-100,0] ps`, a
1,200 ps horizon and 100/200/300/400 ps query grids with 12/6/4/3 real future
frames. Padding is masked, not repeated as stationary truth. Manifest SHA-256 is
`900fbd41f40754dc61aee22320e751389e0ef7eabd628b751c24250bc0ea76c3`;
the valid store index SHA is
`c8b91ad7a430de907ab4695f87bb0eb97f234efb2db618b6b9aa246f58ccb7d5`.

The train-only, future-informed latent diagnosis found normalized unit-source
noise RMS near 11.31 (h) and 19.59 (v), versus latent-motion RMS ranges
3.40--3.68 and 1.70--2.13. Generated-error RMS was 3.56--3.75 and 2.56--2.95.
Equal-normalized-norm H4 perturbations moved decoded coordinates by only about
0.0036--0.0042 Å through h but about 0.194 Å through v. This diagnoses a scale
and sensitivity mismatch; it does not establish a better prior, and P1 did not
change noise, teacher, normalization statistics or the main flow objective.

## The three evaluation families

1. Geometry: bond RMSE, angle error/validity and clash rate, with aligned RMSD,
   dRMSD and contact F1 as supporting evidence.
2. Motion: atom/residue RMSF MAE and correlation, plus system-mean RMSF MAE,
   correlation and spread ratio.
3. Time: fixed-history MSD curve error and common physical horizons, native-grid
   MSD, finite-difference velocity/displacement increments, internal-distance
   and torsion increments, and correct-versus-wrong-clock diagnostics.

Aggregation is draw/seed mean, then anchor mean, replica mean and equal system
weight. Confidence intervals use paired resampling of systems, never atoms.
Formal generation is observed-only Euler16 and never best-of-N.

## P2: G/M four-arm result

All four arms warm-started matching model weights but reset optimizer,
scheduler, cursor and RNG symmetrically. Each completed two effective epochs
and 21,208 successful updates with identical exposure and sampler manifests.

| Arm | Main result relative to B0 | Final checkpoint SHA-256 |
|---|---|---|
| B0 | frozen fallback | `ff8121aa47a7c1885b4ad59ece586dcc90a3d8a12c67d71fb767f586c6b36258` |
| G | bond improved; fixed-history MSD and clash worsened | `e3ba2dfcf89b13f76b3633b1730f56cf8bd7359680051384f4931fe85893d8e6` |
| M | residue RMSF improved; bond and clash worsened | `6c7ad59594a4a806e29c4e5e6a1de30dbcf1ad435cc67ab647ca553123ae0091` |
| GM | bond and residue RMSF improved; clash worsened | `aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540` |

No non-baseline arm passed both the frozen candidate and safety gates.
`selection.json` therefore selected B0 for P3; its SHA-256 is
`198216cac78afae06fc6da4672f91db42b568414c4504c5932009784d88e993b`.
This negative result was retained rather than forcing GM.

## P3: sampled-distribution result

Eight fixed train calibration batches resolved feature scales to 1.02159 Å
(residue RMSF), 3.58516 Å² (displacement squared), 0.041901 Å (distance
increment) and 0.00222178 Å² (increment product). Gradient calibration targeted
0.05 of main-flow gradient norm for each auxiliary and resolved weights
0.000264925 (energy score) and 0.0236574 (observed bond). The 64-update paired
pilot, strict resume/continuation, target mutation, BF16 and DDP checks passed;
42 targeted tests passed.

Both formal arms completed 21,208 updates from the same B0 parent. J1 activated
the auxiliary exactly 2,651 times, from update 8 through 21,208. Main exposure,
sample identities, history, query/flow time, bucket and view streams matched
J0 exactly.

| Error (lower is better) | J0 | J1 | J1 - J0, paired 95% CI |
|---|---:|---:|---:|
| bond RMSE (Å) | 0.41299 | 0.33115 | -0.08184 [-0.08707, -0.07705] |
| fixed-history MSD curve MAE (Å²) | 2.61775 | 3.11625 | +0.49850 [0.42882, 0.57977] |
| residue RMSF MAE (Å) | 0.41197 | 0.43163 | +0.01967 [-0.02448, 0.07637] |

Thus sampled calibration produced a real geometry--time trade-off: bond error
clearly improved while fixed-history MSD clearly worsened. The RMSF interval
crossed zero. The predeclared optional fork required all three directions to be
clear, so `bond_fork_decision.json` says not to run the bond keep/release fork
(SHA-256
`fc1fd8f159641c693b78d4b9df36025080cef4e408b0ef61b848b43b709c5d0f`).

Final checkpoints are J0
`60e14c3e7cc59c8e4c1a2ff38512df829db7418446cc82bd3ed3587722696f5e`
and J1
`8859a499be7d06f31ba33fb9e71febf5019cebb9ced1947d0c277326da0c66e1`.

## Review and cost

P2 formal launch passed independent r02 review, and the later selector-path
repair passed fresh r07 review. P3 r01 requested one budget-evidence fix; after
ordinary and sampled cadence-weighted profiles were added, fresh no-context
r02 returned PASS. Review records live under
`agent/frame_gm_calibration_v2/reviews/`.

P2 formal training is accounted at 31.581 GPU-hours from the frozen profile
ledger (limit 96). P3 actual per-device elapsed training time was 6.569 hours
for J0 and 7.996 for J1, or 14.565 GPU-hours total; the prelaunch profile bound
was 16.850 (limit 96). The eight P2 generation evaluations consumed 6.130
GPU-hours and the four P3 evaluations 3.479. Pilot/profile and metric-recompute
logs do not all contain elapsed fields, so no false exact grand total is
reported; recorded work stayed within the frozen allocations.

## Limitations and recommendation

The evaluation has only eight validation systems and remains exploratory model
selection. It has no sealed-test result and no comparison establishing
production parity. J1 should not replace B0/J0 merely because bond RMSE is
lower: its fixed-history time behavior is clearly worse.

If a later task is authorized, the most informative next study is an
independent-validation investigation of source-prior scale and the conflict
between local geometry penalties and trajectory-distribution objectives. It
should be a new frozen protocol, not a post-hoc change to this task's metrics or
trigger. No fourth stage, extra fork, source-noise change or hidden production
claim belongs to this completed scope.
