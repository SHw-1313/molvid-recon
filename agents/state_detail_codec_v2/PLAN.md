# PLAN — zero-preserving state/detail temporal codec v2

## 1. Goal

Build and validate a deterministic, block-local temporal tokenizer for a future molecular Video
DiT. Compare ratio-1, ratio-2, and ratio-4 state/detail representations while keeping the selected
TorchMD spatial encoder fixed. Determine whether four-frame compression can retain useful motion
without the information destruction observed in the old single-bank ratio-4 pooling codec.

This phase ends after code validation and the three-system/nine-trajectory T0 experiment. The
64-system T1 experiment is specified for later planning but is not authorized in this execution.

## 2. Questions answered in this phase

1. Are R1/R2/R4 lifting, masking, scalar/vector shapes, and inverse decoding correct?
2. Does the detail path preserve exact zero motion by construction?
3. Can coordinates be reconstructed without a complete per-atom `x0` bypass?
4. On the prior three-system/nine-trajectory protocol, do all four controls train normally and
   use their latent banks rather than silently collapsing or reading the target?
5. At matched `8C` latent feature volume, does explicit R4 state/detail behave differently from
   generic two-head pooling?

T0 cannot select the production ratio. Its output is an operator-review packet that decides
whether the implementation is trustworthy enough to run T1.

## 3. Starting point

```text
repository: /data4/users/sihao/workspace/PVB
source branch: feat/visnet-spatial-v2
source commit: 045dbb8809e9d7aee418eb355b6477e05088d4fd
target branch: feat/state-detail-codec-v2
spatial backend for all new runs: torchmd_et
```

ViSNet v1/v2 implementations and evidence remain reproducible history. Do not remove them. The
choice of TorchMD is fixed for this phase; no spatial-backbone experiment is permitted.

## 4. Scope boundaries

In scope:

- new deterministic state/detail codec classes and contracts;
- R1/R2/R4 block packing and decoding;
- capacity-matched R4 unstructured pooling control;
- removal of the new path's complete per-atom `x0` coordinate bypass;
- static T=1 schema support and synthetic repeated-static correctness tests;
- unchanged coordinate/geometry/dynamics training losses;
- T0 engineering, three-system training, evaluation, reports, and operator pause.

Out of scope:

- a prediction/forecasting objective or history/future split;
- two-frame observation, 18-frame windows, observation adapters, or generated prefixes;
- DiT, diffusion, flow, rollout, or trajectory-generation decoder;
- VAE, KL, posterior sampling, VQ, MMD, adversarial latent matching;
- AF3/MSA/PLM conditions;
- static/dynamic large-data joint training;
- multi-timescale scientific comparison;
- T1 data materialization or training;
- geometry-backbone changes.

## 5. Codec task

For trajectory examples, the encoder sees the full valid 16-frame clip and the decoder
reconstructs those same frames. A future training task will decide which frames are observed by
masking or corrupting latent tokens; the tokenizer itself does not change its block layout.

Static structures remain genuine `T=1` samples. They are not repeated into 16-frame fake
trajectories. A repeated-static T=16 clip exists only as a synthetic property test.

## 6. Spatial feature interface

For each frame and atom, TorchMD provides:

```text
h: [T, N, C]       invariant scalar channels
v: [T, N, 3, C]    SE(3)-equivariant vector channels
C = 128 in T0
```

Every temporal addition, subtraction, normalization, projection, and pooling operation on `v`
must leave the Cartesian axis intact. Learned vector maps operate only on channel axes; scalar
features may supply invariant gates.

## 7. Coordinate gauge and no-anchor contract

The old model passes the complete first-frame coordinates as `CodecLatent.x_anchor` and decodes:

```text
x_hat[t, atom] = x0[atom] + delta[t, atom]
```

The new path must not do this. It may retain only a per-sample translation origin:

```text
origin[b] = masked arithmetic centroid of valid atoms in frame 0
x_centered[t, atom] = x[t, atom] - origin[abid[atom]]
```

The encoder receives centered frames. The coordinate decoder reconstructs centered per-atom
coordinates from latent scalar/vector features, then adds `origin[b]`. No per-atom coordinates
are stored in the new latent metadata. Rotation is represented by equivariant vector channels,
not by a leaked frame.

The centroid rule, atom mask, dtype, and sample mapping must be explicit and tested. If data are
already centered, the same rule still applies and records the near-zero origin.

Old checkpoint schemas retain their old `x_anchor` behavior. New and old coordinate contracts
must not be silently interchanged.

## 8. Temporal transform

Let `F` denote either h or v.

### R1

```text
S = F0
```

One C-wide state bank per frame; detail is absent and `detail_valid=false`.

### R2

```text
S = (F0 + F1) / sqrt(2)
D = (F1 - F0) / sqrt(2)
```

One C-wide state bank and one C-wide detail bank per two-frame block.

### R4

```text
S01  = (F0 + F1) / sqrt(2)
D01  = (F1 - F0) / sqrt(2)
S23  = (F2 + F3) / sqrt(2)
D23  = (F3 - F2) / sqrt(2)
S    = (S01 + S23) / sqrt(2)
Dmid = (S23 - S01) / sqrt(2)
```

The state bank encodes `S` at width C. The detail encoder compresses `[Dmid,D01,D23]` into one
C-wide bank. The detail decoder expands it back to those three ordered coefficients; fixed inverse
Haar then reconstructs four frame features.

These terms describe common content and intra-block residuals. They are not a learned separation
of molecular slow and fast modes.

## 9. Latent capacity

For a 16-frame clip:

| Control | Time tokens | Active banks per token | Active feature volume |
|---|---:|---:|---:|
| R1-SD | 16 | state C | 16C |
| R2-SD | 8 | state C + detail C | 16C |
| R4-SD | 4 | state C + detail C | 8C |
| R4-Matched-Pooling | 4 | bank A C + bank B C | 8C |

R2 halves the DiT time-token count but intentionally preserves total latent feature volume. R4
quarters the time-token count and halves total active feature volume. Reports must include both
notions of compression.

## 10. Structured latent schema

The new state/detail latent must expose, without relying on positional guessing:

```text
state_h, state_v
detail_h, detail_v                    # absent/masked for R1 and static T=1
token_mask
detail_valid / detail_component_mask
block_frame_mask
block_time_ps and/or exact frame_time_ps needed for decoding
abid
topology
sample_origin                        # [B,3], never [N,3]
ratio, mode, coefficient ordering, width metadata
```

The matched-pooling latent exposes two unstructured banks with the same `4 x 2C` capacity but
must not be mislabeled as state/detail.

## 11. Zero-preserving construction

Required identities:

```text
E_detail(0) = 0
D_detail(0) = 0
```

Use bias-free channel projections, zero-origin activations, no additive affine detail offsets,
and no dropout in the detail path. State and clock information may gate a nonzero detail
multiplicatively, but cannot add detail. The decoder's state contribution is shared within a
block; within-block variation can only come from decoded detail.

For a repeated-static input, all detail coefficients, detail latents, decoded detail
coefficients, within-block feature differences, and coordinate differences must be zero within
the declared FP32 tolerance. This is a correctness property, not a training loss.

Absolute time must not create motion. Relative time may condition reconstruction only through a
zero-preserving detail gate.

## 12. Matched pooling comparator

Use two independent learned scalar-gated pooling heads over each non-overlapping four-frame
block. Each head produces one C-wide h/v bank, for total width 2C and volume 8C over four tokens.
Its decoder expands the two unstructured banks into four frame features without the per-atom x0
bypass.

This control asks whether state/detail algebra adds value beyond extra latent capacity. It is not
required to satisfy zero motion. Its parameter count may differ naturally and must be reported.

The existing old R4 single-bank result remains a historical observation only.

## 13. Decoder

The state/detail decoder performs:

```text
state bank -> one full-width common coefficient
detail bank -> zero-preserving detail coefficient(s)
fixed inverse lifting -> per-frame h/v
shared framewise equivariant coordinate head -> centered coordinates
add sample origin -> x_hat
```

Do not use the current additive time-query upsampler in the new path. Do not enable the optional
spatial refiner. The decoder API must not accept target coordinates.

R1 uses the same state projection and framewise coordinate head but performs no temporal
aggregation. It is the zero-compression rate point, not a separate no-temporal ablation.

## 14. Masks and clocks

Invalid frames cannot contribute to state or detail. A token and each detail coefficient need
explicit validity. Full T0 trajectories have sixteen valid frames; static T=1 and synthetic
padding tests must remain supported.

Use actual `time_ps`; do not infer time from the frame index. Haar decomposition remains over
ordered frame features. Relative times are preserved as metadata/conditioning so irregular and
different native intervals are not conflated. T0 scientific runs use only `dt_100ps` to avoid
mixing the time-scale question into codec validation.

## 15. Losses

No latent distribution or magnitude loss is added. Use the existing components:

```text
coordinate:   1.0
local:        0.1
bond:         0.1
velocity:     0.1
acceleration: 0.05
```

Resolve the schedule after the loader size is frozen:

| Training fraction | Active terms |
|---|---|
| 0–10% | coordinate + local + bond |
| 10–30% | add velocity |
| 30–100% | add acceleration |

Do not add zero-motion consistency, detail shrinkage, state/detail orthogonality, KL, VQ, or
feature-reconstruction losses to rescue a failed implementation.

## 16. Correctness gates

Before real GPU training, verify:

- fixed lifting and inverse lifting analytically and numerically;
- exact R1/R2/R4 token/bank shapes and capacity counts;
- h/v masks, static T=1, T=16, padded/partial blocks, and irregular clocks;
- zero detail and zero coordinate motion for repeated-static clips;
- nonzero detail and finite/nonzero detail gradients for moving clips;
- scalar invariance, vector and coordinate equivariance, and translation-origin behavior;
- no per-atom coordinate bypass and no decoder target input;
- frame/sample isolation;
- old checkpoint/legacy code regression;
- new config/model/checkpoint round trip;
- save/resume and optimizer state;
- full test suite and source checks.

## 17. T0 protocol

### T0-A: bounded micro/overfit

Use the exact three selected systems to prove forward/backward, component gradients, checkpoint
resume, loss decrease, and finite metrics. Include at least one single-clip overfit whose purpose
is correctness rather than model comparison.

### T0-B: prior three-system/nine-trajectory run

Expected immutable data identity:

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
three replicas per system
T=16
dt=100 ps
441 train clips
117 fixed late-holdout clips
```

The worker must verify this against the stored indexes/manifests and stop on mismatch. Train all
four controls for thirty complete epochs with exact nonreplacement epoch coverage, one seed,
FP32, common frozen TorchMD frame-encoder weights, the same optimizer/loss schedule, and no
spatial refiner.

T0 is evidence that code and training behavior are sane. Do not select R2 or R4 from three
systems.

## 18. Evaluation

Report absolute values and curves for:

- aligned RMSD and dRMSD, overall and per frame;
- error grouped by block offset (R2: 0/1; R4: 0/1/2/3);
- bond RMSE, contact error, clash rate;
- velocity and acceleration RMSE;
- RMSF correlation/error;
- one lagged-correlation or ACF metric;
- frequency retention;
- block-boundary jump;
- state/detail or bank A/B norms and utilization;
- R2/R4 full decoding versus detail-zero decoding;
- latent token count, active elements, parameters;
- spatial, packing, decoding, and end-to-end time/memory;
- per-system metrics, not only pooled-window averages.

Repeated-static correctness is reported separately from dynamic quality. Static T=1 schema is
tested but static training is not part of T0.

## 19. Operator review barrier

After T0, create a review packet and set:

```text
status: WAITING_FOR_OPERATOR_REVIEW
```

The packet must make it possible to inspect:

- code diff and contracts;
- zero-motion tests;
- no-anchor evidence;
- all four training curves;
- absolute train/holdout metrics;
- detail utilization and counterfactual decoding;
- performance and memory;
- checkpoint resume;
- warnings, failures, and unresolved questions.

Stop before any T1 data selection, manifest generation, training, or evaluation.

## 20. Planned T1 after later approval

This is documentation only, not current authority:

```text
64 independent systems
48 train / 8 validation / 8 test
up to 3 trajectories per system, about 192 total
24–32 sampled 16-frame clips per training trajectory per epoch
20 minimum, 30 target, 40 maximum epochs
one seed for all four controls
common frozen TorchMD frame encoder
single native time bucket
target <=18–20 GPU hours per control
```

With two GPUs, run two waves; with four, one control per GPU. Selection uses validation only and
test is evaluated after configuration/stopping rules are frozen. T1 requires a later explicit
operator prompt.

## 21. Repair addendum — T1 gates (2026-09-04)

The operator review requested a repair-and-repeat-T0 cycle. The accepted Haar/state-detail
algebra, four control names, deterministic AE objective, `torchmd_et` backend, and no-anchor
constraint remain binding. This addendum does not authorize T1.

### Repair hypotheses

1. The latent topology field is currently frame-expanded and coordinate-dependent; replacing it
   with a validated static N-axis chemical schema will remove a future information leak without
   changing the current decoder.
2. The frozen TorchMD vector features plus a pointwise coordinate head may not contain enough
   centered coordinate information. A cached single-clip diagnostic must separate feature
   insufficiency from decoder insufficiency before selecting a bounded repair.
3. A learned centered-coordinate equivariant vector stem is the smallest permitted repair. If it
   cannot pass the declared single-clip R1 reconstruction gate, a bounded generated-latent global
   decoder may be evaluated; no fixed per-atom coordinate anchor may return.
4. Several evaluator labels are currently scientifically inaccurate: raw coordinate error is not
   aligned RMSD, contact-count difference loses pair identity, absolute-coordinate ACF is mostly
   static shape, and clip-wise RMSF/aggregation semantics are underspecified.

### Repair gates

- Static topology metadata is N-axis, coordinate-independent, T-invariant, and excludes radius
  edges, distances, vectors, frame-expanded arrays, and target coordinates.
- Aligned RMSD, centroid-gauge raw RMSD, pair-aware contacts, dynamic ACF, and explicitly aligned
  RMSF are implemented and fixture-tested while preserving legacy/raw names where needed.
- A genuine one-sample single-clip R1 run reports curves and an operator-visible threshold frozen
  before training. It must be finite, converge clearly, improve over centroid/origin-only
  reconstruction, and exercise every intended new module with nonzero finite gradients.
- R1/R2/R4-SD share the same reconstruction stem/global decoder and initialization policy; the
  matched control is reported as latent-volume-matched with its natural parameter count.
- Repeated-static zero detail/motion, genuine T=1, partial-block policy, SE(3), target isolation,
  checkpoint/config, legacy regression, frozen encoder, and exact accepted capacity contracts
  continue to pass.
- A repeated T0 is allowed only after the single-clip R1 gate and ratio smoke pass. It uses the
  same 441/117 data, seed, FP32, `torchmd_et`, frozen common encoder, four controls, and fixed
  loss schedule; reports absolute train/holdout curves and all required evaluator/runtime
  diagnostics.

### Required order and stop condition

1. Audit source state, reproduce the old headline rows, and record pre-repair hashes.
2. Repair static topology metadata and evaluator semantics with focused tests.
3. Run the one-clip cached-feature R1 diagnostic and then the smallest permitted reconstruction
   repair. Stop if the no-anchor/frozen-feature combination is not decodable.
4. Run bounded R2/R4-SD/matched smoke only after R1 passes; then repeat the exact T0 if all gates
   remain valid.
5. Append observed evidence to the handoff, create a new operator packet with the actual repair
   commit and artifact paths, set `Status: WAITING_FOR_OPERATOR_REVIEW`, and stop.

No T1 manifest, system selection, T1 benchmark, T1 training/evaluation, DiT, forecasting,
observation masking, rollout, AF3/MSA, VAE/KL/VQ, static-data training, or later architecture
phase may begin in this repair cycle.
