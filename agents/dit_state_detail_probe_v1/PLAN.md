# PLAN — R2/R4 state-detail latent DiT probe v1

## 1. Question and outcome

This phase asks one bounded question:

> Can one shared molecular DiT implementation learn the frozen R2-SD and R4-SD codec latents,
> and does the four-frame tokenizer retain its compute advantage after generated latents are
> decoded back to trajectories?

The phase implements the common DiT path and proves its contracts. It does not select a final
world-model architecture, claim long-rollout dynamics, or begin scaling-law training.

R4-SD is the provisional primary candidate because it produces four temporal tokens and 8C active
latent volume for a 16-frame clip. R2-SD is the provisional safety candidate because it produces
eight temporal tokens and 16C active volume with a simpler two-frame state/detail decomposition.
R1-SD remains a codec reconstruction upper bound. R4 matched pooling is not a DiT candidate.

## 2. Repository and isolation

Expected existing repository:

```text
/data4/users/sihao/workspace/PVB
```

Audited source at plan creation:

```text
repository: https://github.com/SHw-1313/molvid.git
source branch: fix/state-detail-codec-v2-t1-gates
source commit: 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541
new branch: feat/dit-state-detail-probe-v1
recommended worktree: /data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1
```

The active T1 checkout and every T1 output are read-only. Create a separate branch and worktree.
Never edit, reset, clean, rebase, stash, or reuse the working directory in which T1 is running.
If the source branch has advanced, record the new commit and compare it with the audited source.
Do not silently change the codec implementation or inherit an unreviewed semantic change.

## 3. Current authorization boundary

Authorized now:

- phase documentation and root `AGENTS.md` transition on the new branch;
- latent adapter, normalization, interpolant, DiT, observation-mask plumbing, trainer, evaluator,
  configuration, checkpoint contracts, and tests;
- CPU/synthetic correctness tests;
- one bounded real-CUDA smoke per candidate on an idle GPU, each at most 100 optimizer steps and at
  most 15 minutes;
- generation of an operator review packet.

Not authorized now:

- modifying or opening the T1 test split;
- choosing T1 winners before its frozen validation procedure completes;
- training on the full 48-system T1 train split;
- any DiT run longer than the bounded smoke;
- joint codec/encoder fine-tuning;
- static-data mixing, multiple physical-time buckets, long rollout, AF3/MSA, VAE/KL/VQ,
  classifier-free guidance, energy guidance, ensemble control, or scaling-law experiments.

After the implementation and smoke packet, stop at `WAITING_FOR_T1_AND_OPERATOR_REVIEW`.

## 4. Unified tensor notation

Use the following notation in code comments, assertions, contracts, and reports:

```text
B  batch size
T  decoded trajectory frames; v1 uses T=16
N  total atoms in a ragged batch
M  total molecular blocks/residues in a ragged batch
C  frozen codec width; current value C=128
r  temporal ratio, r in {2, 4}
K  latent temporal length, K=T/r; K=8 for R2 and K=4 for R4
Dh DiT scalar width
Dv DiT vector-channel width
```

Frozen codec output:

```text
state_h   [K, N, C]
state_v   [K, N, 3, C]
detail_h  [K, N, C]
detail_v  [K, N, 3, C]
token_mask              [B, K]
block_frame_mask        [B, K, r]
block_time_ps           [B, K, r]
abid                    [N]
static topology fields  [N] or [2, E_static]
sample_origin           [B, 3]
```

`raw_detail_h` and `raw_detail_v` are diagnostics from the deterministic codec. They are not part
of the generative latent, must not enter the DiT, and must not be required when decoding generated
latents.

## 5. Frozen codec boundary

The DiT consumes only the encoded R2-SD or R4-SD latent. The TorchMD frame encoder, centered-vector
stem, state/detail codec, and coordinate decoder are frozen and run in evaluation mode.

Required invariants:

- codec parameters receive no gradients;
- the DiT never receives target coordinates, raw Haar coefficients, radius edges, or frame-expanded
  graph metadata;
- only static N-axis topology, physical time, masks, atom/block identity, and one `[B,3]` origin
  accompany the generated latent;
- decoding uses the public no-target-coordinate codec API;
- R2 and R4 use the same DiT code and the same hidden-size/depth policy;
- checkpoints record the exact codec checkpoint SHA256 and codec model-contract hash.

The implementation may use a reviewed T0 checkpoint for shape/smoke work. A scientific pilot must
wait for operator-approved T1 codec checkpoints and training-only latent statistics.

## 6. Latent packing and normalization

For each `(time block, atom)` token, pack state and detail on the channel axis:

```text
h_codec = concat(state_h, detail_h, dim=-1)  [K, N, 2C]
v_codec = concat(state_v, detail_v, dim=-1)  [K, N, 3, 2C]
```

Project to the model widths:

```text
h = Linear(2C -> Dh)(h_codec)                 [K, N, Dh]
v = AxisPreservingLinear(2C -> Dv)(v_codec)  [K, N, 3, Dv]
```

The vector projection is bias-free and acts only on the channel axis. It must never mix xyz axes.
State and detail are not expanded into separate sequence tokens; otherwise the intended temporal
token reduction would be lost.

`sample_origin` is derived only from the clean observed prefix: use the observed frame-0 centroid
when H>0 and the fixed zero vector when H=0. It must never be computed from target-only future
coordinates.

Statistics are fitted on the training split only and stored separately for R2 and R4:

- scalar mean and standard deviation per channel for state_h and detail_h;
- vector RMS scale per channel for state_v and detail_v;
- no directional vector mean subtraction;
- state masks exclude padded/invalid tokens, and detail masks additionally include the codec's
  `detail_valid` field;
- zero or non-finite scales fail before training;
- the statistics file records manifest, codec, split, ratio, and source hashes.

Normalization and inverse normalization must round-trip valid synthetic values within FP32
tolerance. R2 and R4 statistics may differ, but their estimation algorithm is identical.

## 7. Flow objective

Use a minimal rectified-flow objective in normalized latent space. Do not reuse the old coordinate
pair matcher without replacing its rank-specific broadcasting assumptions.

For normalized data latent `z_data`, isotropic Gaussian `eps`, and one flow time `tau ~ U(0,1)` per
sample:

```text
z_tau   = (1 - tau) * eps + tau * z_data
u_target = z_data - eps
```

The DiT predicts `u_target` for each of the four generated fields. The same scalar `tau` is used
for scalar and vector fields of one sample. Vector noise is isotropic over xyz, so rotating both
the data and sampled noise rotates the vector target and prediction.

Loss is the equal-weight mean of four mask-normalized field losses:

```text
L = 1/4 * (L_state_h + L_detail_h + L_state_v + L_detail_v)
```

Each field is averaged over its valid predicted elements before fields are combined. Do not let
the three Cartesian components or the larger R2 token count implicitly reweight the objective.
No coordinate-space auxiliary loss, KL, VQ, adversarial term, or unreviewed SNR weighting is added
in v1.

## 8. Observation task

The codec always tokenizes the same complete 16-frame clip. Observation versus prediction is a
DiT training-task mask, not a different codec.

The public batch interface carries:

```text
frame_observation_mask [B, T]
latent_observation_mask [B, K]
```

For v1, prefix-completion observations must end on a block boundary shared by R2 and R4. Supported
history lengths are `H in {0, 4, 8}` frames. A latent token is observed only when every valid frame
in its codec block is observed. Partial-block observation must fail with an actionable error; it
must never reveal a coefficient that depends on a future frame.

During training and sampling:

- observed latent elements remain clean;
- noise and flow loss apply only to unobserved valid tokens;
- an observation flag is embedded into each token;
- observed tokens are clamped after every ODE update;
- invalid/padded elements remain zero and never contribute to attention or loss.

The two-frame-history experiment is intentionally deferred. It requires a separate observation
conditioner or a reviewed partial-block policy for R4 and must not be smuggled into this phase.

## 9. Molecular DiT block

Implement a factorized scalar/vector DiT rather than dense attention over all `K*N` atom-time
tokens.

Each block contains:

1. block/residue spatial interaction at each latent time;
2. temporal attention across `K` tokens for the same atom;
3. scalar/vector feed-forward interaction;
4. AdaLN-Zero modulation from flow time and declared conditions.

AdaLN may add scalar shifts. Vector modulation is multiplicative/gated only; it must not inject an
additive vector derived from invariant conditions.

### 9.1 Spatial interaction

Pool atom features to molecular blocks using `block_id` and `abid`, apply per-sample block-level
attention, then broadcast block context back to atoms. Attention queries/keys are invariant
scalars. The same scalar attention weights mix scalar and vector values. Vector channel maps are
bias-free and do not mix xyz.

Atom-level dense all-pairs attention is prohibited. The implementation must report `N`, `M`, and
the actual attention complexity. A simple dense block-level backend is permitted for the bounded
probe, but its contract must identify it so a later sparse/block-local backend can replace it.

### 9.2 Temporal interaction

Use full bidirectional attention over the latent `K` axis during denoising. Do not reuse the
prefix-causal mask from the deterministic temporal codec. Forecasting causality is enforced by
the observation mask and by hiding unknown clean latents, not by preventing noisy future tokens
from interacting during denoising.

Temporal attention uses invariant scalar queries/keys, scalar/vector values, the real
`block_time_ps`, and masks. Frame indices are not physical time.

### 9.3 Conditioning

The minimum conditioning set is:

```text
flow time tau
physical block clock and time span
ratio id {2,4}
observed/unobserved flag
atom type, block type, component id
```

AF3/MSA, energy, slow-mode, ensemble, task text, and environment controls are outside v1.

### 9.4 Initial probe size

Use one configurable implementation. Suggested smoke defaults are:

```text
Dh=256
Dv=128
depth=4
heads=8
ffn_multiplier=4
dropout=0
```

These are smoke defaults, not a production model. R2 and R4 must use exactly the same model-size
rule and approximately the same parameter count. Do not tune each ratio separately before the
comparison.

## 10. Sampling

Implement deterministic ODE sampling with explicit steps and a fixed solver in v1. Euler with
`8` and `16` step settings is sufficient for correctness/performance comparison. Sampling must:

- start unobserved fields from isotropic normalized noise;
- preserve observed fields exactly by clamping after every step;
- preserve invalid masks as zero;
- inverse-normalize the final generated fields;
- reconstruct a valid `StateDetailLatent` without raw-detail diagnostics;
- decode through the frozen codec;
- record seed, solver, step count, codec hash, stats hash, and model hash.

No stochastic SDE, classifier-free guidance, rejection sampling, or energy correction is added.

## 11. Implementation layout

Prefer focused files and small compatibility edits:

```text
module/state_detail_latent_adapter.py
module/molecular_dit.py
module/latent_rectified_flow.py
trainer/dit_trainer.py
evaluation/dit_evaluation.py
config/dit_state_detail_probe.yaml
scripts/run_state_detail_dit_smoke.py
tests/test_state_detail_latent_adapter.py
tests/test_molecular_dit.py
tests/test_latent_rectified_flow.py
tests/test_dit_trainer.py
```

Modify `module/__init__.py` or evaluation exports only when necessary. Do not rewrite
`module/model.py`, the codec algebra, the frame encoder, or the T1 scripts.

## 12. Correctness evidence

CPU/synthetic tests must cover:

- R2 and R4 shape contracts and shared code path;
- exclusion of raw detail and target coordinates;
- normalization round-trip and train-only statistics provenance;
- scalar invariance and vector/coordinate SE(3) equivariance;
- isotropic vector-noise rotation fixture using the same rotated noise;
- no xyz mixing in every vector projection;
- no cross-sample, cross-padding, or invalid-token contamination;
- block pooling/broadcast correctness for ragged samples;
- physical-time sensitivity and frame-index independence;
- observation-mask conversion for H=0/4/8;
- rejection of partial R4/R2 blocks and future-dependent clean-token leakage;
- observed-token clamping at every solver step;
- four-field loss normalization;
- checkpoint/config/stats/codec-contract round-trip;
- deterministic seeded sampling;
- legacy codec and active T1 tests remain unchanged.

The bounded CUDA smoke must demonstrate:

- finite forward/backward/optimizer execution for R2 and R4;
- nonzero finite gradients in every intended DiT module;
- frozen codec and frame encoder hashes remain unchanged;
- checkpoint save/resume for one step;
- generated latent decodes without `raw_detail_*`;
- runtime, peak allocated/reserved memory, and trunk-only token throughput;
- no collision with active T1 GPUs or output directories.

## 13. Planned scientific pilot after later approval

This section is design only and is not authorized by this prompt.

Use the same independent-system training/validation provenance as T1, approved R2/R4 best codec
checkpoints, and training-only latent statistics. Train matched small DiTs separately for R2 and
R4 with identical optimizer, parameter policy, data exposure, observation mixture, solver, and
budget. Keep test closed.

Required comparisons:

- codec oracle decode versus generated-latent decode;
- H=4 primary prefix completion and H=8 auxiliary completion;
- persistence and constant-velocity baselines;
- future aligned RMSD and dRMSD;
- bond RMSE, contact F1, clashes;
- velocity/acceleration error, dynamic correlation, RMSF correlation, frequency power;
- observed/generated boundary error;
- latent flow loss by field;
- trunk-only throughput, peak memory, and token count;
- at least four samples per condition for diversity diagnostics; do not use best-of-N as primary.

Provisional default rule, frozen before a real pilot:

- prefer R4 if its generated future aligned RMSD and dRMSD are each no worse than 1.10 times R2,
  contact F1 and dynamic correlation each fall by no more than 0.02 absolute, and boundary error
  is no worse than 1.25 times R2;
- otherwise retain R2 as the default;
- report both absolute results and the compute difference; do not hide a quality/efficiency
  Pareto trade-off inside one composite score.

## 14. Stop and review

After implementation, tests, and the bounded smoke:

```text
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
real DiT pilot: NOT_STARTED
T1 test access from this worktree: FORBIDDEN
```

The operator review packet must distinguish:

- code-contract pass/fail;
- bounded execution pass/fail;
- unverified scientific hypotheses;
- exact tasks that require T1 completion and later authorization.
