# DECISIONS — zero-preserving state/detail temporal codec v2

## D1 — TorchMD is fixed

Use `torchmd_et` for every new codec control. ViSNet v1/v2 code and artifacts remain compatible
history but are not retrained or compared in this phase.

## D2 — codec and generation task remain separate

The codec encodes and reconstructs the complete 16-frame clip. Observation masks, two-frame
conditioning, history/future splitting, DiT denoising, and rollout belong to later phases.

## D3 — deterministic AE

The codec has no KL, stochastic posterior, VQ, MMD, adversarial prior, or latent-distribution
loss. Latent statistics are logged for diagnosis only.

## D4 — dual-bank capacity is intentional

R1 uses 16 C-wide state tokens. R2 uses eight tokens with C-wide state and C-wide detail banks,
preserving 16C active feature volume while halving the future DiT sequence length. R4 uses four
tokens with the same two banks, yielding 8C active feature volume and four temporal tokens.

Ratio denotes temporal-token reduction. Reports must not misstate R2 as a twofold reduction in
all latent elements.

## D5 — hierarchical Haar, not constant-velocity residual

Use fixed orthonormal Haar lifting for R2 and hierarchical R4 state/detail coefficients. The
three R4 detail coefficients are `Dmid`, `D01`, and `D23` in that fixed order. They are temporal
resolutions inside one four-frame block, not molecular slow/fast modes.

## D6 — block-local tokenizer

No cross-block temporal attention is added to the codec. The future DiT is responsible for
long-range temporal interaction. R4's first token may encode frames 0–3 because it is a clean
target token, not an observed forecasting condition.

## D7 — zero motion is enforced by construction

The state/detail detail encoder and decoder satisfy `f(0)=0`; state/time cannot add motion.
Repeated-static zero detail and zero decoded motion are correctness gates, not loss terms or
optional ablations.

## D8 — static and repeated-static have different semantics

Static `T=1` has no valid detail and uses a full C-wide state token. Synthetic repeated-static
`T=16` has valid detail equal to zero. Static structures are never repeated to create training
trajectories.

## D9 — remove the complete x0 bypass in the new path

No per-atom first-frame coordinates are stored in a new latent or given to its decoder. Retain
only one masked centroid/origin vector per sample to restore translation. Geometry must be
reconstructed from latent h/v. Old checkpoint schemas retain their original `x_anchor` behavior.

## D10 — matched pooling is new-framework and capacity matched

The only extra T0 control is an R4 two-head unstructured pooling codec with four 2C-wide tokens
and no per-atom x0 bypass. The old single-bank `4 x C` R4 results are historical and are not
retrained or included in the locked four-way ranking.

## D11 — fixed losses, no semantic regularizer

Use coordinate/local/bond/velocity/acceleration losses and the frozen staged schedule. Do not add
detail shrinkage, orthogonality, feature reconstruction, or zero-motion consistency loss.

## D12 — no spatial refiner

The optional spatial refiner is disabled for all controls so temporal-codec and basic decoder
behavior remain visible.

## D13 — T0 is not architecture selection

The prior three-system/nine-trajectory dataset is an implementation and training-sanity test. It
cannot decide the production ratio or support a scaling claim.

## D14 — frozen TorchMD during the locked T0 comparison

All four controls load identical recorded TorchMD frame-encoder weights and freeze them during
the 30-epoch comparison. Separate smoke evidence must show that gradients can flow when the
encoder is unfrozen. This isolates the temporal codec without forbidding later co-adaptation.

## D15 — hard operator gate before T1

After code gates and T0 reports, the phase stops in `WAITING_FOR_OPERATOR_REVIEW`. T1 data
materialization and execution require a later explicit user approval; passing automated tests
does not authorize continuation.

## D16 — planned T1 scale

After approval, T1 uses 64 independent systems split 48/8/8 by system, up to three trajectories
per system, capped resampled clips, one native time bucket, and a per-control target of 18–20 GPU
hours. This decision records the intended design but grants no current execution authority.

## Repair addendum — T1 gates (2026-09-04)

### D17 — preserve the accepted codec algebra

The repair does not change the four control names, block-local orthonormal Haar transforms,
R4 coefficient order, zero-preserving detail paths, capacity definitions, deterministic AE loss,
`torchmd_et` selection, or no-anchor constraint. Any implementation change must be justified by a
correctness gate or the bounded R1 diagnostic.

### D18 — topology metadata is static chemical metadata

New latent topology is aligned to the N-atom latent axis and may contain atom/block/component
identity and static covalent bonds only. Frame-expanded radius edges, coordinates, distances,
edge vectors, contacts, and frame offsets are prohibited even if the current decoder ignores the
field.

### D19 — reconstruction repair is diagnostic-led

Before selecting a decoder repair, cache one frozen-feature trajectory clip and compare the
current pointwise head against a centered-coordinate vector-stem hypothesis. A vector stem is the
preferred minimal repair; a generated-latent global context module is permitted only if the stem
is insufficient. A fixed per-atom coordinate bypass is never permitted.

### D20 — evaluator names must describe their signal

New metrics distinguish Kabsch-aligned RMSD from centroid-gauge raw RMSD, preserve pair identity
for contacts, compute lag/ACF on a rigid-body-handled dynamic signal, and state RMSF alignment and
aggregation semantics. Existing raw metrics may remain only under explicit legacy/raw names.

### D21 — genuine single-clip gate precedes T0 repeat

The repair must use exactly one sample ID for the R1 overfit diagnostic, freeze the reconstruction
threshold before observing the result, and report train curves plus raw/aligned RMSD, dRMSD, and
bond RMSE. Percentage total-loss reduction alone cannot pass this gate. R2/R4/matched smoke and
the three-system repeat are forbidden until the R1 gate passes.

### D22 — matched pooling naming is truthful

Unless the implementation is changed to scalar-gated pooling, the comparator is documented and
reported as linear two-bank pooling. Its natural parameter count is shown and it is not used to
select a ratio without that qualification.

### D23 — mandatory repair stop

The repair worker ends after the new bounded T0 review packet with `WAITING_FOR_OPERATOR_REVIEW`
and `T1 status: NOT_STARTED`. No T1 manifest, benchmark, data selection, training, evaluation,
or later architecture work is authorized by this repair phase.

### D24 — freeze the R1 reconstruction gate before the repaired run

The formal single-clip R1 gate uses `atlas_5e3e_A_R1_w000000`, T=16, FP32, frozen `torchmd_et`,
and the no-anchor centered-origin decoder.  Its predeclared thresholds are final aligned RMSD,
centroid-gauge raw RMSD, and dRMSD <= 1.5 Å, covalent bond RMSE <= 0.5 Å, at least 50% aligned
RMSD improvement over origin-only prediction, finite/converged curves, unchanged frozen encoder,
and finite nonzero gradients for every intended stem/codec parameter.  The threshold is fixed
before inspecting the repaired run and does not authorize T1.

### D25 — bound and log the formal single-clip R1 gate

The repaired R1 gate runs exactly 1,000 optimizer steps on the same one-sample clip, logs
optimizer and reconstruction metrics every 25 steps including step zero, and saves/resumes the
final checkpoint for one additional optimizer step.  The 1,000-step budget and D24 thresholds
are frozen before inspecting this repaired result; no ratio smoke or T0 repeat is implied.

### D26 — explicit authorization for bounded T1 execution (2026-09-04)

The operator explicitly authorized the T1 sequence after the repaired T0 review packet and
removed the requirement for another confirmation before launching the complete runs. The worker
must freeze an independent 64-system 48/8/8 system-level split, run four 200-step profiles, and
launch the four one-seed full controls in parallel only after all profiles pass. Ratio selection
uses validation only; the selection rule is frozen before test evaluation is opened. After T1,
additional seeds are run only for the top two controls. Idle local and `neibu` GPUs may be used
in parallel, but no existing process may be killed, preempted, or otherwise disrupted. This
authorization does not permit any later architecture phase or T1 broadening.

### D27 — token-valid T1 system eligibility (2026-09-04)

The first hash-ranked 64-system candidate set failed the pre-profile sampler audit because
some selected clips had `frames*atoms > 80000`. The frozen `max_tokens=80000` contract is not
relaxed and systems are not partitioned or dropped after selection. The corrected manifest ranks
only systems for which all 186 native R1/R2/R3 windows satisfy `frames*atoms <= 80000`, still
selects 48/8/8 from the existing source-level train/valid/test partitions, excludes the three
historical T0 systems, and records the eligibility counts and per-system maximum atom counts.
The initial invalid candidate manifest remains preserved as preflight evidence and is not used.
