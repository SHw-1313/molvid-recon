# DECISIONS — R2/R4 state-detail latent DiT probe v1

Prepared decisions are binding for this implementation. Append a decision only when an observed
repository fact makes a real change necessary. Do not rewrite prior entries after seeing results.

## D001 — two candidates only

The DiT code path supports `ratio2_state_detail` and `ratio4_state_detail`. R4-SD is the
provisional scalable candidate and R2-SD is the safety candidate. R1-SD remains an external codec
upper bound; R4 matched pooling is excluded.

## D002 — separate branch and worktree

Development occurs on `feat/dit-state-detail-probe-v1` in a new worktree based on audited commit
`23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541`. The active T1 checkout, processes, outputs, and
phase records are read-only.

## D003 — codec is frozen

The frame encoder, coordinate stem, temporal codec, and coordinate decoder are frozen. Generated
latents use the existing public no-target-coordinate decode path. No joint fine-tuning is allowed
in v1.

## D004 — generated latent excludes raw coefficients

The DiT generates `state_h`, `state_v`, `detail_h`, and `detail_v`. `raw_detail_h` and
`raw_detail_v` are codec diagnostics and are neither DiT inputs nor generated outputs.

## D005 — state/detail share one sequence token

State and detail banks are concatenated on the channel axis and projected into one token at each
`(latent time, atom)` location. They are not represented as two sequence tokens. Consequently R2
has K=8 and R4 has K=4 for T=16.

## D006 — equivariant scalar/vector trunk

The DiT maintains scalar `[K,N,Dh]` and vector `[K,N,3,Dv]` streams. Scalar attention weights may
mix vector values, while learned vector maps act only on channels and are bias-free. No learned
matrix mixes xyz axes.

## D007 — factorized interaction

Each DiT block uses block/residue spatial interaction plus per-atom temporal attention. Dense
attention over all atom-time tokens is prohibited. The initial block-level dense backend must be
explicitly versioned and replaceable.

## D008 — bidirectional denoising attention

Latent temporal attention is bidirectional. Causality comes from hiding unknown clean latents and
clamping observations, not from the deterministic codec's prefix-causal attention mask.

## D009 — rectified flow

V1 uses normalized latent rectified flow with Gaussian source, linear interpolation, and velocity
prediction. The loss is the equal-weight average of four independently mask-normalized latent
field losses. Coordinate auxiliary losses and alternative diffusion parameterizations are out of
scope.

## D010 — field-aware normalization

Scalar fields use per-channel training means and standard deviations. Vector fields use
per-channel RMS scales and no directional mean. Statistics are ratio-specific, mask-aware,
train-only, hashed, and immutable for a run.

## D011 — block-aligned observation in v1

The same 16-frame codec is used for all tasks. V1 supports prefix lengths H=0,4,8 so observations
end on a boundary valid for both R2 and R4. Partial blocks fail explicitly. Two-frame history is a
later observation-conditioning experiment, not a codec change.

## D012 — physical clocks, not frame indices

The DiT receives actual block timestamps and time spans. Ratio id is an additional condition and
does not replace physical time.

## D013 — static topology only

The generative model may condition on static N-axis atom/block/component identity and covalent
topology. It must not receive target-derived radius graphs, distances, edge vectors, contacts, or
coordinates as metadata.

## D014 — exact observation clamping

Observed normalized state/detail fields remain clean and are restored after every ODE step.
Unobserved valid fields alone receive noise and flow loss. Invalid elements remain zero.

## D014a — target-free origin and detail validity

For H>0, `sample_origin` comes from the clean observed frame-0 centroid; for H=0 it is a fixed zero
vector. It never uses target-only future coordinates. Detail losses and noise intersect the codec's
`detail_valid` field in addition to token validity.

## D015 — bounded current authorization

This run may implement, test, and execute at most one 100-step/15-minute real-CUDA smoke per
candidate on idle GPUs. It may not start a real DiT pilot, open T1 test, or use the complete T1
training split.

## D016 — pilot is a later decision

Scientific R2/R4 comparison waits for approved T1 codec checkpoints and training-only latent
statistics. Its provisional R4-default quality guardrails are recorded in PLAN before results,
but the pilot remains unauthorized in this phase.

## D017 — smoke reads immutable T0 payload in the audited source worktree

The dedicated target worktree contains the T0 clip-store manifest and index but not the
historical data.bin payload. The bounded smoke therefore reads the existing T0-valid payload
from /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid
through the target code, without copying or modifying the source worktree. The smoke reports this
provenance explicitly; its randomly initialized codec and T0-derived statistics are not pilot
inputs.

## D018 — review-fix contracts remain local to the existing DiT path

The 2026-09-07 review repair keeps the factorized DiT architecture and corrects three contracts
in place: vector normalization and FFN operations are SO(3)-equivariant under nonzero residual
gates; observed origins reuse the frozen codec's loss-masked frame-0 centroid; and conditional
evaluation slices every future and temporal metric by the explicit history prefix. BF16 autocast
is real CUDA-only execution policy, and all six latent-statistic tensors plus provenance are
serialized with the checkpoint. These repairs do not authorize T1 data, a production pilot, or
any later architecture work.
