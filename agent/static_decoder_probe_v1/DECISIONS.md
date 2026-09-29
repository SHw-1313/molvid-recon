# Static decoder probe v1 decisions

## 2026-09-29

- Use released A1 as the only formal parent. All three configs enforce the same
  model-only warm start and optimizer/scheduler/cursor/RNG reset; exact resumes
  require a checkpoint SHA256 and must descend from that parent.
- Keep a dedicated `T=1` path. Static time zero is a coordinate origin, not a
  physical transition or motion label. Clean coordinates exist only as targets;
  all coordinate-derived conditions are rebuilt from the perturbed input.
- Use coordinate MSE + 0.1 clean-target bond loss + A1's released clash rule,
  multiplied by the one shared train-only gamma. Gamma uses the complete A1
  decoder-gradient objective, including actual-source sampled clash, and the
  same 20% clean mixture as training.
- Reuse the dynamic AdamW so decoder moments continue naturally. Every static
  step fails closed if inactive parameters or pre-existing inactive optimizer
  state change. Static updates do not advance dynamic LR, cursor, step, or RNG.
- Match actual supervision, not nominal record size. Both static arms use
  303--1,000-atom structures, at most six samples/5,000 atoms per update, and a
  common gate on samples, total atoms, and `loss_mask`-supervised atoms.
- Internal pool: one midpoint frame from the first indexed clip for each of 62
  eligible A train systems and all R1/R2/R3 replicas (186 trajectories).
- External pool: deterministically select 3,000 fully supervised compatible
  records from the existing processed PDB train store derived from
  `/data5/PVB/pdb`; reject fragments, invalid topology/coordinates/types, and
  exact PDB-chain overlaps with A train/valid/sealed-test manifest IDs.
- The user explicitly waived strict near-homology filtering for this bounded
  version on 2026-09-29. Preserve that as a limitation; do not misstate exact
  overlap exclusion as sequence/structure homology filtering.
- The profiled 1,000-step plan plus full evaluation exceeds the eight-GPU-hour
  task cap. Before observing any comparison outcome, uniformly reduce all arms
  to 500 dynamic / 125 static updates, keep cadence 4 and every other contract,
  and re-freeze/re-review the new identities. Preserve full 8-system/lag/view
  evaluation coverage with one common generation seed and report that reduced
  sampling uncertainty.

## 2026-09-30

- Accept the independent delta PASS for the exact 500-step identities, then run
  the three arms sequentially on GPU1. Preserve the initial baseline OOM rather
  than stopping either pre-existing GPU1 process; restart the unchanged arm with
  CUDA expandable segments only after the other use subsides.
- Use one common generated seed (`0`) while retaining persistence and oracle.
  Keep all eight systems, replicas, anchors, histories and lags. Treat system
  bootstrap as paired system uncertainty, never as generation-seed variance.
- Require a separately reviewed fail-closed held-out static evaluator. Freeze
  its validation manifest/store/model/recipe/config hashes, 8 x 3 record grid,
  sigma grid and record-local noise identity before reading checkpoint metrics.
- Preserve the failed baseline static preflight caused by legacy records without
  explicit `frame_mask`. Normalize through the public record validator rather
  than weakening the frozen protocol; obtain an independent targeted PASS before
  rerunning.
- Interpret external PDB as a positive static decoder-reconstruction result but
  a negative/inconclusive dynamic-transfer result. Do not promote A3 or claim
  net denoising, homology-controlled generalization, convergence or production
  parity. Keep `keep_release_trigger=false`.
- Do not scale the decoder-only external recipe. A future external-data test, if
  authorized, should target the shared latent/DiT path, add homology control and
  multiple generation seeds. Internal MD is the more credible narrow candidate
  for clash reduction, with bond and motion constraints retained.
