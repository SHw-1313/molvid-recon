# Independent static-response experiment

## Decision and scope

User authorized independent implementation and experiments from the ABCD
integration, with critical selection rather than blind implementation of the
September 30 HTML plan. This worktree branches directly from `946917b`:
`exp/static-response-abcd-20261004`. Only the static paths from `2f24ed4`
were transplanted. No Mamba code, history switch, C8 expansion or external
PDB corpus is included. Original worktrees, baseline models and live C8
remain untouched. A1/4-block/legacy G+M is the common scientific parent.

Selected: **S, static shared response**, because it directly addresses the
user's unmet requirement and permits a narrow intervention. Deferred:

- **U, future-feature mean/variance memory**: conditional moments are not
  themselves a calibrated multimodal forecast. It risks overlapping the
  existing stochastic flow and introducing an auxiliary representation
  problem before the generative interface is understood. Not implemented.
- **K, finite-time occupancy**: meaningful only with an independently
  validated real-MD predictor and adequate transition coverage. No learned
  self-evaluator or self-generated supervision is added. Not implemented.
- External static structures, angle/chirality/clash additions and a new
  task/noise-level adapter are deferred to avoid changing multiple factors.
  Initial static geometry is coordinate MSE plus clean-target bond distance;
  it is not a complete chemical-validity objective.

## Three arms and gradient boundaries

| Arm | Dynamic objective | Static forward | Static gradient recipients |
|---|---|---|---|
| S0 | A1 unchanged | none | none |
| Sdec | A1 unchanged | identical to Strunk | shared decoder only |
| Strunk | A1 unchanged | RF + endpoint + actual-source reconstruction | shared spatial DiT/G/FFN/adapters and decoder |

Sdec is **not the old D checkpoint or old decoder-only task**. It runs exactly
the same static RF and geometry computations as Strunk, with autograd inputs
restricted to decoder parameters. The RF term is constant with respect to
those parameters; it is logged, not claimed as decoder RF training. Total
training losses across arms are not a quality ranking.

The teacher is frozen. Static conditions are a corrupted single structure;
clean coordinates/teacher features are target-only. Physical-time embeddings,
history/M, future temporal attention and decoder temporal layers are bypassed.
There are no repeated static frames or fabricated zero-motion labels. The
static flow-progress embedding is shared; flow progress is not physical time.

Each update performs dynamic backward then static backward, followed by one
joint clip, AdamW step and scheduler advance. Sdec static backward preserves
already accumulated dynamic trunk gradients exactly. Joint clipping can
still indirectly rescale the dynamic update, so it is logged explicitly;
no claim that the full Sdec trunk trajectory is unchanged from S0 is made.
Both static RNG streams and the gradient-scope contract are checkpointed.

## Frozen first pilot

- Same SHA-verified A1 checkpoint `52d922f8…aa14a9`, fresh optimizers and A1
  learning rates, BF16, 512 dynamic updates **per arm**, seed 20261004.
- One real clip/update, balanced train-only ordinary/fixed-history lag
  buckets and a deterministic atom-size stride; ordinary H4/H8, fixed H4.
  This is a bounded continuation pilot, not a full effective epoch.
- Sdec and Strunk use the same independent internal-MD static schedule,
  cycling valid single frames; corruption sigmas .1/.3/.8 Å in 1:2:1 ratio.
  Static RF and endpoint reconstruction every step; actual-source Euler4
  reconstruction every eight successful updates.
- Sixteen **training** calibration pairs. One common frozen auxiliary weight:
  `min_g 0.1 * median(||dynamic_g||) / median(||static_g||)` for shared DiT
  and shared decoder. This fixes the former global-norm masking problem but
  is not a hard per-batch cap. Per-pair ratios and gradient cosines are saved;
  conflicts are not projected away. No validation-based lambda search.
- Equal dynamic exposure and optimizer updates, **not equal FLOPs**. Static
  arms also differ in backward cost; wall time and peak allocation are logged.
- One final checkpoint/arm, strict reload before evaluation. No periodic
  checkpoint accumulation, environment upgrades or sealed-test access.
- Reserve >=8 GiB on launch and >=4 GiB before each arm; stop without deleting
  any files if the reserve is lost.

## Validation and interpretation

The frozen bounded validation selection is eight systems, R1/anchor0/H4,
100/400 ps grids, FP32 Euler16, generated seeds 0/1/2. Each arm produces
48 generated rows plus 32 oracle/persistence rows. This does not replace
the full formal ordinary/fixed-history, multi-anchor or long-rollout suite.

Primary readouts: bond RMSE, vdW clashes/1000 atoms, vdW severity, residue
RMSF MAE and MSD-curve MAE. Paired deltas are averaged within each system
before a 2,000-resample system bootstrap. CIs do not estimate training-seed
variation; unavailable measurements are not silently converted to zero.

Static validation covers each system at .1/.3/.8 Å, actual-source Euler4.
Compare sampled trunk output with identity/noisy input, source without trunk
and observed latent only. The latter two are **OOD sensitivity**, not evidence
from separately trained ablations. If the output cannot beat simply retaining
the noisy input, do not call the task successful denoising. If removing the
trunk does not degrade results, investigate decoder shortcuts before scaling.

No automatic default-model promotion. A candidate needs independently
interpretable static behavior and generated dynamic gains over both S0 and
Sdec without an unresolved geometry/motion tradeoff; follow with additional
training seeds and full validation before combining with Mamba.

## Checks and artifacts

- 15 targeted tests passed on CUDA: static shared/temporal separation,
  FP32/BF16 restricted gradients, equal static forwards, exact joint-update
  accumulation and resume for both scopes, full/streamed sampled backward,
  calibration boundaries, legacy static tests and paired evaluation reduction.
- Real A1 sample `atlas_2ejn_B_R1_a000000_dt_100ps_H4`, FP32 Euler16 seed0:
  generated coordinates are byte-identical to the prior verified integration
  reference (`max_abs_A=0`).
- Real evaluator smoke: one validation system, three dynamic rows and twelve
  static diagnostic rows completed. This checks execution, not model quality.
- Evidence: `runs/static_response_20261004/cuda_tests.xml`,
  `a1_parity.json`, `evaluator_smoke/{protocol,summary,dynamic_rows,static_rows}.json`.

Run from the prescribed container / `torch-ito`, isolated GPU 2:

```bash
python -u -m tools.run_static_response \
  --output runs/static_response_20261004/pilot512 --updates 512
```

This first source checkpoint is prepared for launch. Runtime status must be
read from `pilot512.log`, `pilot512/protocol.json`, `calibration.json`, per-arm
`steps.json`/`summary.json`, and final `comparison.json`; do not infer completion
from this design document.
