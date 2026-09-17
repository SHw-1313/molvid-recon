# ACCEPTANCE — R2/R4 state-detail latent DiT probe v1

## A. Repository isolation

- work occurs on `feat/dit-state-detail-probe-v1` in a separate worktree;
- the audited base commit and initial worktree status are recorded;
- active T1 processes, branches, outputs, manifests, and phase files are not modified;
- all Python/test/training commands run through `enter-container` and `torch-ito`;
- no package installation, dependency upgrade, dataset download, or destructive Git operation
  occurs.

## B. Candidate and codec boundary

- only R2-SD and R4-SD are DiT candidates;
- both use the same adapter, trunk, loss, and model-size policy;
- codec/frame-encoder parameters are frozen and their hashes remain unchanged;
- generated decoding works with `raw_detail_h/v=None`;
- target coordinates, raw Haar coefficients, and coordinate-derived topology never enter DiT
  conditioning.

## C. Shape and packing

- T=16 maps to K=8 for R2 and K=4 for R4;
- state/detail are packed along channels, not expanded into separate tokens;
- scalar and vector inputs/outputs have the exact documented shapes;
- ragged `abid`, block ids, token masks, and sample boundaries are validated;
- invalid fields are zero and cannot affect valid outputs;
- detail masks intersect `detail_valid`, while state masks use valid tokens.

## D. Normalization and flow

- training-only, ratio-specific statistics are mask-aware and provenance-hashed;
- scalar normalization round-trips in FP32;
- vector normalization uses RMS scale without directional mean and round-trips in FP32;
- non-finite or near-zero scales fail before optimizer creation;
- interpolant and velocity targets match the declared rectified-flow equations for arbitrary
  tensor ranks;
- four field losses are separately normalized and then equally averaged.

## E. Equivariance and topology

- scalar outputs are rotation invariant;
- rotating data and the same sampled vector noise rotates vector predictions consistently;
- every vector linear map is bias-free and channel-only;
- no layer mixes xyz axes with a learned Cartesian matrix;
- spatial mixing respects `abid` and block boundaries;
- no atom-level all-pairs attention is present;
- topology input is static and coordinate independent.

## F. Time and observation

- flow time and physical trajectory time have separate embeddings;
- changing valid physical timestamps changes the intended time conditioning;
- replacing physical time with the same frame indices is detectable by a test;
- H=0,4,8 masks map correctly for R2 and R4;
- partial-block observation fails explicitly;
- mutating unobserved future coordinates after condition construction cannot change the clean
  observed condition;
- H>0 origins use only the observed frame-0 centroid, and H=0 uses the fixed zero origin;
- observed latent values are exact after every ODE integration step;
- no loss is accumulated on observed or invalid fields.

## G. DiT execution

- factorized block-spatial and atom-temporal paths both receive finite nonzero gradients;
- AdaLN-Zero/conditioning, input/output projections, and all four output fields receive finite
  nonzero gradients in a bounded training fixture;
- vector AdaLN is scale/gate only and does not add an invariant-conditioned vector shift;
- seeded sampling is reproducible;
- 8-step and 16-step Euler sampling produce finite masked latents and decodable trajectories;
- checkpoint save/resume preserves model, optimizer, scheduler, stats, ratio, codec, and data
  contracts.

## H. Compatibility

- existing state/detail codec and evaluator tests remain unchanged and pass;
- active T1 scripts retain their behavior and are not imported in a way that opens test data;
- old model/checkpoint behavior is not silently reinterpreted;
- configuration options that change semantics are serialized; unsupported combinations fail
  before training.

## I. Bounded CUDA smoke

For R2 and R4 separately:

- at most 100 optimizer steps and at most 15 wall-clock minutes;
- finite loss and finite/nonzero intended-module gradients;
- frozen codec and frame-encoder state hashes before/after are identical;
- checkpoint resumes for one additional step;
- generated latent without raw-detail diagnostics decodes successfully;
- trunk-only throughput, total throughput, peak allocated/reserved memory, and parameter count are
  recorded;
- the run uses an audited idle GPU and does not collide with T1 output paths.

## J. Evidence and stop

- `TASKS.md` reflects actual status rather than intended status;
- `HANDOFF.md` records changed files, commands, hashes, tests, warnings, and blockers;
- `OPERATOR_REVIEW.md` links exact source and artifact evidence;
- no real DiT pilot, full-T1-data DiT training, or T1 test access occurs;
- final status is `WAITING_FOR_T1_AND_OPERATOR_REVIEW`.

Passing code and smoke gates proves implementation readiness only. It does not prove R2 or R4 is
the better generated-latent representation.
