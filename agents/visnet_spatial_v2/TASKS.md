# TASKS — ViSNet spatial backbone v2

Status legend: `[ ]` pending, `[-]` active, `[x]` complete, `[!]` blocked.

## Preparation and audit

- [x] V200 inspect worktree, record HEAD/status/environment, and create or enter `feat/visnet-spatial-v2` without discarding user changes — source HEAD was `bca9ea98df643f4965b009688f2967bd669e7fe2`; target branch was absent and was created with untracked user files preserved.
- [x] V201 update root `AGENTS.md` for the v2 phase and limited pinned-reference network exception — root instructions now name v2 and the two exact Git-only network exceptions.
- [x] V202 clone/fetch AI2BMD and PyG references only under `/data4/users/sihao/workspace`, checkout exact commits, and record hashes/licenses — both detached checkouts are clean at the requested commits with MIT licenses recorded.
- [x] V203 review v1 implementation against both references and fill `REFERENCE_PARITY.md` local mappings before substantive code changes — all requested discrepancy diagnoses are recorded with primary/cross-check symbols and planned tests.
- [x] V204 record the exact v1 TorchMD/ViSNet result and manifest/protocol hashes that may be reused — v1 tiny results and artifact hashes are recorded in the v2 handoff.

## Reference-faithful primitives

- [x] V210 implement/carry faithful cosine cutoff and ExpNormal/Gaussian RBF operators — deterministic primary-reference fixtures pass in FP32.
- [x] V211 implement faithful Sphere for lmax-1 and lmax-2 with self-loop-safe direction handling — lmax shapes and pinned values pass.
- [x] V212 implement faithful VecLayerNorm modes and trainable-vector-norm option — none/rms/max_min and trainable semantics are tested.
- [x] V213 implement faithful NeighborEmbedding and EdgeEmbedding — node-conditioned edge fixtures pass.
- [x] V214 add deterministic pinned-reference fixtures and primitive parity tests — primitive, base, edge, node, and full-block fixtures are recorded in REFERENCE_PARITY.md.

## ViS-MP geometry

- [x] V220 implement faithful base ViS-MP attention/message/aggregate/update without incoming-edge softmax — direct pinned base-message/full-block fixtures pass.
- [x] V221 implement vector rejection and persistent non-final edge updates — edge evolution and rejection tests pass.
- [x] V222 implement `vertex_type=edge` — direct primary-reference fixture passes.
- [x] V223 implement `vertex_type=node` and `vertex_type=none` — node fixture and none-path tests pass.
- [x] V224 implement final-layer behavior plus scalar/vector output normalization — final no-update and output norm tests pass.
- [x] V225 add one-layer and full-block reference parity tests for lmax-1/lmax-2 — full deterministic output hashes are recorded.
- [x] V226 add angle, dihedral, edge-evolution, final-layer, and geometry-gradient discriminating tests — explicit fixed-distance angle/dihedral tests pass.

## Project adapters and versioning

- [x] V230 add `visnet_v2_radius` and prove exactly one native radius build — native adapter call-count test passes.
- [x] V231 add `visnet_v2_bonded` using the existing external graph without rebuilding it — external no-rebuild test passes.
- [x] V232 isolate atom block-type and binary bond extensions; zeroed extensions recover the reference path — extension isolation/zero-bond tests pass.
- [x] V233 implement internal `v_full` and final lmax-2-to-public-l1 adapter — `[M,8,C]` internal and `[M,3,C]` public shape tests pass.
- [x] V234 add real config semantics for `vertex_type`, `rbf_type`, `vecnorm_type`, and `trainable_vecnorm` — option behavior tests pass.
- [x] V235 version/extend model contracts and checkpoints while preserving all v1 loading — v1/v2 contract tests pass.
- [x] V236 correct source attribution and license notices — v2 provenance points to the two pinned MIT references; v1 header is corrected without changing its path.

## Correctness and regression gates

- [x] V240 pass lmax-1/lmax-2 shape, T=1/T=16, SE(3), frame/sample isolation, and finite-gradient tests — Gate 0 specialized and full suite pass.
- [x] V241 prove every accepted geometry option changes real computation and no option is metadata-only — vertex, RBF, VecLayerNorm, rejection, and edge-update discriminators pass.
- [x] V242 pass v1 output/checkpoint regression and unchanged TorchMD regression — selected v1 regression suite and micro baselines pass; v1 artifacts remain untouched.
- [x] V243 pass the full relevant test suite, py_compile, and `git diff --check` — 89 tests passed with one existing warning.
- [x] V244 produce implementation/parity evidence JSON with reference and fixture hashes — IMPLEMENTATION_EVIDENCE.json records exact commits, licenses, fixtures, tolerances, and commands.

## 500-step micro comparison

- [x] V250 freeze the micro protocol before seeing v2 outcomes — protocol fixes seed 20260902, three clips, FP32, 500 updates, max_tokens=80000, and checkpoint/resume.
- [x] V251 run `torchmd_et` or verify a strictly reusable stored micro baseline — local CUDA run completed.
- [x] V252 run/verify legacy `visnet_bonded` v1 — local CUDA run completed; legacy geometry-module gate is not applicable.
- [x] V253 run `visnet_v2_bonded`, lmax-1, vertex-edge — local CUDA run completed.
- [x] V254 run `visnet_v2_bonded`, lmax-2, vertex-edge — neibu CUDA run completed with logical clip-wise accumulation after the initial packed-batch OOM.
- [x] V255 verify finite training, geometry-module gradients, >=30% loss reduction, checkpoint save/resume, runtime, and memory — all v2 runs pass these gates; exact values are in the micro result JSONs.
- [x] V256 select the final v2 geometry config from frozen absolute metrics and record the decision — lmax=1 selected by lower final loss (0.330271 vs 0.335912), lower runtime, and lower peak memory.

## Exact tiny comparison

- [x] V260 verify immutable 441/117 manifest, split, normalization, evaluator, and protocol hashes — source/index hashes and all semantic rows match; only path-dependent built-manifest hash differs.
- [x] V261 decide whether stored TorchMD/v1 results are strictly reusable; rerun only if comparability fails — semantic/protocol comparability passed, so immutable v1 rows were reused without modification.
- [x] V262 run the selected `visnet_v2_bonded` config for the locked 30-epoch tiny protocol — neibu worker completed 30 epochs / 4979 steps in FP32.
- [x] V263 verify exact train coverage, complete holdout evaluation, finite metrics, and checkpoint resume — 441/117 coverage, 30 holdout rows, finite result, and step 4979→4980 resume passed.
- [x] V264 run `visnet_v2_radius` micro/performance comparison; run its full tiny experiment only if scientifically needed — 500-step native-radius micro passed; full tiny radius run was not scientifically required.

## Reporting and stop

- [x] V270 generate absolute-metric and gap-recovery JSON/CSV/Markdown/plots — final tiny report includes six eval curves, runtime annotations, performance summary, absolute metrics, and 95.78% dRMSD gap recovery.
- [x] V271 classify the result using the six allowed v2 outcome labels — `IMPLEMENTED`: parity passed, tiny quality gate failed on the 60% train-loss reduction threshold.
- [x] V272 append complete commands, evidence, hashes, changed files, limitations, and blockers to `HANDOFF.md` — final dated handoff entry appended below.
- [x] V273 stop without beginning AF3, state-detail, trunk, rollout, scaling, or full-data work — this phase ends here.
