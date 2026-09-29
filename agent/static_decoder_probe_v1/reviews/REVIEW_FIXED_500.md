# Independent delta re-review of 500-step launch identities

Verdict: **PASS** for starting the frozen three-arm released-A1 screen at 500 dynamic updates per arm and 125 static updates per auxiliary arm.

This delta approval supersedes the 1,000-step launch identities in `REVIEW_FIXED.md`. It applies only to:

- Code commit `07b6ea76a1460eb37f85154cb3770983ad64c490`.
- Baseline config SHA256 `576bde56ebd26650a26fe4804ae9246e86ba94dbeb806dcd300ab43a536343fe`.
- Internal config SHA256 `be6fefec137aca42e65858f286353c164c6ce70a706f8b1e1adffd2350e09f39`.
- External config SHA256 `4d0815adb6c405108b9e37f62df520f43b6e4f49bb438a85539969740e29d5e7`.
- Resolved experiment SHA256 `d773b5e64467a8ce310aa106b7a6a35911653b046ae14f97b34a254d63ea8188`.
- Released-A1 parent SHA256 `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.

## Severity-ranked findings

### BLOCKER

None.

### HIGH

None.

### MEDIUM

None.

### LOW — evaluation cost estimate requires runtime confirmation

The projected one-seed evaluation cost of 1.57 GPU-hours is calculated as `4.71 / 3`, assuming cost scales linearly with the number of generated seeds and non-generation overhead is negligible. The released evaluator actually runs persistence and oracle paths once in addition to three generated paths (`/data4/users/sihao/workspace/molvid-recon-gm-motion-v3/tools/evaluate_gm_a.py:47-50`, `70-96`), so the new wall time will be somewhat greater than exactly one third of the old wall time.

This does not block the training launch because the frozen projection retains 2.03 GPU-hours of headroom and requires actual stage-by-stage accounting. Before full evaluation, the seed-override evaluator should receive a small timing preflight. Optional held-out static diagnostics must be stopped before the cap if necessary, and no arm may be truncated asymmetrically.

## Delta verification

### 1. The budget reduction is uniform and was made before outcomes — pass

All arms now specify exactly 500 dynamic updates:

- Baseline: `agent/static_decoder_probe_v1/configs/formal_baseline_500.json:208-230`.
- Internal: `agent/static_decoder_probe_v1/configs/formal_internal_md_500.json:248-270`.
- External: `agent/static_decoder_probe_v1/configs/formal_external_pdb_500.json:264-286`.

The three configs retain the same training seed, BF16 precision, H4/H8 order, component learning rates, dynamic data and normalization identities, and released-A1 model-only warm start (`formal_baseline_500.json:17-25`, `208-230`; `formal_internal_md_500.json:17-25`, `248-270`; `formal_external_pdb_500.json:17-25`, `264-286`).

A normalized comparison against the previously approved 1,000-step configs found no changes outside:

- `training.stages[].updates`;
- arm-specific output roots;
- the two auxiliary exposure targets.

No model, loss, gamma, data, pool, cadence, perturbation, optimizer, scheduler, RNG, or parent identity changed.

### 2. Paired dynamic and parent contracts remain exact — pass

The resolved 500-step experiment binds all arms to the same code commit and parent and records 500 matched dynamic updates (`agent/static_decoder_probe_v1/artifacts/resolved_experiment_500.json:2-24`, `84-95`).

Independent reconstruction produced the same dynamic contract hash for all arms:

`510938f7606011382784e0112e6608abc8c32de8e2d0e7c2e393979d6bbd26fc`.

The two auxiliary arms also produced the same revised static shared-contract hash:

`5a2a7ce9d8cef0d52511226d824d14aefec3cb377598e291a814d2681ebd14b8`.

The launch contract points to the new config paths and hashes while retaining the exact released-A1 warm-start checkpoint and SHA256 (`resolved_experiment_500.json:25-82`). Output roots move to `formal_500/`, preventing collision with the superseded 1,000-step artifacts.

### 3. Exact 4:1 cadence and 125-update exposure contracts — pass

Both auxiliary configs retain cadence 4 and now require exactly 125 static updates:

- Internal target: `agent/static_decoder_probe_v1/configs/formal_internal_md_500.json:200-211`.
- External target: `agent/static_decoder_probe_v1/configs/formal_external_pdb_500.json:200-211`.

They continue to share:

- Target: 739 +/- 3 samples.
- Target: 527,677 +/- 3,773 total atoms.
- Target: 527,677 +/- 3,773 supervised atoms.
- Six-sample and 5,000-atom update caps.
- 303–1,000 source-atom bounds.
- Static seed `20260929`.
- The same perturbation, loss coefficients, and calibrated gamma (`formal_internal_md_500.json:213-241`; `formal_external_pdb_500.json:229-257`).

I independently rebuilt both deterministic schedules from the actual frozen pools and invoked the runtime exposure validator:

- Internal: 125 updates, 737 samples, 531,450 total atoms, 531,450 supervised atoms.
- External: 125 updates, 742 samples, 523,904 total atoms, 523,904 supervised atoms.

Both pass the shared target exactly. The atom plans occupy the symmetric tolerance endpoints, and the sample plans lie within the shared three-sample tolerance.

### 4. Scientific and numerical evidence remains applicable — pass

The code, parent, data, teacher, model, loss, gamma and static-source contracts did not change. The previously reviewed CUDA evidence therefore remains applicable:

- Smoke result SHA256: `e76611870ebe1bd7c7d8db6f242ee80492a0fb36a4b7b98fb98b0ce47778ca79`.
- Gamma report SHA256: `2e09ed560c50d5938ffce1b8fe6c2080d001972fba79400154b2a9a966351478`.
- External eligibility SHA256: `971aee0b385fa50e28f35d63a3890adfe3d61734814a909510367841781e1e69`.

The runtime still binds gamma to the exact released-A1 parent/config/release, auxiliary recipe, data, normalization, statistics, teacher, model, dynamic loss, sampled loss, perturbation and pre-gamma static loss contracts (`molvid/cli/train_frame_joint.py:1150-1213`). Static cadence, independent RNG, dynamic RNG preservation, checkpoint/resume state and actual exposure checks are unchanged (`molvid/cli/train_frame_joint.py:1310-1337`, `1393-1423`, `1731-1760`, `1797-1839`, `1863-1884`).

No new CUDA rerun is required for this config-only budget reduction. The existing three-arm eight-step preflight remains representative of the unchanged training path, memory behavior, static execution and strict resume contracts.

### 5. GPU-budget arithmetic — acceptable with recorded assumptions

The frozen prospective accounting is:

- Three-arm 500-step training: `6.76 / 2 = 3.38` GPU-hours.
- Three-arm one-common-seed ordinary plus fixed-history evaluation: projected `4.71 / 3 = 1.57` GPU-hours.
- Prospective training plus dynamic evaluation: `4.95` GPU-hours.
- Already completed recorded CUDA result/preflight time: approximately `0.27` GPU-hours.
- Conservative allowance for interrupted/tests not represented in result timers: `0.25` GPU-hours.
- Held-out static decoder diagnostic allowance: `0.50` GPU-hours.
- All-in projection: `5.97` GPU-hours.
- Remaining margin under the eight-GPU-hour cap: `2.03` GPU-hours.

The old A1 cost ledger supports the 4.71-GPU-hour three-seed baseline: one arm’s ordinary and fixed-history evaluations took approximately 1.063 and 0.507 GPU-hours (`/data4/users/sihao/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/formal_cost_ledger.json:120-170`).

Actual time must be recorded after every arm and evaluation stage. The single-seed evaluator timing must be measured before full evaluation because persistence/oracle overhead does not divide by three. The reserved margin is adequate for this uncertainty, provided optional diagnostics stop first and arms remain symmetric.

### 6. Evaluation scope and statistical limitation — acceptable, separately gated

The frozen evaluation plan keeps all eight validation systems, replicas, anchors, H4/H8 views and lag buckets. Only the number of generated draws changes from seeds `[0,1,2]` to the one common seed `0`.

This preserves matched condition/noise comparisons across arms but materially reduces stochastic sampling information. Reports must state that:

- There is only one generated draw per condition.
- Between-generation-seed variance cannot be estimated.
- Replicas, anchors and systems must not be relabeled as independent generation seeds.
- Conclusions are bounded screening evidence, not final convergence or production parity.

The current released evaluator hard-codes three seeds and a five-row-per-record contract (`evaluate_gm_a.py:47-50`, `70-96`). The planned explicit seed override and corresponding protocol/row-count update are not part of training commit `07b6ea7` and are therefore not approved by this delta review. They must receive targeted review and a short cost/coverage check before evaluation. This does not alter the approved training code or checkpoint identity.

### 7. External-arm limitation remains unchanged

The explicit user waiver permits the reviewed external pool without sequence/structure near-homology filtering. Exact PDB-chain overlap, representation compatibility, quality gates, store/data hashes and sealed-test handling remain unchanged. External findings must continue to be described as additional PDB geometry supervision, not a homology-controlled novelty experiment.

## Verification record

- Confirmed HEAD remains exactly `07b6ea76a1460eb37f85154cb3770983ad64c490`.
- Independently recomputed all three new config hashes and the new resolved-experiment hash.
- Schema-validated all three configs using the candidate’s runtime validator.
- Independently recomputed the three equal dynamic-contract hashes and both equal static shared-contract hashes.
- Independently reconstructed and validated both complete 125-update exposure plans from the actual frozen pools.
- Confirmed normalized 500-versus-1,000 config differences are limited to budget, output roots and exposure targets.
- Reverified the unchanged smoke, gamma and external-eligibility hashes.
- Performed no code or artifact edits, started no training, and opened no sealed test trajectories.

## Final decision

**PASS** for launching the exact frozen 500-dynamic-update three-arm training screen.

This PASS does not authorize the future evaluator modification without its targeted review. Before launch, rehash HEAD, all three configs, the resolved 500-step experiment, parent checkpoint, gamma report and external eligibility manifest. Any identity change invalidates this verdict.
