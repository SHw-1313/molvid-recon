# Independent re-review of candidate `07b6ea7`

Verdict: **PASS** for starting the frozen bounded released-A1 paired screen.

This approval applies only to:

- Code commit `07b6ea76a1460eb37f85154cb3770983ad64c490`.
- Baseline config SHA256 `b6370b09aa70e390bc9d56b3459ab360eb3900e3868aeaf0735169bbbcb38ee4`.
- Internal config SHA256 `2ab78281a306f18c2fa6041c46a789e89b48c4b699818a677d1e7a43d2c1a4ae`.
- External config SHA256 `ef990ae8f06fc1f8cc8cabf36897a5441eee4dcf1ca6e4eda99f345d87441148`.
- Resolved experiment SHA256 `49bfdecd6162e9e752b0e7d106afbb48bc6d795ced0733968b47d9c091cc6718`.
- Released-A1 parent SHA256 `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.

The baseline/internal comparison is ready. The external arm is also technically eligible under the user’s explicit near-homology waiver and the exact hashed manifest reviewed here.

## Severity-ranked findings

### BLOCKER

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None affecting launch correctness.

## Non-blocking scientific and operational limitations

1. **External homology limitation.** The external selector excludes exact PDB-chain identity overlap with released-A train, validation, and sealed-test manifest identities, but it does not perform sequence- or structure-level near-homology filtering. This is accurately recorded as an explicit user waiver, not misrepresented as an exclusion (`agent/static_decoder_probe_v1/artifacts/external_pdb_3000_eligibility.json:27-38`; `agent/static_decoder_probe_v1/artifacts/data_card.json:51-66`). Consequently, the external arm may test added PDB geometry supervision, but its result must not be described as a homology-controlled measure of structural novelty or held-out generalization.

2. **GPU-hour margin.** Linear extrapolation of the three eight-step preflights is approximately 6.8 GPU-hours for training before evaluation. This fits the nominal eight-hour ceiling narrowly. Actual cumulative GPU-hours should remain recorded, and the frozen step-500 checkpoints should be used uniformly if observed throughput shows that the complete training-plus-evaluation plan will exceed the authorized cap. This is an operational constraint, not a code blocker.

## Review of prior blocking findings

### 1. Released-A1 orchestration and symmetric initialization — closed

All arms bind the same immutable A1 parent, use model-only warm start, and reset optimizer, scheduler, cursor, counters, and RNG rather than inheriting training state (`agent/static_decoder_probe_v1/artifacts/resolved_experiment.json:22-95`; `molvid/cli/train_frame_joint.py:883-908`; `molvid/training/joint.py:1150-1248`). The CLI requires parent and resume SHA256 identities and rejects continuation semantics for this paired experiment (`molvid/cli/train_frame_joint.py:883-908`, `1120-1168`).

The freezer verifies that the three configs have the same dynamic contract and parent and that both auxiliary arms share the same static objective, RNG, and exposure contract (`tools/freeze_static_decoder_experiment.py:60-121`). Its recorded launch arguments bind the exact parent and config hashes (`tools/freeze_static_decoder_experiment.py:123-168`; `agent/static_decoder_probe_v1/artifacts/resolved_experiment.json:25-82`).

### 2. Exact 4:1 cadence, dynamic isolation, and resume — closed

Static work is triggered after every fourth successful dynamic update (`molvid/cli/train_frame_joint.py:1731-1760`). Static scheduling has an independent seeded order, six-sample/5,000-atom batch caps, commit-on-success semantics, and resumable epoch/cursor/exposure state (`molvid/training/static_schedule.py:160-318`).

The implementation snapshots and checks global CPU and CUDA RNG states around every static update (`molvid/cli/train_frame_joint.py:1744-1759`). Static schedule and RNG states are included in rank-local checkpoints and restored on strict resume (`molvid/cli/train_frame_joint.py:1393-1423`, `1797-1839`). Exact-resume loading also checks the full training contracts and restores model, optimizer, dynamic RNG, step, and successful-update count (`molvid/training/joint.py:1039-1148`).

The final preflights have identical dynamic sample IDs, histories, flow-time draws, scheduled auxiliary flags, learning rates, and execution hash `abbae4b784aaef17578e9e36fd423fc5bf563f769acc387c2b040a3735849bbf` across all arms (`runs/static_decoder_probe_v1/preflight_07b6ea7/baseline/train_metrics.jsonl:1-8`; `internal_resume/train_metrics.jsonl:1-8`; `external_pdb/train_metrics.jsonl:1-8`). The SHA-bound internal step-4 resume reaches step 8 with the expected second static update and cumulative exposure.

### 3. Gamma calibration denominator and traceability — closed

Gamma calibration computes the decoder gradient of the complete active dynamic objective and records per-term decoder-gradient norms, raw/weighted values, and zero-gradient terms (`tools/run_static_decoder_pilot.py:225-299`). Its schedule activates the actual-source sampled branch using the released recipe, including the sampled clash contribution (`tools/run_static_decoder_pilot.py:359-414`).

The final report is bound to the exact code, A1 parent/config/release/auxiliary recipe, data, normalization, statistics, teacher, model, dynamic loss, sampled loss, perturbation, and pre-gamma static loss contracts. Runtime rejects any mismatch (`molvid/cli/train_frame_joint.py:1150-1213`).

The independently verified gamma artifact SHA256 is `2e09ed560c50d5938ffce1b8fe6c2080d001972fba79400154b2a9a966351478`. It records:

- Nine train batches.
- One scheduled actual-source sampled-clash batch.
- Dynamic decoder-gradient median `0.1572979837656021`.
- Pre-gamma static median `68.0927505493164`.
- Gamma `0.00017325410836320802`.
- Achieved median ratio `0.075`.
- Static clean probability `0.20`.

These identities and values appear at `runs/static_decoder_probe_v1/a1_smoke_final_07b6ea7/gamma_calibration.json:2-12`, `683-686`, and `881-1081`.

### 4. Strict T=1 and clean-target isolation — closed

Static input validation requires exactly one valid frame, coordinate-origin time zero, and no physical time interval (`molvid/training/static_decoder.py:104-117`). Noise is applied to coordinates and coordinate-derived block positions before the frozen teacher is called; the decoder’s observed context therefore derives from the perturbed structure. Clean coordinates remain separate as the reconstruction target (`molvid/training/static_decoder.py:176-238`).

The objective is sample-equal coordinate MSE plus clean-target bond loss and the released-A physical clash rule, followed by the single calibrated gamma (`molvid/training/static_decoder.py:323-405`). CUDA evidence records zero prediction change under target mutation and zero disabled-path difference in FP32 and BF16 (`runs/static_decoder_probe_v1/a1_smoke_final_07b6ea7/result.json:108-158`, `227-277`).

### 5. Decoder-only AdamW behavior — closed

The static trainer reuses the dynamic AdamW so decoder moments and configured decoder weight decay continue naturally. It clears gradients, backpropagates only through the decoder, checks that no inactive parameter received a gradient, performs the optimizer step, and then verifies inactive weights and all pre-existing inactive optimizer slots remained unchanged (`molvid/training/static_decoder.py:462-605`). It never advances the dynamic scheduler (`molvid/training/static_decoder.py:607-631`).

The CUDA test primes AdamW moments for every dynamic component before executing a static step and checks inactive weights and state tensors bitwise (`tests/test_static_decoder.py:220-263`). This closes the earlier fresh-optimizer-only evidence gap.

### 6. Pool identity and matched exposure — closed

The internal pool uses one midpoint frame per eligible A-train system/replica and requires complete R1/R2/R3 strata (`molvid/training/static_schedule.py:35-83`). The external pool requires 2,000–5,000 unique dataset indices and materializes only the frozen selection (`molvid/training/static_schedule.py:86-157`).

Both configs freeze cadence 4, six samples and 5,000 atoms per update, source sizes 303–1,000 atoms, identical perturbation/loss/gamma, and a common 250-update exposure target (`agent/static_decoder_probe_v1/configs/formal_internal_md_1000.json:200-241`; `formal_external_pdb_1000.json:200-257`). Runtime validates planned and completed samples, total atoms, and supervised atoms (`molvid/training/static_schedule.py:238-264`, `321-347`; `molvid/cli/train_frame_joint.py:1310-1337`, `1863-1884`).

I independently reconstructed the deterministic 250-update plans:

- Internal: 186-entry pool, 1,468 samples, 1,057,698 total/supervised atoms.
- External: 3,000-entry pool, 1,482 samples, 1,054,901 total/supervised atoms.
- Shared target: 1,475 ± 7 samples and 1,056,299 ± 1,399 total/supervised atoms.

Both plans pass the frozen gate exactly.

### 7. External eligibility and fail-closed binding — closed under waiver

The selector reads only sealed-test identity metadata, not trajectory payloads, and removes exact PDB-chain overlap across train, validation, and sealed-test identities (`tools/select_external_static.py:87-123`, `133-180`). It deterministically selects 3,000 records subject to finite/nondegenerate coordinates, complete supervision, supported atom types, atom bounds, and nonempty topology (`tools/select_external_static.py:41-69`, `124-247`).

Runtime binds and verifies the store index and `data.bin` SHA256, selection-report SHA256, full eligibility-manifest SHA256, source/split/store, selection count/hash/seed, code/data/normalization identities, atom bounds, supervision policy, direct-overlap declaration, and exact waiver identity (`molvid/cli/train_frame_joint.py:1222-1295`).

The manifest SHA256 is `971aee0b385fa50e28f35d63a3890adfe3d61734814a909510367841781e1e69`. It contains 3,000 unique selected indices, atom range 303–998, full loss-mask supervision, 20 rejected exact-overlap candidates, and `sealed_test_trajectories_opened=false` (`agent/static_decoder_probe_v1/artifacts/external_pdb_3000_eligibility.json:26-70`). I additionally audited all selected records for tensor shapes, supported atom/block IDs, and valid bond indices; no incompatibilities were found.

### 8. CUDA memory and formal-path evidence — sufficient

The factual A1 smoke result SHA256 is `e76611870ebe1bd7c7d8db6f242ee80492a0fb36a4b7b98fb98b0ce47778ca79`. Warm start loaded all 481 tensors and initialized none. FP32/BF16 target mutation, disabled-path, inactive-parameter, and teacher-state deltas are zero. The 4,975-atom static profile peaked at 802,621,440 allocated bytes, and phase cache release is explicitly recorded (`runs/static_decoder_probe_v1/a1_smoke_final_07b6ea7/result.json:26-55`, `90-343`, `430-434`).

All three eight-step formal-path preflights completed. Peak allocated memory was approximately 48.0 GB for every arm, including A’s expensive sampled updates. The external arm executed two real PDB static updates, while the internal run demonstrated SHA-bound step-4 to step-8 resume.

## Verification record

- Confirmed HEAD is exactly `07b6ea76a1460eb37f85154cb3770983ad64c490`.
- Independently recomputed all supplied smoke, gamma, eligibility, config, and resolved-experiment SHA256 values.
- Independently reconstructed both complete 250-update exposure plans.
- Independently audited the 3,000 selected external records for representation compatibility.
- Independently reran the final schedule/common-contract subset: 10 tests passed.
- Reviewed the supplied final verification record of 29 focused CPU tests and four CUDA tests passing.
- Performed no source/model edits and started no formal training.
- Opened no sealed test trajectory payloads.

## Final launch decision

**PASS.**

The exact frozen baseline/internal paired screen may start. The external arm may also start using the exact reviewed eligibility manifest under the explicit near-homology waiver. Rehash all listed configs, manifests, evidence, parent checkpoint, and HEAD immediately before launch; any identity change invalidates this approval and requires re-review.

This approval is for a bounded 1,000-dynamic-update screen, not production parity or final scientific convergence. External-arm conclusions must retain the near-homology limitation and must not claim a homology-controlled diversity benefit.
