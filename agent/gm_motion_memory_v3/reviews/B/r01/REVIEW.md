REVIEW_LANE: B
REVIEWED_COMMIT: 43bbadf0491c8b3815be03ae1f3ad0114484227a
VERDICT: FIX_REQUIRED

## Scope and reviewed evidence

Independent read-only static integration audit of the fixed candidate. `git rev-parse HEAD` matched the full SHA above. I read the supplied review prompt/request, root and task AGENTS, EXPERIMENTS.md, MEMORY_DESIGN.md, DATA_TRAIN_EVAL.md, WORKFLOW.md, experiment_spec.json, and B/pilot_config.json. I inspected the learned-memory implementation, split teacher boundary, raw bank loader, model/DiT/decoder integration, batch preparation, training/warm-start paths, generation entry point, pilot launcher, and targeted test source. I did not read writer implementation_status.md, SELF_REVIEW.md, or implementation conversation.

Factual evidence reviewed:

- `runs/learned_memory_v1/raw_pilot_r01/data_card.json`: SHA-256 independently checked as `235a671c01f48b367656733f0174d45aaad9a28ed774ebc311b01e85afe42a3c`, matching the pilot config. It records 192 systems in the inherited train split, three pilot systems/nine trajectories, two size-profile systems, native 10 ps, and no test/300 ps query access. Coverage is an audited prefix, not a verified full trajectory tail.
- `runs/learned_memory_v1/pilot_preflight.log`: launcher stops because B has no assigned GPU; this is not model or teacher numerical evidence.
- Coordination `resources.json`: B assignment is empty; no GPU was claimed during this review.
- No CUDA smoke, teacher parity result, pilot metrics, size profile, or completed A recipe was supplied. No Python/model/test command was executed, and there was no CPU model fallback. Parent/teacher/statistics SHA checks are implemented in the launcher, but this review did not independently deserialize or numerically compare those checkpoints.

## Blocking finding

### B-R01-01 — Gradient gate requires nonzero gradients for mathematically redundant key biases (P2)

**Locations:** `molvid/codec/learned_memory.py:290–291,319–325`; `tools/pilot_learned_memory.py:130–135`; related test `tests/test_learned_memory.py:82–84`.

**Trigger:** Any otherwise valid pilot training step for which the gradients of `history_encoder.key.bias` or `history_encoder.label_key.2.bias` evaluate to their correct zero value. The same invalid requirement exists in the standalone memory gradient test.

**Evidence:** Both linear layers use the default `bias=True`. Each bias adds the same vector to every source key of a given group. Thus either bias adds the same scalar `query dot bias` to every source logit for each query. The softmax in line 325 removes that common shift. All downstream consumers—including memory values and slot-weighted time features—depend on these logits through those softmax weights. Consequently these two bias gradients are mathematically zero. The pilot nevertheless raises whenever any parameter has `p.grad.abs().sum()==0`. Nonzero floating-point cancellation residue may sometimes satisfy the gate, but that is not evidence of a useful gradient path.

**Effect:** The bounded pilot and CUDA gradient test can reject a correctly connected memory graph. Alternatively they can pass based on roundoff in unidentifiable parameters, giving misleading gradient evidence. The current check is unsuitable as the gate for 200+ successful updates.

**Minimum fix:** Remove the two redundant additive key biases (`bias=False` for `key` and the last `label_key` linear), or explicitly exclude documented unidentifiable parameters from the nonzero requirement. Require finite gradients for participating trainable parameters and meaningful nonzero gradients/updates for the light encoder, scale/label interaction, slots, and readout paths; do not require every individual tensor to be nonzero on every permissible input. Record the architecture/config identity change normally.

**Minimum check:** On an assigned CUDA slot, run the focused memory backward check in FP32 and BF16 and one real parent-warm-start trainer step. Verify finite gradients, nonzero gradient norms on the intended paths, and actual updates to light encoder and slot parameters. Then run the bounded pilot; no whole-suite gate is needed.

## Static findings and nonblocking notes

- The B branch of `prepare_frame_joint_batch` encodes the two explicit observed anchors and the Q-only future batch separately. `prepare_memory_observation` validates `[-800,0]` ps, raw/anchor mapping and exact coordinate correspondence. Its observed origin is reused for future encoding. The long raw history never enters the teacher in this path. This is a code-level finding, not measured teacher-call or latent parity evidence.
- `RawHistoryObservation` rejects positive observed times, non-prefix masks, and invalid last-observed anchors. Future coordinates are absent from the raw observation and from `sample_learned_memory`'s interface. The existing future-mutation pilot checks remain unexecuted.
- The legacy history encoder and old M are replaced, not simultaneously activated. G/depth-four requirements are explicit. New memory parameters enter the history optimizer group; the teacher is frozen. Warm-start reports explicitly retire old history/M tensors and permit the new history tensors to initialize. Actual parent load success remains unverified.
- Vector operations mix channels only or use invariant scalar gates/weights. Chemical-token pooling and atom-to-group gathering isolate packed systems by construction. There is no Q×N×R×K attention. SO(3), atom permutation, mixed-length padding, and BF16 behavior still need CUDA evidence.
- Scale analysis uses physical times and signed, quadrature-weighted details with support checks. The level path remains valid for the three-point short/400 ps case; single-frame motion-known labels differ from repeated static history. Static-zero and all-detail-masked claims are supported structurally but are not numerical test results.
- `FrameJointModel.prepare_condition` preserves the new encoder graph, and the model reuses the prepared condition across differentiable sampled steps. The pilot intentionally disables actual-source auxiliary objectives until an A recipe is available; it cannot establish B/A loss-recipe parity.
- Raw window indexing uses common 12.8 ns past/4.8 ns future eligibility and fixed two-anchor times. The pilot crosses nine banks, three spans, three observed intervals, and three query lags over its 243 updates. It is explicitly not a paired formal exposure scheduler.
- Profiling is incomplete by design in this candidate: `profile_banks` are hashed but never executed by the pilot, and the final manifest explicitly says separate size profiling remains required (`tools/pilot_learned_memory.py:48–50,148–151`). Current hooks aggregate teacher geometry and inclusive memory time, and there is no separate backward timing. Before post-pilot approval, collect the specified median/largest-system profiles and separate observed/target teacher, graph, memory, and backward costs; do not present the current launcher as completed profiling.
- The supplied config records only a common commit and the data-card hash; the launcher does not verify the data-card file itself or record/check its executing code SHA. Bank hashes are checked. For a reviewable post-pilot submission, preserve the executing code commit, config hash, and resolved artifact identities alongside the outputs.

## Minimum remaining numerical evidence and permitted continuation

After B-R01-01 is repaired, an assigned CUDA slot may be used for the targeted memory tests and real teacher split/full parity and future-mutation checks, followed by the bounded pilot and required size profiles. Preserve teacher frame counts (two observed plus Q targets), frozen-teacher hash, trainable/optimizer ownership, nonzero light-encoder gradients, finite losses, timing, and peak memory. Test generation and prepared-condition reuse with the same fixed code/config identity.

This report is not post-pilot approval and does not authorize a scientific quality conclusion. Missing numerical evidence is an independent environment limitation, not a claim that pilot quality is inadequate. Formal M-short/M-long training still requires the completed immutable A recipe, paired exposure/RNG setup, CUDA evidence, and independent review of the repaired candidate. Data auditing and code repairs may continue without taking an unassigned GPU.
