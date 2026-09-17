# Luna worker prompt: repair the graph-runtime review findings

Act as the sole implementation worker for the scientific machine-learning repository at `/workspace/PVB` (host checkout: `/data4/users/sihao/workspace/PVB`). Work from branch `fix/graph-runtime-v1`, whose reviewed HEAD is `9ad0b7c99718839a9a20008bbb04152646c119e0`; the review base was `6459d3c0e823491b09295930bf6d5929a8f9961b`.

Your goal is to fix the concrete runtime, scientific-validity, checkpoint, acceptance, synchronization, and repository-artifact problems listed below. Do not redesign the codec or change its scientific objective.

## Mandatory operating rules

1. Read `agents/AGENTS.md`, `agents/PLAN.md`, `agents/TASKS.md`, `agents/DECISIONS.md`, and `agents/HANDOFF.md` before editing.
2. Record the initial `git status --short`, `git rev-parse HEAD`, Python, PyTorch, CUDA, and GPU visibility.
3. Treat unrelated changes as user-owned. Do not reset, discard, rewrite, or overwrite them.
4. Use `enter-container`, then activate `/home/sihao/miniforge3/bin/activate torch-ito`, for all Python, test, training, profiling, and evaluation commands.
5. Do not install packages or access the network.
6. Keep production geometry and loss reductions FP32 where currently required. Preserve coordinate autograd and the existing SE(3) rules.
7. Do not add CPU/dense fallback behavior to the production graph path.
8. Do not change legacy dyVAE/PRETRAIN/MD/ADJ/DPO/inference behavior.
9. Do not claim convergence or scientific quality from the bounded acceptance runs.
10. Do not perform destructive Git history rewriting or delete user artifacts without explicit operator approval. In particular, see the artifact task below.
11. Update `agents/TASKS.md`, `agents/DECISIONS.md` when a policy changes, and `agents/HANDOFF.md` with commands, evidence, files, and blockers.

Implement the tasks in priority order. Do not mark a task complete until its tests and acceptance conditions pass.

## P1-A: repair the standard CUDA evaluation path

### Defect

`eval_codec._predictor` currently moves a `ClipBatch` to CUDA before registering its topology. `PVBFrameEncoder.prepare_batch` requires topology registration from the CPU batch. A fresh topology-mode model therefore fails on the first standard evaluation forward with:

```text
RuntimeError: topology_id '<id>' was not registered on CPU before CUDA transfer
```

The same ordering error exists in the DDP section of `smoke_codec.py`, which moves the batch before calling the model.

### Required change

- In every topology-mode inference/evaluation/DDP caller, call the underlying codec model's `prepare_batch(cpu_batch)` before transferring tensor fields to CUDA.
- Make this ordering hard to misuse. Prefer one small shared helper for "prepare CPU graph inputs, then transfer" if it can be introduced without broad refactoring.
- Do not silently copy `bond_index`, `atom_ptr`, or other graph metadata back from CUDA.
- Ensure distance-only mode does not attempt topology registration.
- Update `eval_codec.py`, `smoke_codec.py`, and any other callers found by tracing direct `PVBCodecModel`/`PVBFrameEncoder` forwards.

### Acceptance

- A fresh topology model can evaluate a CPU `ClipBatch` on CUDA through the standard `eval_codec` predictor.
- A regression test must fail if transfer happens before topology registration.
- The DDP smoke prepares each rank's CPU batch before transfer and no longer depends on a warm cache.
- The existing CPU `dense_test` tests continue to pass.

## P1-B: persist and enforce the complete graph/model contract in checkpoints

### Defect

`CodecTrainer.checkpoint_state()` saves only `CodecTrainConfig.as_dict()`, which contains `time` and `training` but no model configuration. In particular it omits:

- `bond_construction.mode` and its cutoffs/cap;
- `neighbor_backend`;
- `spatial_execution`;
- spatial and temporal architecture settings;
- graph/cache contract version;
- the canonical-reference policy or immutable inferred-bond contract.

Topology and distance-only models intentionally have identical parameter state keys, so `load_state_dict()` cannot detect a graph-mode mismatch. `eval_codec.py` also omits `bond_construction` from its model-config whitelist, while `evaluate_selected_three_system_overfit.py` trusts the CLI `--bond-mode` without checking the checkpoint.

### Required change

- Define a versioned, JSON-safe model/graph contract and persist it in every new codec checkpoint.
- It must contain every constructor option that can alter forward semantics, even if it does not alter parameter shapes.
- Persist enough distance-only metadata to reproduce the exact frozen graph without consulting validation target coordinates. This may be a versioned canonical-reference manifest plus hashes, or a versioned non-parameter inferred-bond payload. Keep it outside `model_state` so parameter keys remain unchanged.
- Make evaluators reconstruct the model from the checkpoint contract by default.
- Reject conflicting YAML/CLI model or bond-mode settings with a clear error. Do not silently prefer one source.
- Propagate all supported model keys in the standard evaluator, including `bond_construction`, `spatial_execution`, and relevant cache/cutoff settings.
- For legacy `pvb.codec.checkpoint.v1` files without a model contract, require an explicit compatibility path and explicit graph mode. Do not guess silently. Document the migration behavior.
- Consider bumping the checkpoint schema if the new required fields cannot be represented safely as a backwards-compatible extension.

### Acceptance

- Loading a distance-only checkpoint as topology, or topology as distance-only, fails before inference.
- Passing a mismatched `--bond-mode` and `--run-dir` to the selected-system evaluator fails clearly.
- Standard distance-only evaluation uses the checkpoint's graph mode and exact frozen graph contract.
- Model `state_dict` parameter keys/shapes/count remain unchanged.
- Tests cover non-shape semantic mismatches such as cutoff, temporal ratio, time scale, neighbor cap, and bond mode.

## P1-C: make checkpoint resume atomic and configuration-consistent

### Defect

`CodecTrainer.load_checkpoint()` requires `payload["config"]` but ignores it except for a separate precision field. A checkpoint saved with `lr=0.1` currently loads successfully into a trainer configured with `lr=0.2`, leaving `trainer.config.lr == 0.2` while the loaded optimizer has `lr == 0.1`. Loss schedules, bucket weights, warmup, and other semantics can likewise diverge. Model and optimizer state are loaded before the precision mismatch check, so a rejected load can partially mutate the trainer.

### Required change

- Validate the checkpoint schema, complete model/graph contract, training config, precision, normalization contract, and sampler compatibility before mutating model, optimizer, counters, or statistics.
- Define and document resume policy:
  - exact semantic fields must be restored from the checkpoint or match exactly;
  - only a small explicit allowlist of operational overrides may differ, such as output directory and a larger final `max_steps`;
  - unsupported overrides must raise before state mutation.
- Ensure optimizer hyperparameters and `trainer.config` agree after a successful load.
- Preserve the existing sampler epoch/batch restoration, but validate it as part of the atomic preflight.
- If load fails, prove that model parameters, optimizer state, counters, and normalization statistics remain unchanged.

### Acceptance

- Tests reject mismatched learning rate, warmup, loss schedule, bucket weights/tolerances, precision, graph mode, and sampler contract.
- A permitted `max_steps` extension resumes at the exact batch and produces the same next optimizer step as an uninterrupted run.
- A failed load is atomic.
- Save/load of a matching checkpoint remains supported.

## P1-D: implement the FP32 regression gate that Phase A claims to pass

### Defect

`molvid_graph_engineering_v1_2_reviewed/ACCEPTANCE.md` requires an FP32 output/loss/gradient regression. `scripts/run_engineering_acceptance.py::_fixed_control_evidence` currently checks only finite values, parameter names, and graph invariants, then writes the new tensors. It never compares them with a trusted reference and never computes parameter or coordinate gradients. The configured `fp32_scalar_*`, `fp32_vector_*`, `total_loss_relative`, and `gradient_relative` tolerances are copied to JSON but not enforced.

### Required change

- Add real, failing FP32 comparisons for at least:
  - spatial scalar output;
  - spatial vector output;
  - decoded coordinates;
  - every loss component and total loss;
  - representative parameter gradients, with coverage checks for all trainable submodules;
  - coordinate/position gradients.
- Use the configured tolerances in the pass/fail decision.
- A reference must be immutable during the run. Never regenerate the reference and compare it to itself in one invocation.
- Because the historical CPU and repaired CUDA capped edge sets differ, distinguish clearly between:
  - cap-unsaturated synthetic cases where exact topology-edge semantics and output/gradient equivalence are required;
  - the explicitly versioned repaired CUDA reference used for the fixed real batch.
- If the full trusted reference is kept in an external artifact store, require its manifest/hash and fail clearly when it is absent. Do not commit another large binary reference.
- Compute BF16 tensor error as `norm(candidate - reference) / max(norm(reference), eps)`, not merely the difference of tensor norms.
- Add a production-CUDA SE(3) regression; CPU `dense_test` equivariance alone is insufficient for this acceptance claim.
- Set Phase-A status to passed only after every required gate has run and passed. The JSON must record measured differences, thresholds, and reference hashes.

### Acceptance

- A finite perturbation to a loaded model parameter causes the FP32 regression gate to fail.
- Detaching a required gradient path or zeroing a submodule's gradients causes the gate to fail.
- Perturbing a tensor while preserving its overall norm is detected by the BF16 comparison.
- All configured tolerances are referenced by executable assertions, not only copied into output JSON.

## P2-E: enforce train/validation isolation for distance-only references

### Defect

`train_codec.py` and `scripts/run_phase_b_acceptance.py` concatenate train and validation datasets before choosing the lexicographically earliest canonical coordinates for each topology. If a validation record sorts earlier, validation coordinates determine the inferred bonds used during training.

The currently bundled selected-system data happens not to trigger this: train starts at `w000000` and validation at `w000049`. The generic path is still unsafe.

### Required change

- Training graph construction must never inspect validation records or coordinates.
- Choose training references only from the training split or from a predeclared system-level manifest created independently of the split.
- Evaluation must reuse the frozen training/system reference contract from the checkpoint or manifest. It must not infer the training graph from validation targets.
- A topology that exists only in validation needs an explicit, split-independent reference policy; otherwise fail rather than leak.
- Record reference IDs, coordinate hashes, split provenance, and inferred-bond hashes in protocol/checkpoint metadata.

### Acceptance

- Construct train and validation records with the same topology but conflicting threshold-crossing distances and an earlier validation ID. Mutating the validation coordinates must not change any training reference or inferred bond.
- The protocol records no validation sample as a training graph reference.

## P2-F: remove implicit CUDA synchronization from the forward hot path

### Defect

The production forward path repeatedly converts CUDA tensors to Python values or branches on CUDA scalar tensors. Examples include:

- `int(atom_ptr[...])` and Python conditions in `PVBFrameEncoder.prepare_batch/build_graph`;
- `torch.any(...)`/`torch.equal(...)` used as Python booleans during every forward;
- `.item()` and Python branching in `CudaRadiusNeighborList.__call__`;
- redundant GPU `prepare_batch` after the trainer already registered the CPU batch.

These operations create implicit synchronization and undermine pinned/non-blocking loading.

### Required change

- Keep immutable per-sample atom offsets/counts as host metadata or precompute them during CPU collation/registration.
- Move static contract validation to CPU data validation or one-time cache registration where possible.
- Remove the redundant GPU topology-prepare pass from the model forward while retaining a clear error for callers that skipped CPU preparation.
- Keep per-forward edge construction and geometry device-resident and differentiable.
- Preserve strict cap/error semantics. One-time synchronization during canonical distance-bond registration is acceptable if documented; repeated synchronization in every training forward is not.
- Do not weaken correctness checks silently. Relocate them or make a clearly separated debug/preflight mode when necessary.

### Acceptance

- Profile representative topology and distance-only training steps with `torch.profiler` after warmup.
- Report `aten::_local_scalar_dense`, CUDA synchronization, data-wait time, and step latency before/after.
- The steady-state graph/model forward should not contain batch-size-proportional Python scalar extraction from CUDA tensors.
- Output, loss, parameter-gradient, and position-gradient regressions still pass.

## P2-G: remove large checkpoint blobs from the proposed Git history

### Defect

The reviewed branch adds roughly 1.6 GiB of `.pt` files. The three baseline contracts are about 934 MiB and the three repaired contracts about 531 MiB, in addition to overfit checkpoints. This violates the repository rule against committed generated checkpoints and permanently inflates clone/fetch/CI cost.

### Required change

- Inventory every newly tracked binary artifact with path, size, purpose, and SHA-256.
- Move required artifacts to the operator-approved artifact store or Git LFS; keep only small JSON/Markdown/CSV metadata, hashes, manifests, and reproduction commands in normal Git.
- Add a CI guard against unexpectedly large tracked files and generated checkpoints.
- Simply deleting the files in a later commit does not remove their weight from branch history. Prepare the exact history-cleanup or clean-commit reconstruction plan needed before merge.
- Do not run `git filter-repo`, rebase, force-push, delete the existing artifacts, or otherwise rewrite history until the operator explicitly approves the exact operation and confirms artifact preservation.

### Acceptance

- The proposed merge history does not introduce the 1.6 GiB blobs into normal Git object history.
- Every externally stored reference has a checked-in manifest, hash, schema/version, and reproduction command.
- A clean clone can run unit tests without downloading full experimental checkpoints; acceptance requiring external references fails with an actionable message if they are absent.

## Additional scalability check

Audit `DistanceOnlyBondCache` capacity behavior. It precomputes all references into a bounded cache and evicts old entries; a dataset with more topology IDs than `distance_bond_cache_capacity` may later fail because evicted references are never reloaded. Also store and validate the exact atom count/identity contract for each distance-only topology rather than inferring compatibility only from the maximum edge index. Add tests or report a precise blocker if full-data topology counts cannot be inspected.

## Required test plan

Run the smallest relevant tests first, then the full suite. At minimum add and execute tests covering:

1. CPU topology registration before CUDA evaluation transfer.
2. Standard topology and distance-only evaluation entry points.
3. DDP smoke ordering on at least two ranks when GPUs are available.
4. Checkpoint model/graph contract persistence and mismatch rejection.
5. Atomic resume rejection and uninterrupted-versus-resumed next-step equality.
6. Train/validation reference isolation.
7. FP32 scalar/vector/coordinate/loss/parameter-gradient/position-gradient regression.
8. BF16 true relative tensor difference.
9. Production CUDA SE(3) behavior.
10. CUDA synchronization profiling.
11. Distance-bond cache capacity and atom-count mismatch.
12. A tracked-file size guard.

Use commands equivalent to:

```bash
enter-container
source /home/sihao/miniforge3/bin/activate torch-ito
cd /workspace/PVB
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider <targeted tests>
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

Serialize GPU-heavy work. Do not start long/full-data training. The pre-fix full suite was `42 passed, 1 warning`; passing that unchanged suite alone is not sufficient because it did not cover the broken standard CUDA evaluation or the missing FP32 gate.

## Required final handoff

Return all of the following:

1. A finding-by-finding table for P1-A through P2-G: fixed, partially fixed, or blocked.
2. Exact files and symbols changed.
3. Exact test/profile/acceptance commands and concise results.
4. Checkpoint schema and legacy-migration behavior.
5. Proof that train/validation references are isolated.
6. FP32/BF16 measured differences and thresholds.
7. CUDA synchronization before/after evidence.
8. Binary-artifact inventory and the operator action still required, if any.
9. Remaining unverified areas or scientific risks.
10. Final `git status --short` and confirmation that no unrelated changes were altered.

Do not stop after making the standard tests green. The work is complete only when the semantic evaluation, resume, regression, split-isolation, synchronization, and artifact acceptance conditions above are either demonstrably satisfied or reported as explicit operator blockers.
