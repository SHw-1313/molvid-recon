# B learned-memory formal pre-launch review r01

**Verdict: FAIL — do not launch the frozen formal M-short or M-long jobs yet.**

The reviewed model/dataflow and the paired one-epoch experiment are scientifically consistent with the B specification, but the fixed source has three P1 launch blockers: a checkpoint written by the fresh warm-start path cannot be loaded by the advertised strict-resume path; the 32 GPU-hour formal cap has no cumulative durable accounting or runtime guard; and training/evaluation do not validate the actual XTC sidecar bytes against the per-sidecar hashes in the frozen manifest. There are no P0 findings.

## Review identity and boundary

- Reviewed source commit: `e10148d49068da91bde37aca808b10f56c200cc4` (`HEAD` at review time). The review includes the B implementation in its ancestry and the absorbed public A release base. `git diff --check e10148d^..e10148d` was clean.
- The released A source recorded by the release evidence is `af60c05aa02ce46872edbad0711833317773436e`; the immutable A1 parent checkpoint is `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.
- Frozen formal config identities:
  - M-short: `7d103a90dbd80dfd013c5953c9aaae8982eb5a2c32b778c2057850ea939f608d`
  - M-long: `5a18ff4210e600c40d1f69121785bd37ae5407aaca50f2f03ca440d895282321`
  - `frozen_configs.json`: `7917b420ddb3d1ffd83a2a1dd9b6e551d4e206dd81757620b4d519a2f7d5f875`
  - `budget_decision.json`: `9db1e69ec2ee53cc747bdc22dec7c4cef11efa59b05432b4cc09ac94c8f97136`
- Supporting artifact identities:
  - XTC offset manifest: `21c3dc05271b7c018278b3e4e6972b29987a4b513bc21ac122adfaf9ceda0e3f`
  - one-window direct/sidecar comparison: `ae509d0aeacf996e6ee0cf5001ad810447e059745c833e63c88fb5db37d5265d`
  - recipe release: `38702552fd847f79cf5a6e9c3f0d4b2a104ef7c590316ac6c5395ec2c94626f2`
  - capacity release: `eaa6cb42de10a732654fce4421746e85067987af95f7dc2db644d96c4aebb342`
- I treated `agent/gm_motion_memory_v3/runtime_paths.json` and all other current worktree-only coordination changes as runtime/evidence context, not reviewed model source. No model, training, data, config, or artifact file was edited by this reviewer.
- I read the required root and task documents before examining source and artifacts. I did not use implementation-conversation claims as approval evidence.
- I did not run CUDA or occupy GPU3. The supplied targeted result is **10 passed** on GPU3 (the parametrized learned-memory tests plus common-schedule tests); I did not independently rerun it. I performed read-only source/config/hash checks and a CPU-only checkpoint-payload audit through `enter-container` with `torch-ito` and CUDA hidden.
- Evidence is pre-production and ATLAS-only. It does not establish real-checkpoint/data production parity beyond the named A1 parent, raw ATLAS artifacts, pilots, and bounded evaluation probes.

## Findings by severity

### P0

None.

### P1 — strict resume rejects checkpoints produced by the same run

**Trigger.** Start either formal arm fresh, let it save a checkpoint, then invoke `tools/train_learned_memory_formal.py --resume ... --resume-sha256 ...` with the same frozen config.

**Evidence.** `molvid/training/learned_memory_formal.py:237-249` constructs a fresh `FrameJointTrainer` without a continuation parent. On a fresh run, `load_warm_start` sets `trainer.continuation_parent` to a non-null A1 model-only warm-start record (`molvid/training/joint.py:1213-1223`), and that record is included in `contracts()` and saved in every B checkpoint. On resume, the newly constructed trainer still has `continuation_parent=None` and calls `load_checkpoint` immediately (`molvid/training/learned_memory_formal.py:259-280`). `load_training_checkpoint` requires exact contract equality before restoring any state (`molvid/checkpoints.py:577-590`).

The CPU payload audit of the existing 128-update pilot checkpoints confirmed that both `contracts.continuation_parent` and `extra_state.continuation_parent` contain the A1 warm-start record. A fresh trainer therefore expects `None` where its own checkpoint contains that record. The load fails with `training checkpoint contracts differ` before cursor, optimizer, or RNG restoration.

**Impact.** The two 9741-update jobs are not recoverable through their documented path after interruption. This violates the exact-resume requirement and turns an ordinary interruption into either lost compute or an unreviewed restart.

**Required closure.** Hydrate and validate the immutable continuation-parent contract before strict contract comparison, without weakening the remaining exact checks. Preserve optimizer, scheduler, cursor, exposure, global RNG, training generator, sampled generator, execution hash, and paired hash. Demonstrate a checkpoint created by the fixed B runner loading under the same frozen logical config and continuing with the same next batch/RNG result as an uninterrupted reference. Review the resulting fixed source commit; do not treat a test against different source as closure.

### P1 — the formal GPU budget is projected but neither cumulative nor enforced

**Evidence.** The frozen decision correctly reduces both arms to one epoch: 9741 updates per arm, projected paired cost `23.493326089222368` GPU-hours after a 1.15 safety factor, below the 32-hour formal cap. The underlying completed pilots used 0.13543 and 0.13301 GPU-hours for 128 updates. Separately, the completed 36-prefix M-long evaluation profile supports a steady-state two-arm projection of about `6.8753` GPU-hours for 2304 prefixes, below the 8-hour evaluation cap. These projections support the chosen exposure; they do not replace cumulative accounting of actual runtime.

The runner, however, starts timing anew on every invocation. `train_manifest.json` records `gpu_hours_this_invocation`; the `finally` block rewrites one `cost_ledger.json` with only the current invocation (`molvid/training/learned_memory_formal.py:477-527`). Prior elapsed GPU time is not stored in the checkpoint or accumulated on resume, and no code compares elapsed/cumulative time with `budget.formal_gpu_hours`. The CLI exposes no formal budget guard. This remains a defect even after the resume blocker is repaired.

**Impact.** A slowdown or one or more resumes can silently exceed the explicitly frozen 32 GPU-hour formal authorization. Overwriting the ledger also makes the paired formal cost unauditable. Since the formal budget belongs to the pair, independently completing one arm must not consume the other arm's committed exposure without an explicit symmetric decision.

**Required closure.** Add durable append/accumulate accounting across invocations and both arms, restore/validate it on resume, and enforce the frozen formal cap at checkpoint-safe boundaries. The launch procedure must preserve the one-epoch paired design or stop without making an asymmetric formal comparison. Verify failure/interruption and resume accounting with a bounded test.

### P1 — frozen XTC sidecar hashes are recorded but not enforced by the reader

**Evidence.** The manifest contains 600 records with `xtc`, XTC size/mtime, sidecar path, and `sidecar_sha256`. The reviewer independently hashed all 600 present sidecars: `checked=600 missing=0 mismatch=0`. The supplied direct comparison also shows exact coordinate/time equality for 49 requested frames from one trajectory (`1bkp_A/R1`, raw indices 1280 through 1760 at stride 10).

At runtime, `RawMemoryDataset` verifies only the offset manifest file SHA and carries its aggregate `cache_hash` into the dataset identity (`molvid/data/learned_memory_formal.py:193-207`). `_load_item` derives a sidecar filename from the XTC name rather than binding the file to the corresponding manifest record. `_read_raw_frames` accepts the sidecar after checking only its embedded XTC size/mtime and length (`molvid/data/learned_memory_formal.py:276-290`); it does not compare the file SHA with `sidecar_sha256`.

A syntactically valid sidecar whose offsets are shifted uniformly can retain the embedded source metadata and pass the relative-time checks: every requested read is shifted by the same native frame, so timestamp differences remain correct while the physical anchor identity changes. The frozen data hash binds the expected hashes through the manifest, but the actual bytes used by training/evaluation are not checked against them.

**Impact.** Exact raw-frame identity is a scientific input to this experiment. The current artifacts match today, but the formal runner does not fail closed if an ignored/mutable sidecar changes after review or before a resumed invocation.

**Required closure.** At dataset initialization or first use, map every required XTC to its frozen manifest record, verify the resolved XTC and sidecar paths, XTC identity, sidecar SHA-256, frame count, and basic offset invariants. Add a targeted corruption test that includes a uniform offset shift. Re-run the complete 600-sidecar identity audit under the fixed source before formal launch.

### P2 — evaluation computes the grid but does not finish the required statistical/reporting workflow

`tools/evaluate_learned_memory.py` correctly fixes Euler16 and seeds 0/1/2, uses 8 systems x 3 replicas x 2 anchors, crosses observed dt 100/200/400 ps with query lag 100/200/**300**/400 ps, evaluates persistence/oracle/generated paths, and aggregates draw → anchor → replica → system. It resumes rows by stable keys, saves per-system rows plus representative profiles/coordinates, and checks future leakage and one memory preparation per prefix.

The complete grids are 576 prefixes/2880 rows for M-short and 1728 prefixes/8640 rows for M-long. Current artifacts are expected pre-launch probes only: a one-prefix M-short smoke and M-long profiles of 8 and 36 prefixes. The 36-prefix profile covers one anchor's complete 3-span x 3-dt x 4-lag grid and completed 180 rows in `0.1133125507` GPU-hours, with peak allocation `5,347,197,952` bytes, one memory preparation, teacher observed2/target12, and five representative profiles/coordinates. The measured 8-to-36-prefix steady-state slope projects both complete grids (2304 prefixes) at about `6.8753` GPU-hours, within the 8-hour evaluation allocation; this remains a projection rather than a runtime guard.

There is no B-specific postprocessor for system-paired bootstrap confidence intervals, short-versus-long paired comparisons/selection, required figures, low/medium/high MD-motion example selection, `report.md`, or `selection.json`. The evaluator's summary is a point-estimate bucket table and its GPU time is again invocation-only. This does not block training after the P1 items are fixed, but it must be implemented and reviewed before a scientific selection or completed-line claim.

### P3

None.

## Checks that passed within the evidence boundary

### Raw data, paired sampling, and physical time

- Raw ATLAS native time is treated as 10 ps. Requested frames lie on raw indices divisible by 10, so the observed grid is true 100/200/400 ps sampling rather than reconstruction from a pre-cut 100 ps clip.
- Both arms use the same 12.8 ns-past/4.8 ns-future eligibility set: 192 train systems, 576 trajectories, 29376 eligible anchors. M-short and M-long therefore share anchors and futures. The two observed teacher anchors are exactly `[-800, 0]` ps and each target has Q12.
- M-short supplies the inclusive 0.8 ns history; M-long supplies 0.8/3.2/12.8 ns histories. The frame counts follow the requested physical dt (for example 9/33/129 frames at 100 ps and 3/9/33 at 400 ps).
- Observed-history dt and future query lag are separate schedule fields. Training uses query lags 100/200/400 ps; 300 ps is rejected from training and appears only in raw validation. History physical time and diffusion flow time remain separate model inputs.
- The frozen epoch schedules contain 9741 updates per arm. Their arm-specific execution hashes are `b684a785...` and `04c8d879...`, while the paired anchor/dt/query schedule hash is identical: `604ef984a13f2e065ee7049518dc2f48e349bc13206b8ef081b5fcf94eb9627c`.

### Architecture and conditioning boundary

- The active path is shallow and readable: shared two-layer scalar/vector local atom encoder (64/32) → residue/chemical tokens → physical-ps multiscale analysis plus recent path → 8 learned slots (128/64) → 256/128 conditioning → four-layer per-frame DiT → decoder.
- `history_backend=learned_memory`, DiT depth 4, G enabled, and legacy M disabled are enforced by model/config contracts. The old R4 `HistoryEncoder`, old motion encoder, and old motion-context weights are absent from the active graph and explicitly retired in warm-start reports.
- The new backend does not invoke the legacy whole-clip teacher. Training constructs observed context from the two anchor frames and targets from the 12 future frames. Pilot hooks observed exactly `[2, 12]` teacher calls per update; generation accepts observed raw history, anchors, and query times without a future-coordinate argument.
- Raw history, anchors, and future are aligned to one last-observed Kabsch reference. Future coordinates affect only target encoding/scoring. The eval leakage probe changed target latent values (max difference 6.5945) while leaving observed/query tensors identical and generated coordinates exactly unchanged.

### Optimization, gradients, and checkpoint reconstruction

- The trainer verifies every trainable parameter is owned exactly once by the optimizer. Pilot resolved models contain 16,122,904 parameters, 15,561,680 trainable, with explicit history/DiT/decoder learning-rate groups.
- The learned memory is prepared once per prefix, reused through the base and differentiable K2 sampled-source terms, and is not detached. The source sampler is not the inference-only no-grad sampler. Pilot hooks recorded one preparation; gradient diagnostics were finite.
- Both 128-update paired pilots completed at fixed commit `e10148d...`, were finite, and peaked at 57,570,977,792 and 57,574,855,168 allocated bytes (about 53.62 GiB). Their warm-start reports loaded 387 compatible tensors, initialized 37 learned-memory tensors, retired 94 legacy history/M tensors, and reported no unexpected keys or shape mismatches.
- Inference checkpoint reconstruction is strict on model contract/state keys, order, shape, and dtype. The completed evaluation smoke loaded a B pilot checkpoint and generated successfully. This pass applies to inference reconstruction only; strict training resume is the P1 failure above.
- The A1 recipe, teacher, statistics, configuration, and parent checkpoint hashes are checked at launch. Sealed test remains unopened in the dataset, pilots, and evaluation protocol.

## Formal launch decision

**FAIL / BLOCKED FOR LAUNCH at `e10148d49068da91bde37aca808b10f56c200cc4`.** Neither frozen formal arm is approved to start from this reviewed source. Repair all three P1 findings in one implementation-owned source commit, preserve the frozen scientific configs unless a documented identity update is necessary, produce bounded resume/budget/sidecar evidence, and submit that exact commit and evidence for fresh review. P2 may be closed after training but before formal evaluation claims or B selection.
