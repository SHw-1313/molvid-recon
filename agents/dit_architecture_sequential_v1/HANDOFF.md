# R4 DiT architecture sequential v1 handoff

## 2026-09-14 preflight

- Target worktree: `/data4/users/sihao/workspace/molvid-dit-architecture-sequential-v1`
- Branch: `exp/dit-architecture-sequential-v1`
- Audited base and initial HEAD: `027659fe872a6c869b14ecce5b799356bd695694`
- Initial target status contained only the operator-supplied untracked directory
  `agents/dit_architecture_sequential_v1/`; it was preserved.
- The required commit object, branch, and worktree all existed, so no fetch or worktree creation was
  performed.
- Runtime entrypoint exists at `/usr/local/bin/enter-container`. The interactive container maps
  this worktree to `/workspace/molvid-dit-architecture-sequential-v1`; `conda activate torch-ito`
  was used for every Python command below.
- GPU audit command:
  `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader`
  plus the matching compute-process query. GPUs 0--5 had active training/inference processes and
  GPUs 6--7 each held about 79 GiB for vLLM. No process was touched.

### Isolation provenance

- Read-only capacity branch: `exp/dit-capacity-data-v1`, observed HEAD
  `b7754260e0c22ebf8f75a4a68c1e323773297eeb`.
- Commit `8eb0a0d45f26c3b761c72c740f4c7da192afc1fb` changes the legacy source runner so each trainer
  constructs its own adapter before constructing/loading its model. It also contains the capacity
  runner and its CUDA cross-mutation/resume test; the unrelated capacity implementation will not be
  imported wholesale.
- Commit `f839de97840c7896a56566fec06d6cbe6f93eaea` makes capacity source decode use the capacity arm's
  data adapter. The sequential runner/evaluator must preserve the same arm-local adapter rule.
- Capacity HANDOFF reported isolation CUDA evidence passing on neibu at numerical code commit
  `b68dafa968b72519e976f795b7611673c1154a0f`, with shared frozen codec/statistics only.
- At inspection time C48 was still listed in the neibu `G48, C48, C48D8` asynchronous queue and the
  host-visible formal output contained only `preflight.json`. No checkpoint was read or inferred to
  be complete from its name.

### Frozen experiment assets

- Manifest: `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`
- R4 codec: `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`
- Codec expected SHA256: `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`
- Statistics: `/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1/outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio4_state_detail/shared/statistics.pt`
- Statistics expected SHA256: `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`
- Sealed test payloads have not been opened.

### Next actions

1. Inspect current construction/checkpoint/evaluation paths and apply only the minimum arm-local
   adapter repair needed in this worktree.
2. Add direct parameter-storage, optimizer/scaler/RNG, checkpoint-resume, and sequential evaluation
   loading checks.
3. Verify whether a completed C48 becomes available; otherwise profile and budget a clean baseline
   training once an idle CUDA GPU is available.
4. Implement Round 1 only. Rounds 2--4 remain blocked by the required sequential decisions.

### Environment note

- The native `apply_patch` helper failed before writing because its inner bwrap could not configure
  loopback (`RTM_NEWADDR: Operation not permitted`). Changes were applied as an equivalent reviewable
  `git apply` patch through the approved outer command.

## 2026-09-14 Round 1 implementation and metadata evidence

### Imported repairs and new contracts

- Imported only the 21-line `scripts/run_dit_source_ab.py` arm-local adapter construction diff from
  audited B commit `8eb0a0d45f26c3b761c72c740f4c7da192afc1fb`; no capacity runner, data code,
  output, or branch merge was imported. The same commit is the provenance for successful-update /
  AMP-skip accounting now extended for this phase.
- Imported the AdamW scalar-step restore behavior from audited commit
  `9936976cfd46b437477f7df7948db7516490afc0`, then made the new sequential checkpoint contract
  strict: no implicit successful-update fallback is accepted. The one explicit C48 conversion reads
  the audited nested capacity field and requires the exact capacity schema.
- New evaluation loads one complete model plus its own adapter, validates the model/config/data/codec/
  statistics contract, evaluates it, and releases it before loading the other arm.
- The sequential runner owns distinct model/adapter/optimizer/scaler/training-generator objects per
  arm, stores generator state and sampler cursor, counts future-only atom-frame tokens, and resumes
  paired training only from the largest atomic checkpoint present at equal exposure for both arms.
- C48 reuse is accepted only after a PASS completion summary plus exact 4,500-update / 267,988,032-
  token exposure, audited `b68dafa...` frozen experiment contract, conditional/base48/depth-4 recipe,
  train/valid semantic index identity, optimizer/scaler/RNG/cursor, and frozen codec/statistics checks.
- Round 1 adds only `ffn_norm_source=post_adaln|pre_adaln`. The FFN vector branch continues to read
  post-AdaLN vectors; only the scalar norm slots may read the saved pre-AdaLN vector, accumulated in
  FP32. Reference and factorized-v2 paths share this explicit contract.

### Preflight and fixed selection

Command (inside `enter-container`, `torch-ito`):

```bash
python scripts/run_dit_architecture_sequential_v1.py --stage preflight
```

Result: PASS. The artifact is
`outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/preflight.json`.
It records 48 train / 8 valid / 8 sealed test systems, codec SHA256
`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`, statistics file SHA256
`4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`, and
`test_payload_opened=false`. The registered selection has the first sorted eight train systems and
all eight valid systems at R1/window30, all 72 final valid clips, and the historical
`atlas_1v7r_A`/`atlas_2wlt_A`/`atlas_5e3e_A` × R1/R2/R3 smoke clips.

C48 remains unavailable: the required read-only
`C48/train_summary.json` is absent, so no checkpoint was opened or guessed complete. The fallback
path will profile first and then train a clean scratch baseline under the frozen total budget.

### Checks completed without claiming CUDA evidence

Commands (inside `enter-container`, `torch-ito`):

```bash
python -m py_compile scripts/run_dit_architecture_sequential_v1.py \
  evaluation/dit_architecture_sequential_v1.py \
  tests/test_dit_architecture_sequential_cuda.py module/molecular_dit.py \
  module/dit_backend_v2.py trainer/dit_trainer.py
python -m pytest --collect-only -q tests/test_dit_architecture_sequential_cuda.py
python -m pytest -q tests/test_dit_architecture_sequential_cuda.py \
  -k 'future_token_accounting or paired_resume'
```

Results: Python compilation passed; 9 targeted tests collect; all five metadata/I/O tests passed.
They prove the fixed generation-seed key, H4/H8 future-token counting, and common atomic paired-
checkpoint/JSONL truncation. They are not recorded as CUDA numerical evidence. `ruff` is not
installed in the frozen environment; no package was installed to add it.

### Current CUDA constraint and next exact order

A second local GPU audit still found GPUs 0--3 occupied by the existing ESM DDP job, 4--5 by Boltz,
and 6--7 by vLLM. No process was touched and no CUDA test/profile/smoke/training was attempted.
When one card becomes genuinely idle, continue in this order:

1. gated Round 1 CUDA correctness tests;
2. generation-inclusive end-to-end profile and frozen budget;
3. clean scratch baseline (unless C48 has become fully complete and passes every import check);
4. compact baseline generation evaluation;
5. 3-system/9-trajectory Round 1 smoke;
6. directly affected CUDA backend/source/checkpoint regressions;
7. paired Round 1 continuation, fixed generation evaluation, and decision.

Rounds 2--4 remain intentionally unimplemented until the preceding scientific decision exists.

### Round 1 extension and stale-evidence safeguards

A static audit found that the first implementation could emit `NEEDS_COMMON_EXTENSION` while
`train-round1 --resume` still targeted the original exposure. The runner now freezes the extension
size/tokens in `budget.json`, permits it exactly once after a matching decision, resumes both arms
from a common equal-exposure checkpoint, records cumulative and invocation cost separately, and
selects the simple control if the single extension remains inconclusive. Decision loading now
rejects quick/final summaries whose checkpoint SHA256 is stale after continuation. After this change,
Python compilation and all five metadata/I/O tests passed after the additional checks. The latest refreshed PASS preflight records
runner SHA256 `f198637cc11a08f4c45e75ace7ea1d601244475106ccc434855799b88a2947a2` and source hash
`4fc588113409a7dd2285bc18cc32128925fcb15d25f588942136afa2887ae1eb`.

A fresh GPU audit still showed the same eight cards occupied (0--3 ESM DDP, 4--5 Boltz, 6--7 vLLM);
no process was touched and no CUDA numerical command was started.

### Static hot-path and reference-control audit

- The first runner rebuilt the full sampler plan and `clip_spec_table()` at every update. No
  scientific command used that path. It now caches both per epoch/per invocation, matching the fixed
  capacity runner's semantics. A metadata regression proves exactly one plan construction per epoch.
- The 4,500-update future-token calculation now completes in 7.187 seconds and returns 167,474,596.
  A separate read-only schema audit reproduced B's frozen capacity count of exactly 267,988,032
  effective atom-frame tokens at 4,500 updates. The values differ intentionally: this phase reports
  only valid future atom-frames, while the capacity contract counts each clip's full effective tokens.
- The baseline evaluator now computes oracle-decode and repeat-last controls once per fixed scope,
  reuses the oracle coordinates across H4/H8 for each clip, hashes the selection/metric code/row file,
  and makes later arms reuse the saved reference summary rather than recomputing it.
- The evaluation loader now rejects generic checkpoints without the sequential generator/cursor/
  schedule provenance and requires the norm source to agree in config, model contract, and checkpoint
  provenance. The gated CUDA isolation test covers this rejection and two independent full loads.
- Baseline startup refuses to switch to a newly appearing C48 after any scratch lineage artifact has
  been written. C48's exact 4,500-update/267,988,032-token contract and numerical source commit
  `b68dafa968b72519e976f795b7611673c1154a0f` were confirmed from B's read-only HANDOFF and fixed code.

No CUDA evidence is claimed for these static changes. The latest GPU audit still found all eight cards
occupied; C48's local completion summary remains absent.

## 2026-09-14 21:37 +08:00 blocked stop

Final audited local commit before the stop: `333f934bfe72428ea6002b649d5b6eb5f4fdc08c`
(`Implement sequential DiT Round 1 scaffold`). The post-commit preflight is PASS and records that
commit with source hash `4fc588113409a7dd2285bc18cc32128925fcb15d25f588942136afa2887ae1eb`.

The final process-level audit found no idle local CUDA device:

- GPUs 0--3: ESM DDP PIDs 3608913--3608916, 44.2--45.8 GiB used, 73--100% utilization.
- GPUs 4--5: Boltz PIDs 3214936--3214937, 10.7 GiB used, 60--93% utilization.
- GPUs 6--7: vLLM PIDs 3154639--3154640, 79.4 GiB used.

The read-only C48 completion summary was still absent. No process was signalled or modified. After
repeated 180-second monitor intervals and fresh process audits, the required idle-GPU condition was
not met. Therefore no CUDA correctness result, profile/budget, clean baseline, smoke, training,
generation evaluation, or Round 1 decision exists. Rounds 2--4 remain intentionally unimplemented.
The truthful terminal state for this invocation is `SEQUENTIAL_V1_BLOCKED`.

Resume only after an idle card is confirmed. Exact order remains: gated Round 1 CUDA tests; profile
and frozen budget; strict C48 import if it has completed or clean scratch baseline; compact baseline
generation plus saved references; 3-system/9-trajectory smoke; directly affected CUDA regressions;
paired Round 1 train/evaluate/decision. Do not implement Round 2 before that decision.

## 2026-09-15 resumed sequential evidence: Rounds 1--2 complete

All CUDA work below ran on the user-authorized `neibu` host through `enter-container` with
`conda activate torch-ito`, mapped source `/workspace/molvid-dit-architecture-sequential-v1`, and
an audited idle physical GPU 2 exposed as `CUDA_VISIBLE_DEVICES=2` / `--device cuda:0`. No test
payload was opened; no other GPU process was touched. Completed small JSON/Markdown evidence has
been copied back under the local target output root; checkpoints, predictions, and data remain on
the remote compute worktree.

### Round 1 -- pre-AdaLN scalar-norm candidate: REJECT

- Parent was the clean C48 lineage at step 4,500. Both arms first received 3,500 added updates and
  then the one permitted common 1,750-update extension: 5,250 successful updates and
  194,960,744 future atom-frame tokens per arm, final step 9,750.
- The unique variable was only `ffn_norm_source=pre_adaln` versus the equal-exposure post-AdaLN
  control. The candidate was rejected: paired primary relative change was -0.2128% (H4) and
  -0.2144% (H8), with only 1/8 systems in the favorable direction for each history.
- Selected continuation is the Round 1 control checkpoint `checkpoint_final.pt`, SHA256
  `ff66c98d6f051fbc1cd3cb8ba29bedaebd453804555a0c4d8d7e2865cdd96375`.
  Recorded Round 1 training cost is 4.947884 GPU-hours. The candidate code/evidence remains
  isolated and reproducible; it is not used as the next parent.

### Round 2 -- calibrated future bond auxiliary: KEEP

- New code adds only differentiable future-bond supervision to RF. The endpoint is
  `z_tau + (1-tau) u`; geometry is decoded through frozen codec weights in FP32, with observed and
  padding frames excluded. Lambda zero is an exact RF no-op. The targeted CUDA suite passed
  `3 passed in 6.06s`; the directly affected existing sequential CUDA regression passed
  `9 passed in 6.60s`.
- Calibration on the fixed selection found three applicable batches. Raw bond/RF gradient ratios were
  0.12184, 0.45097, and 0.43405; lambda `0.2303875588749564` gave median auxiliary/RF gradient
  ratio 0.10. The 3-system/9-trajectory real-clip smoke passed, with 4 genuinely eligible candidate
  optimizer steps (the earlier duplicated-generation reporting count was corrected).
- Profile froze 5,000 added updates per arm (142,319,624 future atom-frame tokens), plus a single
  unspent common 2,500-update extension; it reserved 8 GPU-hours for later rounds and 12 GPU-hours
  for evaluation/recovery.
- Both arms completed the fixed continuation from the Round 1 selected control: 5,000 successful
  updates each, final step 14,750, independent model/adapter/optimizer/scaler/RNG, and unchanged
  frozen codec/frame-encoder hashes. Cost was 3.781926 GPU-hours; peak memory was 78,717,093,888
  bytes. Candidate checkpoint SHA256 is
  `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34`.
- Quick and final independent-load generation evaluations passed. The final 72-valid-clip result
  selected the candidate without using the extension: system-equal future free-generation bond RMSE
  relative improvement was 83.3901% (H4) and 83.8367% (H8), 8/8 systems favorable for both.
  Contact F1 increased by 0.3902/0.3932; amplitude error decreased by 75.29%/72.23%; no motion
  collapse guardrail fired. The decision is `KEEP`, with one training seed and no proof of
  long-rollout stability as remaining risks.

### Next exact action

Proceed to Round 3 only: freeze the selected Round 2 DiT and adapter, train equal local-only (L)
and cross-block (T) coordinate-supervised temporal refiners from identical initial weights and
training pairs, retain P as the no-refiner baseline, and compare T versus L before attributing any
benefit to cross-block connections. Do not add Round 4 history corruption until that decision.

## 2026-09-16 resumed sequential evidence: Round 3 complete

### Round 3 -- post-decoder temporal refiner: KEEP_PARENT

- CUDA correctness on neibu GPU 2 passed `4 passed in 4.40s` for identity, zero-vector correction,
  observed/padding masking, SE(3), sample isolation, L/T connectivity, and gradient isolation.
  The directly affected R2+R3 CUDA regression passed `7 passed in 6.14s`.
- The 3-system/9-trajectory true-clip smoke passed for P/L/T at H4/H8: all outputs finite, latent
  and coordinate observation clamps exact, and each refiner received 9 independent optimizer steps.
- Profile fixed equal L/T exposure at 5,000 updates and 142,319,624 future atom-frame tokens per
  arm, with one unspent 2,500-update common extension. Pair P90 was 2.4141 s/update and peak memory
  was 55.6 GB. The selected Round 2 parent was SHA256
  `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34`.
- Both refiners completed the matched training, independently owned optimizer/scaler/RNG, and left
  parent DiT/adapter/codec hashes unchanged. Cost was 2.774936 GPU-hours; peak memory was
  60,404,111,360 bytes. Final L/T refiner checkpoint SHA256 values are respectively
  `8465fe8298e23d9257e72e4fa05ba9a20ea2b55b967e9dc447c194a2e6d8793b` and
  `85aac79723b4d3f94ef7113358a1a80fc0d5408d582a144959715c0260f7150a`.
- Fixed-noise quick free-generation evaluation selected P. Under the registered
  `abs(log((boundary/within)_prediction/(boundary/within)_MD))` metric, L versus P was -0.10%
  (H4) / -0.10% (H8), favorable in 0/8 and 1/8 systems; T versus L was -5.61% / -4.97%, favorable
  in 0/8 systems. Bond/contact/amplitude guardrails passed, but neither refiner met the primary
  threshold. Therefore no final 72-clip evaluation and no common extension were run.

### Next exact action

Proceed to Round 4 from the frozen selected Round 2 parent with no refiner. The sole new variable is
the pre-registered observed-history corruption policy; future targets stay clean. Do not revive the
rejected R3 refiner as a conditioning change.

## 2026-09-16 R4 recovery and formal paired run in progress

- The first formal R4 invocation reached exactly 500 successful updates and 13,900,316 future
  atom-frame tokens per arm, then OOMed while encoding the first 80k-token clean validation batch.
  Neither arm wrote a checkpoint, so those updates are not part of the scientific comparison.
- The completed histories plus pre-recovery smoke/profile/budget evidence were moved, not deleted,
  to `round4/attempt_001_validation_oom/`. The observed history interval was 3,170 seconds; a
  conservative 3,600 GPU-second recovery debit is reserved in the task-level budget ledger.
- Recovery preserves the selected 72 validation clips, H4/H8, model, optimizer, data, loss, and
  generation contracts. It only re-packs validation deterministically under a registered
  56,672 atom-frame cap and clears inactive allocator cache before each arm's validation.
- A real CUDA double-resident validation passed for both arms: 72 clips, 46 capped batches,
  identical initial RF total `0.7243899320`, and approximately 54.73 GB incremental peak each.
  The `round4/validation_recovery.json` record captures these details.
- After `7 passed` targeted R2/R4 CUDA regression, the 3-system/9-trajectory R4 smoke and the
  revised profile both passed. The frozen base continuation remains 5,000 updates and
  142,319,624 future atom-frame tokens per arm, with only the one 2,500-update common extension
  available if later decision evidence requires it. The restarted formal pair is now running from
  the selected R2 parent with no R3 refiner.

## 2026-09-17 neibu disk recovery and local CUDA continuation

- The neibu container became unavailable after a host-disk fault. The neibu host filesystem remained
  readable over SSH, so no remote process was restarted or modified. The completed final checkpoints
  were copied read-only to the local target; SHA256 verification passed for R4 control
  `994e69e653159db0a3af3101ea8f0cae690a754dcbd8d09ec897c2649d26f2e6` and candidate
  `965641950429e63b6dc6b40849e36f0731f106518e5d2abe419611706c74abc8`. Baseline and R1--R3 final
  checkpoints were also synchronized to preserve the parent lineage.
- Local host audit found physical GPUs 5--7 idle. The local `enter-container` / `torch-ito`
  environment reported PyTorch `2.5.1+cu121`, CUDA available, and eight visible devices. All local
  numeric work below used physical GPU 5 exposed as `CUDA_VISIBLE_DEVICES=5` / `cuda:0`.
- The exact R4 quick evaluation was resumed locally from the synchronized final checkpoints with the
  registered H4/H8, eight-system validation set, fixed seeds, 16-step Euler sampler, and three-segment
  rollout. Both clean arm summaries and rollout completed `PASS`; the regenerated rollout rows SHA256
  is `48c7e9cba5775f82931bd407a5235de53517652cfc988a527cf1038210c8e1c8`, identical to the remote
  evidence. The local decision rerun is `REJECT`, selected arm `control`, with candidate relative
  `E_roll_bond` change `-0.02861728715822079`, 2/8 systems favorable, and clean guardrails passing.
- The user-requested recovery path was therefore sufficient; no further diagnosis of the prior neibu
  interruption was performed. Test payloads remain unopened. The old `status.json` is retained as the
  historical preflight snapshot; final completion evidence is in the run-level final report/status.
- Exact local continuation commands (inside `enter-container` / `torch-ito`) were:
  `CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 python -u scripts/run_dit_architecture_round4.py --config config/dit_architecture_sequential_v1.yaml --stage evaluate-round4 --scope quick --device cuda:0`, followed by the same command with `--stage decide-round4`; the local R4 CUDA check used `CUDA_VISIBLE_DEVICES=6 DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1=1 python -m pytest -q -p no:cacheprovider tests/test_dit_architecture_round4_cuda.py` and passed `5 passed in 6.26s`.

## 2026-09-17 motion metric recheck and historical correction

- No training, checkpoint load/write, generation, or test-payload access was performed. The neibu
  host remained read-only; its completed Round 2 final and Round 3 quick `summary.json` plus
  `generation_rows.jsonl` files were synchronized locally and SHA256-verified. The summaries were
  sufficient for the correction, so rows were hash-recorded but not re-aggregated.
- Added `evaluation/motion_metrics.py`, a shared reader for the current flattened
  `future["rmsf.prediction"]`/`future["rmsf.target"]` fields and the explicitly supported legacy
  nested form. Both forms must agree or a `MotionMetricError` is raised. Zero is retained as a
  valid prediction; missing, non-finite, incomplete, and near-zero-denominator cases carry reasons
  and are classified as `FAIL`, `INCOMPLETE`, or `NOT_APPLICABLE`, never an implicit pass.
- Updated `scripts/run_dit_architecture_round2.py::_motion_ratio` and
  `scripts/run_dit_architecture_round3.py::_motion_values` and their guard decisions to use the
  shared reader and require an explicit `PASS` motion status. No other erroneous system-row
  nested read site remains; the reassessment interpretation code already used flattened keys.
- Targeted regression `python -m pytest -q -p no:cacheprovider tests/test_motion_metrics.py` passed
  `10 passed`. The JSON-only recheck command was `python scripts/recheck_dit_motion_v1.py` inside
  `enter-container`/`torch-ito`; it wrote
  `outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/motion_recheck_v1/`.
- Using the original decision scopes and pairings, Round 2 final candidate/control is complete
  (8/8 systems at H4 and H8) but motion guard FAIL: median prediction-RMSF ratios are
  `0.297096802` (H4) and `0.328881603` (H8), below the unchanged `0.5` threshold. The corrected
  branch is `TRADEOFF` / control, versus the historical `KEEP` / candidate. Round 3 quick
  local/parent passes at `0.997843888`/`0.997508324`; cross-block/parent passes at
  `1.016518180`/`1.018089405`, all with 8/8 complete pairs, so its historical `KEEP_PARENT` remains
  unchanged. Per-system prediction RMSF, target RMSF, ratio-of-means, mean-of-ratios, missing
  lists, and source hashes are in `motion_recheck_v1/report.md` and JSON outputs.
- The old total-report wording that motion guard “passed” is corrected by an appended erratum.
  The corrected recommendation withdraws acceptance of the Round 2 auxiliary candidate; no
  downstream parent re-selection was performed, so R3/R4 remain conditional historical evidence
  from that former parent. The clean Round 1 control lineage is the safe stopping recommendation
  pending a separately authorized rerun.
