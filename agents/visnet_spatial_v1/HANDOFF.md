# HANDOFF — ViSNet spatial backbone v1

Status: prepared; implementation not started.

## Branch

Create:

```bash
git switch fix/graph-runtime-v1
git switch -c feat/visnet-spatial-v1
```

## Prepared scope

Implement three spatial backends under the unchanged ratio-1 temporal codec:

- `torchmd_et`
- `visnet_radius`
- `visnet_bonded`

The authoritative documents are this directory's PLAN, DECISIONS, ACCEPTANCE, TASKS, and
LUNA_CODEX_PROMPT.

## Worker update rules

After each completed task group:

1. update only the corresponding checkbox/status in `TASKS.md`;
2. append a dated evidence entry below;
3. append a decision to `DECISIONS.md` only when an implementation fact forces a deviation;
4. never rewrite `PLAN.md`, `ACCEPTANCE.md`, or root `AGENTS.md`.

## Evidence template

### YYYY-MM-DD — Vxx–Vyy

- Status:
- Base/current commit:
- Files changed:
- Commands:
- Tests:
- Data/manifest:
- Runtime and memory:
- Artifacts:
- Decisions or deviations:
- Blockers:
- Next task:

## Final required handoff

Include:

- complete file list;
- exact environment and dependency versions;
- exact commands;
- test results;
- 3-clip micro-overfit results;
- 441/117 manifest hash;
- exact train coverage evidence;
- per-backbone final metrics and runtime;
- checkpoint/resume evidence;
- recommended development backend;
- unresolved risks;
- explicit statement that state-detail/AF3/VideoDiT work did not begin.

### 2026-09-01 — V00–V03 preparation

- Status: complete; implementation started only after the required phase documents and the historical graph-runtime handoff were read.
- Base/current commit: branch `feat/visnet-spatial-v1`; `git merge-base HEAD fix/graph-runtime-v1` and `git show -s fix/graph-runtime-v1` both resolve to `68dd34e4e6d8467d1af51cdc9404cdfae1dca639` (`record push authentication boundary`). Initial worktree changes are the prepared untracked agent documents only: `AGENTS.md` and `agents/visnet_spatial_v1/`.
- Environment: inside `enter-container`, activated `torch-ito`; Python 3.11.15, PyTorch 2.5.1+cu121, CUDA build 12.1, CUDA available with 8 devices, PyG 2.6.1, torch_cluster 1.6.3+pt25cu121, NVIDIA A100-SXM4-80GB devices.
- Baseline protocol/artifacts: the current `PVBFrameEncoder`/`PVBCodecModel` path in `module/multiframe_codec.py` and `trainer/codec_trainer.py` remains the TorchMD external-graph path; ratio-1 controls are parameterized by `scripts/run_selected_three_system_overfit.py`, while the checked-in `config/codec.yaml` still has the historical ratio-4 default and will be resolved for this phase without changing temporal implementation code. Existing selected-system artifacts are under `outputs/atlas_selected_trajectories/`.
- Commands/tests: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` — 53 passed, 1 pre-existing `torch.load(weights_only=False)` warning. No Python command was run outside `enter-container`/`torch-ito`; no package or network operation was used.
- Decisions or deviations: none. The temporal codec, decoder, losses, optimizer, clip schema, and graph-runtime repairs are out of scope for modification.
- Blockers: none for preparation. GPU jobs are serialized and deferred until implementation gates pass.
- Next task: V10 separate frame-node packing from external graph construction.

### 2026-09-01 — V10–V40 implementation and gates

- Status: implementation and unit/gradient/graph-isolation gates complete; the required micro-overfit is running serially on `neibu` and remains active.
- Base/current commit: branch `feat/visnet-spatial-v1`; base remains `68dd34e4e6d8467d1af51cdc9404cdfae1dca639`.
- Files changed: `module/visnet.py`; `module/multiframe_codec.py`; `module/__init__.py`; `trainer/codec_trainer.py`; `config/codec.yaml`; `train_codec.py`; `eval_codec.py`; `scripts/run_engineering_acceptance.py`; `scripts/run_selected_three_system_overfit.py`; `scripts/evaluate_selected_three_system_overfit.py`; `scripts/run_visnet_spatial_tiny_overfit.py`; `tests/test_visnet_spatial.py`; phase `TASKS.md`.
- Commands/tests: inside `enter-container` with `torch-ito`, local and remote `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider` — `67 passed, 1 warning`; `python -m py_compile` passed for the changed Python modules; `git diff --check` passed.
- Implementation evidence: native `visnet_radius` has no external builder/topology preparation, its native neighbor count is exactly one, `visnet_bonded` consumes the repaired external `FrameGraphBatch` and its binary edge feature, and both variants pass T=1/T=16, SE(3), gradient, and checkpoint-contract tests. The representation block is attribution-marked torchmd-net v2.0.0-derived code and does not instantiate energy/force heads.
- Environment and transfer: exact `torch-ito` was copied to `neibu` at `/data/users/yuansihao/workspace/miniforge3/envs/torch-ito`; local/remote `conda list --explicit torch-ito | sha256sum` both returned `9bc4a80dc7a63d857b1ca08ce3d0f41b0c933acab0219a5c594da2d2b977dd4b`. Remote imports report Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.1, PyG 2.6.1, torch_cluster 1.6.3+pt25cu121, and NVIDIA A100-SXM4-80GB.
- Data/manifest: compact lazy clip store is present at `/data/users/yuansihao/workspace/PVB/outputs/atlas_selected_trajectories/clip_store/`, with `data.bin` size 195951224 bytes and the original train/valid indexes. The tiny runner hashes and validates the exact 441/117 contract before training.
- Artifacts: implementation evidence is `outputs/visnet_spatial_v1/implementation_evidence.json`; generated micro/tiny artifacts remain untracked and are not intended for commits.
- Decisions or deviations: none. The final tiny runner uses the prepared 20-step warmup, FP32, `max_tokens=80000`, replacement-free exact epoch coverage, and a 6000-step safety cap.
- Blockers: none at this stage; the remote micro-overfit process is intentionally not interrupted while code/evidence work continues.
- Next task: V41–V44 complete and verify the three 500-step micro-overfits, then V50–V58 run the exact tiny protocol.

### 2026-09-01 — V41–V52 micro gate and tiny preparation

- Status: complete; all three micro-overfit gates passed and the exact tiny manifest/loader are ready. The final tiny-overfit is next.
- Command: in the activated remote `enter-container`/`torch-ito`, `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -u -m scripts.run_selected_three_system_overfit`.
- Micro artifact: `/workspace/PVB/outputs/visnet_spatial_v1/micro_overfit/run_20260901T190657/summary.json` on `neibu`.
- Micro results: `torchmd_et` initial/final `1.6674200296/0.3131781816` (81.2178% reduction, 303.84 s); `visnet_radius` `1.5450786352/0.4790127575` (68.9975%, 384.25 s); `visnet_bonded` `1.5313600302/0.5777778625` (62.2703%, 269.27 s). All have `finite_losses_and_gradients=true`, checkpoint step 500, and resume step 501. Peak allocated memory was 59055275008 bytes for TorchMD, 37909310976 for native ViSNet, and 37924608512 for bonded ViSNet.
- Graph evidence: TorchMD and bonded ViSNet used the external graph with 1779512 union edges and 159680 binary covalent features; native ViSNet used `native_radius` with 1779510 edges and zero topology preparation/replicated bond edges.
- Tiny preparation: `build_manifest_contract()` returned 441 train, 117 late-holdout, zero overlap, hash `3897187ee968a20c1ed359177c0de4e4ed436998fe828f6be82b3017e6c91963`. The compact remote store is 195951224-byte `data.bin`; loader configuration is lazy, replacement-free, exact epoch coverage, FP32, max tokens 80000, 30 epochs, warmup 20, safety cap 6000.
- Decisions or deviations: none. The micro job was kept serialized on a free remote A100 as requested; local coding and evidence maintenance continued while it ran.
- Blockers: none. The tiny run has not started yet.
- Next task: V53–V58 run/evidence for all three final tiny-overfits and checkpoint resumes.

### 2026-09-01 — V53–V65 final tiny-overfit and stop

- Status: complete. The three selectable spatial backbones, implementation gates, 500-step
  micro-overfits, exact tiny protocol, reports, checkpoint resumes, and final handoff are
  complete. No state-detail, AF3, diffusion, flow-matching, VideoDiT, or next architecture
  phase was started.
- Branch/base/current commit: branch feat/visnet-spatial-v1; HEAD remains the requested base
  68dd34e4e6d8467d1af51cdc9404cdfae1dca639 (record push authentication boundary); all phase
  implementation and evidence changes are uncommitted in the worktree.
- Files changed: config/codec.yaml; eval_codec.py; module/__init__.py;
  module/multiframe_codec.py; module/visnet.py; scripts/evaluate_selected_three_system_overfit.py;
  scripts/run_engineering_acceptance.py; scripts/run_selected_three_system_overfit.py;
  scripts/run_visnet_spatial_tiny_overfit.py; tests/test_visnet_spatial.py;
  train_codec.py; trainer/codec_trainer.py; and the phase TASKS.md/HANDOFF.md files.
  Root AGENTS.md, PLAN.md, and ACCEPTANCE.md were not edited.
- Environment: all substantive implementation, test, and training commands ran through
  enter-container with torch-ito; Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.1, PyG
  2.6.1, torch_cluster 1.6.3+pt25cu121, NVIDIA A100-SXM4-80GB. The local and remote
  explicit conda environment hashes both equal
  9bc4a80dc7a63d857b1ca08ce3d0f41b0c933acab0219a5c594da2d2b977dd4b. Passwordless SSH and
  rsync to neibu were used only for the requested environment/data staging and remote
  execution; no dependency installation/upgrade or external internet access was used.
- Commands and tests:
  - PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider — local and remote
    runs each report 67 passed, 1 warning in torch-ito.
  - PYTHONDONTWRITEBYTECODE=1 python -m py_compile module/visnet.py module/multiframe_codec.py
    trainer/codec_trainer.py train_codec.py eval_codec.py
    scripts/run_selected_three_system_overfit.py
    scripts/evaluate_selected_three_system_overfit.py
    scripts/run_visnet_spatial_tiny_overfit.py tests/test_visnet_spatial.py, plus
    git diff --check — passed.
  - CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -u -m
    scripts.run_selected_three_system_overfit — serialized remote 500-step micro run.
  - CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -u -m
    scripts.run_visnet_spatial_tiny_overfit --micro-summary
    outputs/visnet_spatial_v1/micro_overfit/run_20260901T190657/summary.json
    --implementation-evidence outputs/visnet_spatial_v1/implementation_evidence.json —
    serialized remote final tiny run.
  - The final report was regenerated from all three completed result.json files with the
    patched report writer so parameter counts are present in aggregate rows.
- Implementation evidence: outputs/visnet_spatial_v1/implementation_evidence.json records
  the representation-only ViSNet block, lmax=1, scalar/vector shape, graph-isolation,
  SE(3), gradient, static-T=1, CUDA, and checkpoint-contract gates. The full explicit phase
  suite is green for all three backbones. The existing engineering wrapper still contains a
  pre-existing legacy-control gate; this handoff claims the explicit phase gates above and
  does not claim that unrelated legacy controls were changed or waived.
- Micro-overfit artifact: outputs/visnet_spatial_v1/micro_overfit/run_20260901T190657/summary.json.
  The fixed three clips, 16 frames, and 78,224 effective tokens produced finite losses and
  gradients, at least 30% loss reduction, and step-501 resume for every backend:

  | Backbone | Initial total | Final total | Reduction | Runtime (s) | Peak allocated (bytes) |
  |---|---:|---:|---:|---:|---:|
  | torchmd_et | 1.6674200296 | 0.3131781816 | 81.2178% | 303.840 | 59055275008 |
  | visnet_radius | 1.5450786352 | 0.4790127575 | 68.9975% | 384.253 | 37909310976 |
  | visnet_bonded | 1.5313600302 | 0.5777778625 | 62.2703% | 269.269 | 37924608512 |

  External TorchMD/bonded graphs had 1,779,512 union edges and 159,680 binary covalent
  features; native radius had 1,779,510 edges and zero topology preparation/bond features.
- Final tiny manifest and protocol: authoritative remote run directory is
  /data/users/yuansihao/workspace/PVB/outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557.
  manifest_contract.json records systems atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A,
  replicas R1/R2/R3, train windows w000000–w000048, late holdout windows
  w000049–w000061, T=16, dt_100ps, 441 train clips, 117 late-holdout clips, and zero
  overlap. The authoritative remote built_manifest_sha256 is
  e6ead995dbea156c70933dc0f9dde3e93e5411b78f37aeae2fe099890b301b76. The earlier local
  preflight hash 3897187ee968a20c1ed359177c0de4e4ed436998fe828f6be82b3017e6c91963 was
  computed from a separate compact index representation with the same ordered sample IDs
  but different byte offsets; the remote execution hash is the final run’s authoritative
  contract. The compact remote lazy store is
  /data/users/yuansihao/workspace/PVB/outputs/atlas_selected_trajectories/clip_store/.
  The protocol uses lazy mmap loading, replacement=false, exact no-replacement epoch
  coverage, FP32, max_tokens=80000, 30 epochs, 6000-step safety cap, temporal layers=1,
  temporal ratio=1, lr 1e-4, weight decay 1e-6, 20-step warmup, and grad clip 1.0.
- Tiny-overfit coverage and resume: each backend has 30 epoch records; every epoch reports
  exact train coverage with 441 samples and 441 unique IDs, evaluates all 117 holdout clips
  with 39 clips per system, and remains under the safety cap at step 4979. Each backend saves
  codec_step_00004979.pt and the explicit resume check loads step 4979 and advances to
  step 4980 with status passed.
- Final tiny results:

  | Backbone | Parameters | Tiny quality | Train reduction | Future dRMSD improvement | Final future dRMSD | Runtime (min) | Train tokens/s | Peak alloc / reserved (GiB) | Graph / bond coverage |
  |---|---:|---|---:|---:|---:|---:|---:|---:|---|
  | torchmd_et | 1,974,904 | pass | 66.7757% | 49.8364% | 0.644799669 | 69.288 | 82,979.5 | 56.254 / 78.705 | external / 0.091837 |
  | visnet_radius | 2,133,316 | fail (train reduction 49.3922% < 60%) | 49.3922% | 35.5794% | 0.682236567 | 58.949 | 97,533.0 | 36.094 / 78.326 | native_radius / n/a |
  | visnet_bonded | 2,133,316 | fail (train reduction 49.5524% < 60%) | 49.5524% | 35.2878% | 0.679827729 | 62.029 | 92,690.1 | 36.107 / 78.617 | external / 0.091837 |

  The full per-step training metrics and per-system/all-frame/future evaluation metrics are
  in each backend’s train_metrics.jsonl, epoch_metrics.jsonl, and result.json; each
  backend protocol is in its protocol.json. External graph diagnostics are 56,768 nodes,
  1,272,516 union edges, and 116,864 replicated/binary covalent edges. Native radius has
  56,768 nodes and 1,272,511 distance edges with no external topology preparation.
- Reports and artifacts: the local copy is
  outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557/; it contains
  aggregate_comparison.json, aggregate_comparison.csv, aggregate_comparison.md,
  loss_curves.png, loss_curves.pdf, the manifest/protocol, three per-backbone result and
  metric logs, and three step-4979 checkpoints. The aggregate comparison distinguishes
  implementation pass, micro-overfit pass, and tiny-overfit quality pass. All three passed
  implementation and micro gates; only torchmd_et passed the strict tiny quality gate.
- Recommendation: torchmd_et is the development backend recommendation because both ViSNet
  variants failed the final tiny quality gate. This is a development result from one seed on
  a natural rather than parameter-matched model comparison, with no protein isolation and no
  full-dataset scientific benchmark; it should not be interpreted as a claim that the ViSNet
  representation is invalid. The remote TorchMD run approached the 80-GiB device reservation
  ceiling, while the ViSNet runs used less allocated memory but similar reserved memory.
- Decisions or deviations: no new implementation decision was required and DECISIONS.md was
  not modified. The only staging note is the separate local/remote compact index byte-offset
  hash described above; logical sample-ID coverage is identical and the remote final
  contract is the one used for training.
- Blockers: none. Remote GPU access via passwordless ssh neibu and its copied exact torch-ito
  environment allowed evaluation to continue without pausing local coding or verification.
  No dependencies were installed or upgraded, and no external internet access was used.
- Next task: stop. Do not begin the next architecture phase or any state-detail/VideoDiT work.

### 2026-09-02 — evaluation figures and runtime annotations

- Status: complete; regenerated reporting artifacts from the existing tiny-overfit results
  without rerunning training or changing any acceptance gate.
- Command: inside enter-container with torch-ito, the existing three result.json files were
  passed to scripts/run_visnet_spatial_tiny_overfit.py::_write_reports.
- Figures: the run directory now contains eval_metrics.png/pdf with six holdout metrics
  across all epochs and all three backbones; eval_metrics_torchmd_et.png/pdf,
  eval_metrics_visnet_radius.png/pdf, and eval_metrics_visnet_bonded.png/pdf with the same
  metrics separated by backbone; and performance_summary.png/pdf with training wall time,
  training throughput, holdout evaluation seconds per epoch, and allocated/reserved GPU
  memory. Every eval figure annotates training time, training tokens/s, mean holdout eval
  time per epoch, parameter count, memory, and tiny-quality status.
- Existing data retained: train_metrics.jsonl and epoch_metrics.jsonl remain the detailed
  per-step/per-epoch sources, while aggregate_comparison.json/csv/md indexes the new plots
  and the existing quality/evaluation summaries.
- Tests: report-generation py_compile and git diff --check passed; no model/data/training
  behavior was changed.
- Next task: stop; no next architecture phase.
