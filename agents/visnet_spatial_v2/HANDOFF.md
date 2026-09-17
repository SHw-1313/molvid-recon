# HANDOFF — ViSNet spatial backbone v2

Status: prepared; review/fix worker has not started.

## Starting point

```text
source branch: feat/visnet-spatial-v1
source commit: bca9ea98df643f4965b009688f2967bd669e7fe2
target branch: feat/visnet-spatial-v2
```

The v1 implementation, tests, artifacts, and result semantics must remain reproducible. V2 adds
new backend names and must not overwrite the v1 run directory.

## Prepared scientific diagnosis

The v1 implementation is stable and equivariant but is not reference-faithful ViSNet. In
particular, it lacks reference Sphere/NeighborEmbedding/EdgeEmbedding, persistent edge updates,
vector rejection, real vertex geometry, and VecLayerNorm. It also replaces ViS-MP attention with
incoming-edge softmax and averages attention heads before value aggregation.

The v1 `vertex=True` and `vecnorm_type` fields do not control the corresponding original ViSNet
operations. The v2 parity and no-no-op gates are specifically intended to prevent a repeat.

The dataset contains unique `block_id`, but the current spatial path consumes `btype`. V2 retains
that type embedding only; EPT-style hierarchy remains out of scope.

## Immutable v1 tiny evidence

Run directory:

```text
outputs/visnet_spatial_v1/tiny_overfit/run_20260901T192557
```

Protocol:

```text
441 train clips
117 late-holdout clips
T=16
dt=100 ps
30 epochs / 4979 steps
FP32
temporal_layers=1
temporal_ratio=1
```

Final core metrics:

| Backbone | Train total | Holdout future dRMSD | Future RMSD | Velocity RMSE | Frequency retention |
|---|---:|---:|---:|---:|---:|
| `torchmd_et` | 0.326502515 | 0.644799669 | 0.873626672 | 0.004180777 | 0.670033 |
| `visnet_radius` | 0.379553097 | 0.682236567 | 0.932992843 | 0.004453239 | 0.636193 |
| `visnet_bonded` | 0.374939702 | 0.679827729 | 0.928350341 | 0.004442983 | 0.619523 |

The worker must independently read and hash the exact manifest/protocol/result files before
reusing these values.

### 2026-09-02 — V200-V204 preparation and parity audit

- Status: complete.
- Worktree before mutation: `/data4/users/sihao/workspace/PVB`, branch `feat/visnet-spatial-v1`, HEAD `bca9ea98df643f4965b009688f2967bd669e7fe2`, origin `https://github.com/SHw-1313/molvid.git`; only the user prompt and v2 phase files were untracked. The target branch was created without reset, checkout-over, clean, or deletion.
- Root `AGENTS.md` was updated for the v2 phase. No v1 phase document or cumulative historical agent file was edited.
- AI2BMD reference: commit `497efaa190ee6f6cbc6030710c44208a01ece52d`, MIT license SHA256 `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383`, `visnet/models/visnet_block.py` SHA256 `67ed05a2fcb746460e0783e7f0abb74fd79adb8b0b6ad3977fa8a27270c30eba`, `visnet/models/utils.py` SHA256 `590eebb53061564940aad89b195f31ba78283aa301fed8cde2bc803b08fe3a1e`.
- PyG cross-check: commit `79d33965a40b7fa83616a9f598a0f8619f25d939`, MIT license SHA256 `89cfb6edc309735916a0c1189dccf761add4af6062ddb34b5a26f60d3efadfea`, `torch_geometric/nn/models/visnet.py` SHA256 `f8005ec06fb70730a16e81c0be888eddb6e2dddff8eb1f1e832bd3c31043e561`.
- Immutable v1 artifact SHA256 values: `manifest_contract.json` `21992d86626eaf3b6d08dea12354b65060ec0be54b64c0adf9dc15b18f315a40`; `protocol.json` `d98805fee37c05370bc8fd52fd49c7f28e05532ebdc7366088aa0cc458263564`; `torchmd_et/result.json` `c7d36b57e1eb663b04b82fb0620e80599e6b8209ff16c2d2fb3e78aed7475067`; `visnet_bonded/result.json` `2e408514e90c116a066f59ef2e300552daceda3f1b646a632d04cd9ca23df5b0`; aggregate report `8c1b953214e0c37794496d88523bf4f55e6686003c440ad7f78ae8d7fa3f7cab`.
- The v1/reference audit is complete in `REFERENCE_PARITY.md`. All eight requested discrepancy diagnoses were confirmed; no new decision was forced before implementation.
- Next task: V210-V214, faithful primitive implementation and deterministic pinned-reference fixtures.

## Authorized reference checkouts

```text
/data4/users/sihao/workspace/AI2BMD_visnet_ref
  https://github.com/microsoft/AI2BMD.git
  branch ViSNet
  commit 497efaa190ee6f6cbc6030710c44208a01ece52d

/data4/users/sihao/workspace/pytorch_geometric_visnet_ref
  https://github.com/pyg-team/pytorch_geometric.git
  commit 79d33965a40b7fa83616a9f598a0f8619f25d939
```

This is a narrow user-authorized Git reference fetch. No dependency installation or unrelated
network use is authorized.

## Evidence update template

Append; do not rewrite previous entries.

```markdown
### YYYY-MM-DD — Vxxx-Vyyy

- Status:
- Branch/base/current commit:
- Worktree before/after:
- Reference repositories and exact hashes:
- Files changed:
- Commands:
- Tests and parity tolerances:
- Fixture/config/input/state/output hashes:
- Geometry sensitivity evidence:
- Data/manifest/protocol hashes:
- Training/evaluation metrics:
- Runtime/memory:
- Checkpoints/resume:
- Decisions/deviations:
- Blockers:
- Next task:
```

## Final required handoff

Include:

- complete changed-file list;
- reference commits, licenses, symbol mapping, and parity fixture hashes;
- proof that `vertex_type`, `vecnorm_type`, and lmax are not no-ops;
- angle/dihedral and edge-evolution evidence;
- lmax-2 internal/public adapter evidence;
- v1/TorchMD regression evidence;
- exact environment and commands;
- micro and tiny absolute metrics;
- gap-recovery analysis;
- outcome classification;
- unresolved risks and multi-seed needs;
- explicit statement that no later architecture phase began.

### 2026-09-02 — V210-V256 Gate 0 and Gate 1 evidence

- Status: complete. The reference-faithful v2 primitives, ViS-MP variants, adapters, contracts,
  and discriminating tests are implemented in a new `module/visnet_v2.py`; the v1 production
  path remains available and its stored artifacts were not overwritten.
- References: AI2BMD `497efaa190ee6f6cbc6030710c44208a01ece52d`, MIT license SHA256
  `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383`; PyG
  `79d33965a40b7fa83616a9f598a0f8619f25d939`, MIT license SHA256
  `89cfb6edc309735916a0c1189dccf761add4af6062ddb34b5a26f60d3efadfea`. Source hashes and
  symbol mapping are recorded in `REFERENCE_PARITY.md` and `IMPLEMENTATION_EVIDENCE.json`.
- Gate 0 commands, all run through `enter-container`/`torch-ito`: v2 specialized tests `22
  passed`; full suite `89 passed, 1 warning`; relevant modules/scripts `py_compile`; and
  `git diff --check`. Parity target was FP32 `rtol=2e-5`, `atol=2e-6`; deterministic primitive,
  base, vertex-edge, vertex-node, and full-block local/reference maximum absolute differences
  were zero. Full-block input/state/output hashes are recorded in the parity ledger.
- Gate 0 geometry evidence: fixed explicit-edge-distance angle and dihedral fixtures respond to
  coordinate changes; non-final edge state evolves while the final layer does not update it;
  vector rejection, `vertex_type=none|edge|node`, `vecnorm_type=none|rms|max_min`, trainable
  vector norm, lmax-1/lmax-2 shapes, SE(3), finite/nonzero geometry gradients, native one-graph,
  and external no-rebuild tests pass. lmax=2 carries `[M,8,C]` internally and exports `[M,3,C]`.
- Gate 1 locked protocol: seed `20260902`, three fixed clips, FP32, 500 optimizer updates,
  `max_tokens=80000`, and checkpoint step-500/step-501 resume. Local run:
  `outputs/visnet_spatial_v2/micro_overfit/run_20260902T170456`; remote lmax-2 run:
  `outputs/visnet_spatial_v2/micro_overfit/neibu_run_20260902T174649`. The initial lmax-2 packed
  batch exceeded the local GPU; the rerun used only clip-wise logical gradient accumulation in
  the v2 micro runner, preserving the three-clip logical batch and step count.
- Gate 1 absolute results: `torchmd_et` initial/final total `1.656761527/0.322974653`, reduction
  `80.51%`, `0.9363` steps/s, peak allocated/reserved `55.00/58.41 GiB`; v1 `visnet_bonded`
  `1.551997066/0.569661796`, reduction `63.29%`, `1.1227` steps/s, `35.31/36.05 GiB`; v2
  lmax-1 `1.542722464/0.330271482`, reduction `78.59%`, `0.6339` steps/s,
  `69.14/72.23 GiB`; v2 lmax-2 `1.933403770/0.335912456`, reduction `82.63%`, `0.2591`
  steps/s, `62.68/66.13 GiB`. Both v2 variants have finite/nonzero geometry gradients and
  passed checkpoint resume. Result/checkpoint hashes are in the run directories; remote lmax-2
  result SHA256 is `6cd47e0e3ce84393c2c1cc0f052d2b48c591385358c2eb01594f162fe4dbb487` and
  checkpoint SHA256 is `5fd972515bf9cb6f74a45c7ce02d35990b987ef21b7ae1368e17f5f7fd800122`.
- Selection: v2 lmax-1 is the development configuration by absolute final loss first
  (`0.330271482 < 0.335912456`), then runtime and memory. This is not a claim that it is the
  final scientific winner; the exact tiny comparison remains pending.
- Next task: V260, immutable tiny manifest/protocol comparability and selected v2 bonded run.

### 2026-09-02 — V260-V263 exact tiny comparison

- Status: complete. The isolated `neibu` worker completed the selected `visnet_v2_bonded`
  lmax-1/vertex-edge configuration under the locked tiny protocol: 441 train clips, 117 late
  holdout clips, lazy loading, `replacement=false`, exact epoch coverage, T=16, dt=100 ps,
  FP32, `max_tokens=80000`, 30 epochs, and a 6000-step safety cap. The run reached step 4979
  and checkpoint resume reached step 4980.
- Comparability: source-manifest, train-index, valid-index, all sample IDs, counts, window
  ranges, and zero train/holdout overlap match the immutable v1 run. The only mismatch is the
  builder's `built_manifest_sha256`, which includes absolute source paths; the worker path is
  different. This is recorded explicitly in the report rather than treated as a silent match.
  The immutable v1 TorchMD, v1 radius, and v1 bonded rows were reused; no v1 file was changed.
- v2 absolute metrics: initial train total `0.781103878`, final train total `0.335397243`,
  reduction `57.0611%`; initial holdout future dRMSD `1.060634814`, final `0.646277387`,
  improvement `39.0669%`; final future RMSD `0.879702182`, future bond RMSE `0.160754050`,
  velocity RMSE `0.004224300`, and frequency retention `0.678905182`. Catastrophic dynamic
  regressions were absent and all metrics were finite. The tiny quality gate is **false** only
  because the required train-total reduction is below 60%; the future dRMSD gate passes.
- Gap recovery: relative to v1 bonded (`0.679827729`) and TorchMD (`0.644799669`), v2 future
  dRMSD recovery is `95.7813%`. V2 beats v1 bonded on both future dRMSD and future RMSD, and
  is within 10% of TorchMD on both, but the allowed outcome classification remains
  **IMPLEMENTED** because the locked tiny quality gate failed.
- Runtime: training elapsed `10253.1476 s` (`170.9 min`), `33645.1` tokens/s, holdout eval
  `57.65 s/epoch`, peak allocated/reserved `35.51/37.15 GiB`. External graph diagnostics were
  14192 nodes, 322498 distance edges, 322499 union edges, and 29216 replicated covalent edges.
- Artifacts: remote pre-normalization v2 result SHA256 `e46ff2c23c785acdc8b9b3dc21c2fdf5ca77e0903edb9a989a3fe027473595aa`, checkpoint SHA256
  `7b94e1c437123b1d2322f950308e9892cefecc56ac407476eb819b5b431509ed`. The copied local run
  is `outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930`; its aggregate JSON/CSV/Markdown and loss/eval/performance PNG/PDF reports were regenerated locally with runtime annotations.
- The active-radius task is V264; no later architecture phase has begun.

### 2026-09-02 — V264-V273 final evidence and stop

- Status: complete; this is the final handoff for ViSNet spatial backbone v2. No subsequent
  architecture phase was started.
- Completed task IDs: V200-V204, V210-V214, V220-V226, V230-V236, V240-V244, V250-V256,
  V260-V264, and V270-V273.
- Changed source files: `AGENTS.md`, `module/__init__.py`, `module/multiframe_codec.py`,
  `module/visnet.py`, `module/visnet_v2.py`, `trainer/codec_trainer.py`,
  `tests/test_visnet_spatial_v2.py`, `scripts/run_visnet_spatial_v2_micro_overfit.py`, and
  `scripts/run_visnet_spatial_v2_tiny_overfit.py`. Changed phase evidence files:
  `agents/visnet_spatial_v2/REFERENCE_PARITY.md`, `TASKS.md`, `HANDOFF.md`, and
  `IMPLEMENTATION_EVIDENCE.json`. The user-owned prompt file remains unmodified; v1 phase
  documents, checkpoints, and result files remain untouched.
- Exact final validation commands, run through `enter-container` with `torch-ito`: `PYTHONDONTWRITEBYTECODE=1 python -m py_compile module/visnet_v2.py module/visnet.py module/multiframe_codec.py module/__init__.py trainer/codec_trainer.py tests/test_visnet_spatial_v2.py scripts/run_visnet_spatial_v2_micro_overfit.py scripts/run_visnet_spatial_v2_tiny_overfit.py`; `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_visnet_spatial_v2.py`; `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider`; and `git diff --check`. The Gate 0 result was 22 specialized tests passed and 89 full-suite tests passed with one pre-existing `torch.load` warning. Parity target was FP32 `rtol<=2e-5`, `atol<=2e-6`; observed deterministic fixture maximum absolute difference was zero.
- Reference/fixture provenance: AI2BMD commit `497efaa190ee6f6cbc6030710c44208a01ece52d`, PyG commit `79d33965a40b7fa83616a9f598a0f8619f25d939`; both MIT. License/source hashes and full-block lmax-1/lmax-2 input/state/output hashes are in `REFERENCE_PARITY.md` and `IMPLEMENTATION_EVIDENCE.json`.
- Evidence distinction: implementation pass = **yes** (Gate 0 parity/contracts/graph isolation); micro-overfit pass = **yes** for v2 bonded lmax-1, v2 bonded lmax-2, and v2 radius (finite/nonzero geometry gradients, >=30% reduction, checkpoint/resume); tiny-overfit quality pass = **no** for selected v2 bonded lmax-1 because train-total reduction was `57.0611%`, below the locked `60%` threshold, while future dRMSD improvement and dynamic stability passed.
- Final outcome classification: **IMPLEMENTED**. The selected v2 bonded lmax-1 final tiny future dRMSD/RMSD were `0.646277387/0.879702182` versus v1 bonded `0.679827729/0.928350341` and TorchMD `0.644799669/0.873626672`. Distant from v1 to TorchMD, v2 recovered `95.7813%` of the future dRMSD gap, beat v1 bonded on both future geometry metrics, and stayed within 10% of TorchMD on both. This does not override the failed locked train-loss gate.
- Development-backend recommendation: `visnet_v2_bonded`, lmax-1, `vertex_type=edge`, `rbf_type=expnorm`, `vecnorm_type=max_min`, pending multi-seed confirmation. It is the absolute-metric-selected v2 geometry configuration and avoids the lmax-2 runtime/memory cost. The v2 radius backend is implementation-valid and passed its bounded micro, but it was not run for a second 30-epoch tiny comparison because no material ambiguity required it.
- Runtime/performance summary: selected tiny v2 bonded took `170.9 min`, `33.6k tokens/s`, `57.7 s/holdout epoch`, and `35.51/37.15 GiB` allocated/reserved; v2 radius bounded micro took `973.65 s`, `0.5135 steps/s`, and `35.37/36.98 GiB`, with native radius graph mode and zero binary covalent features. The final annotated eval graph is `outputs/visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/eval_metrics_comparison.png` (and PDF); loss and performance graphs are in the same directory.
- Final artifact hashes: tiny manifest `f3ee81f99543ea6a4c6701c6fe390d77516635421521bd19ee2c136e6d2f32fc`; tiny protocol `dcb9c0dc892e95fec5c3510f0f9a4f2428cb834943ff7f9107233a0a16bff418`; tiny v2 result `9fedb7a53bb366c5a7e9eda708647a136e35644bc0130a4a172ea6df015d75e9`; tiny checkpoint `7b94e1c437123b1d2322f950308e9892cefecc56ac407476eb819b5b431509ed`; aggregate JSON/CSV `726bc5545d1b49bf1c81b86e810cab21df5e92f20a645093eae2a288d18c8ef8`/`f9236e714c6867e5fba72969c5e3a31ab58623b28c6394670f5f2f0e6a4ebbf3`; eval PNG/PDF `cdb8712e1cbbc7d2ea8299789ea842c132a0a2215651e6a63907b68c933df528`/`a0620cde00c11d18e32763ccb7af2f938c3d132187b1cab4dd5aaf410a5a0e6a`; radius result/checkpoint `78add57436dc72c54d55218c113012b1e8c371f6b70720fbba68de5d6eef57e5`/`46b33da70aaade185815886c356b2416b7e6b9f6c89ab1a669206527d089cd15`.
- Limitations: one seed; natural rather than parameter-matched models; logical clip-wise accumulation was required to bound the v2 tiny CUDA graph peak and is explicitly recorded; v2 radius has bounded micro evidence only; multi-seed and a larger scientific benchmark are still needed. No packages were installed/upgraded and no unauthorized network or dataset download occurred.
- Stop condition: implementation, parity, micro, exact tiny, reports, and handoff are complete. Do not begin AF3, state-detail, decoder/trunk, rollout, scaling, or full-data work in this phase.
