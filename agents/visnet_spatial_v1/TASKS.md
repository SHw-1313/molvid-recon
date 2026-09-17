# TASKS — ViSNet spatial backbone v1

Status legend: `[ ]` pending, `[-]` active, `[x]` complete, `[!]` blocked.

## Preparation

- [x] V00 create `feat/visnet-spatial-v1` from `fix/graph-runtime-v1`
- [x] V01 record base commit, worktree status, environment, PyTorch/PyG/CUDA versions
- [x] V02 record current TorchMD ratio-1 temporal baseline protocol and artifact locations
- [x] V03 confirm phase scope and do not modify temporal/decoder/loss implementations

## Spatial abstraction

- [x] V10 add `FrameNodeBatch` and separate frame-node packing from external graph construction — `FrameNodeBatch`/`pack_frame_nodes` are implemented in `module/multiframe_codec.py`.
- [x] V11 retain `build_graph(batch)` as a compatibility wrapper — external backbones retain the wrapper; native radius explicitly rejects external construction.
- [x] V12 add spatial-backbone factory/config parsing with legacy TorchMD default — factory, model config, YAML default, train/eval parsing are implemented.
- [x] V13 add common spatial output contract and runtime metadata — `SpatialEncoderOutput`, `FrameEncoderOutput`, and graph/backbone metadata are implemented.

## ViSNet implementation

- [x] V20 implement representation-only, block-aware `MolViSNetEncoder` — scalar/vector representation block only, with atom and block embeddings.
- [x] V21 preserve upstream license and record exact source/version used — attribution in `module/visnet.py` cites torchmd-net v2.0.0 and MIT licensing.
- [x] V22 implement `visnet_radius` native one-build radius graph path — native CUDA/dense-test path constructs one graph inside `forward_native`.
- [x] V23 prove native path bypasses external topology/graph construction — hard counting/monkeypatch test passes.
- [x] V24 implement `visnet_bonded` on the repaired external graph — consumes `FrameGraphBatch` without a second graph build.
- [x] V25 inject binary covalent edge type into ViSNet edge features — binary edge embedding is part of the radial edge features.
- [x] V26 reject unsupported `lmax=2` clearly — constructor error names the `lmax=1` contract.
- [x] V27 support static `T=1`, dynamic `T=16`, BF16-capable modules, and checkpoint resume — shape, CUDA, contract, and existing resume tests pass; coordinates remain FP32 under feature dtype/autocast.

## Tests

- [x] V30 factory and backward-compatibility tests — new spatial tests plus existing suite pass.
- [x] V31 scalar/vector shape tests — `[T,N,C]` and `[T,N,3,C]` assertions pass for both ViSNet variants.
- [x] V32 no cross-frame/cross-sample edge tests — graph-id isolation assertions pass.
- [x] V33 native-no-double-graph test — native neighbor count is exactly one and external builder/topology hooks are forbidden.
- [x] V34 scalar invariance/vector equivariance tests — translation and rotation checks pass.
- [x] V35 finite/nonzero coordinate and parameter gradient tests — both ViSNet variants pass.
- [x] V36 bonded edge-type sensitivity test — changing covalent topology changes the representation.
- [x] V37 static `T=1` and trajectory `T=16` tests — both paths pass.
- [x] V38 checkpoint contract and resume tests — resolved backbone/graph metadata is round-trippable.
- [x] V39 full existing relevant test suite passes — `67 passed, 1 warning` via `torch-ito`.

## Micro-overfit

- [x] V40 parameterize the existing three-clip runner by spatial backend — runner now serializes all three backbones with the fixed 500-step protocol.
- [x] V41 run 500 steps for `torchmd_et` — remote A100 run reduced aggregate total loss by 81.2178%; resume reached step 501.
- [x] V42 run 500 steps for `visnet_radius` — remote A100 run reduced aggregate total loss by 68.9975%; resume reached step 501.
- [x] V43 run 500 steps for `visnet_bonded` — remote A100 run reduced aggregate total loss by 62.2703%; resume reached step 501.
- [x] V44 verify finite losses, gradients, >=30% total-loss reduction, and resume — all three backbones passed the strict final-loss-below-70%-of-initial gate and checkpoint resume.

## Final tiny-overfit

- [x] V50 build and hash the 441/117 manifest — local preflight hash `3897187ee968a20c1ed359177c0de4e4ed436998fe828f6be82b3017e6c91963`; authoritative remote execution hash `e6ead995dbea156c70933dc0f9dde3e93e5411b78f37aeae2fe099890b301b76` (same ordered IDs, different compact-store byte offsets).
- [x] V51 assert 3 systems, 9 replicas, 441 train clips, 117 holdout clips, and zero overlap — preflight returned `441 117 0`.
- [x] V52 implement lazy no-replacement exact-coverage loader — tiny runner uses lazy mmap, `replacement=false`, and validates every epoch’s complete ID set; 30-epoch execution completed.
- [x] V53 run 30 epochs for `torchmd_et` — remote A100 run completed 30 epochs/4979 steps and passed the tiny quality gate.
- [x] V54 run 30 epochs for `visnet_radius` — remote A100 run completed 30 epochs/4979 steps; execution passed, tiny quality gate was false because train loss reduction was 49.3922% (<60%).
- [x] V55 run 30 epochs for `visnet_bonded` — remote A100 run completed 30 epochs/4979 steps; execution passed, tiny quality gate was false because train loss reduction was 49.5524% (<60%).
- [x] V56 verify every epoch has no missing or duplicate train sample IDs — all 30 epoch records for each backend report exact 441-sample coverage and 441 unique IDs.
- [x] V57 evaluate all 117 holdout clips after every epoch — all 30 epoch records per backend report exact 117-clip coverage and per-system 39-clip metrics.
- [x] V58 verify checkpoint save/resume for all backends — all three saved step 4979 checkpoints and resumed at step 4980.

## Summary and stop

- [x] V60 generate per-backbone JSON/CSV/Markdown/plots — aggregate JSON/CSV/Markdown, loss curves, per-backbone eval figures, and the runtime-performance PNG/PDF figures were generated for the completed run.
- [x] V61 report parameter, quality, throughput, memory, and topology-bond-coverage diagnostics — per-backbone result JSON and aggregate report include these diagnostics.
- [x] V62 classify implementation pass and tiny-overfit quality pass separately — implementation and 500-step micro gates passed for all three; only `torchmd_et` passed the strict tiny quality gate.
- [x] V63 record recommended development backend with caveats — recommendation is `torchmd_et`; both ViSNet variants remain implementation-valid but failed the tiny quality gate under this protocol.
- [x] V64 update HANDOFF with exact commands, evidence, blockers, and next phase — final dated evidence appended below.
- [x] V65 stop; do not begin state-detail or VideoDiT work — no next architecture phase was started.
