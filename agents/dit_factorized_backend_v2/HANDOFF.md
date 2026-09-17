# B handoff

Status: NOT_STARTED

当前只有指令模板；没有运行数值测试或profile。

## 环境与输入

- worktree / branch / base / final commit:
- user changes preserved:
- enter-container invocation / mapped cwd / python / torch:
- GPU UUID / CUDA / BF16 / TF32:
- codec / DiT checkpoint / statistics hashes:
- reference source commit:

## 实现与兼容

- vectorized pooling/broadcast layout:
- actual attention backend/kernel:
- checkpoint contract/load treatment:
- duplicate encoder call test:
- cache validity / invalidation:

## 数值证据

逐项记录case、dtype、四field error、gradient/update error、容差、命令、退出码。注明非零attention gate、实际CUDA与skip情况。

## 性能表

填写R4 small/median/large、R2 median的N/K/M、weights/input hash、reference与optimized forward/backward/optimizer/end-to-end median/p90、tokens/s、warmup后peak、kernel与trace瓶颈。不要引用历史all-process peak代替新测量。

## 交付与建议

- semantic parity:
- speed/memory conclusion:
- unresolved synchronization/padding cost:
- exact commands / small evidence paths / full output paths:
- local commit:
- ready for integration or missing evidence:
- final status:

停在review，不合并A，不训练conditional prior。
## 2026-09-09 B execution evidence

Status: B_BACKEND_READY_FOR_REVIEW.

### Isolation and environment

- Worktree: /data4/users/sihao/workspace/molvid-dit-factorized-backend-v2
- Branch: perf/dit-factorized-backend-v2
- Base: a22f60c4ffd1f502ab300352a00eafe841b1f8d2
- A was checked read-only before creation; no A files, branch, evaluator, process, or checkpoint was modified. No push, merge, or rebase was performed.
- All Python commands ran through enter-container with torch-ito. Runtime was torch 2.5.1+cu121 / CUDA 12.1.
- GPU audit selected physical GPU 5: GPU-f0bea51c-913e-fe29-1cdd-4cde2adb3ff3, NVIDIA A100-SXM4-80GB, 0 MiB and no process at start. The exact UUID recorded by nvidia-smi was GPU-f0bea51c-913e-fe29-1cdd-4cde2adb3ff3.
- TF32 was disabled. Equivalence used FP32; profile used BF16 autocast. The frozen manifest was train/valid only; test clip storage was not opened.

### Implementation

- Reference remains the explicit execution path from source commit a22f60c4ffd1f502ab300352a00eafe841b1f8d2. The optimized attention backend is matmul_softmax_shared_weights_v2: scalar q/k logits and one shared softmax matrix for scalar and xyz-vector values. No SDPA or FlashAttention claim is made.
- FactorizedLayout materializes sample/block groups once from batch metadata, uses (sample_id, block_id) keys, CUDA index_add_ mean pooling, padded BxK block attention, and vectorized per-atom broadcast. Temporal attention is one atom-by-K call. AdaLN, SO3 normalization, FFN, RF objective, widths, depth, heads, and strict state_dict contract are unchanged.
- Runtime backend selection is explicit through execution_backend and execution_contract; factorized_v2 rejects nonzero dropout. Derived layout is not serialized.
- Cache validity uses metadata object identity and tensor version counters; CUDA tests cover cache reuse and topology mutation invalidation. The pilot training loop reuses the _encode_batch latent and no longer calls codec.model.encode(batch) a second time.
- R2/R4 model parameter count is 11,173,120. Checkpoint/model/statistics hashes and all profile input/model hashes are in SUMMARY.json.

### Numerical evidence

- CUDA command exited 0: 4 backend-v2 tests passed, including FP32 forward/gradient/update, BF16, ragged/padding/full-mask, sample/block collision isolation, SO(3), observation masks, topology cache invalidation, and 16-step cache reuse.
- Affected regression exited 0: 15 tests passed across molecular DiT, pilot runner, and trainer tests.
- Real checkpoint equivalence exited 0 for R2 and R4. R2 forward max abs 2.7656555e-5, relative L2 5.0101e-7, gradient max abs 8.0618e-9, AdamW update max abs 1.1620e-5. R4 forward max abs 1.2815e-5, relative L2 4.6887e-7, gradient max abs 4.5227e-8, AdamW update max abs 1.5281e-5. Nonzero AdaLN modulation was present (max abs 0.1456 R2 / 0.1221 R4).
- Real smoke exited 0: R2 and R4 each ran 8 Euler steps, decoded to [16,4194,3], all finite, and observed clamp max abs was 0.0.
- The equivalence/profile artifacts and compact evidence are under evidence/dit_factorized_backend_v2/20260909_backend_v2/SUMMARY.json and SUMMARY.md. Raw runtime output was /tmp/molvid_dit_factorized_backend_v2/20260909_backend_v2 in the torch-ito container namespace.

### Bounded profile

Profile used 5 warmup + 20 measured optimizer steps for each backend and each train-only case: r4_small, r4_median, r4_large, r2_median. Total optimizer steps were exactly 200. CUDA peak statistics were reset after warmup; the preloaded frozen codec remains in the reserved-memory baseline.

| case | N/B/M | backend | total median/p90 ms | forward/backward/optimizer median ms | end-to-end median ms | allocated/reserved MiB |
|---|---:|---|---:|---:|---:|---:|
| R4 small | 3869/3/264 | reference | 7460/7751 | 1537/5823/4.71 | 8845 | 3879/22042 |
| R4 small | 3869/3/264 | factorized_v2 | 100/110 | 37/52/3.76 | 1485 | 4085/22042 |
| R4 median | 4194/2/264 | reference | 7651/8812 | 1695/6013/3.99 | 8400 | 4220/24398 |
| R4 median | 4194/2/264 | factorized_v2 | 108/120 | 41/55/4.37 | 856 | 4295/24398 |
| R4 large | 4430/5/119 | reference | 8700/8986 | 1857/6751/3.95 | 9543 | 4370/24782 |
| R4 large | 4430/5/119 | factorized_v2 | 86/98 | 33/48/3.39 | 929 | 4460/24782 |
| R2 median | 4194/2/264 | reference | 15105/15432 | 3044/11922/3.87 | 15899 | 8287/24452 |
| R2 median | 4194/2/264 | factorized_v2 | 106/111 | 42/64/0.86 | 900 | 8438/24452 |

The model-only median speedups are approximately 75x, 71x, 101x, and 142x for R4 small/median/large and R2 median. The main measured bottleneck is the reference per-time/per-sample/per-block Python spatial path; the optimized path leaves explicit attention, FFN, optimizer, and padding work. No broad claim is made beyond these bounded measurements.

### Completion

- No production checkpoint or large binary artifact was written or staged.
- No formal scientific training, 4500-step run, conditional-prior model, evaluator change, or test-split read was performed.
- Implementation commit: efe0d39 (local only; no push).
- Stop at operator review; do not merge A or push.
