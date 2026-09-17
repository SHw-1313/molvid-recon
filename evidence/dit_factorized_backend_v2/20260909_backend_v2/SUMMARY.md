# DiT factorized backend v2 review summary

Status: B_BACKEND_READY_FOR_REVIEW

Worktree: /data4/users/sihao/workspace/molvid-dit-factorized-backend-v2
Branch: perf/dit-factorized-backend-v2
Base: a22f60c4ffd1f502ab300352a00eafe841b1f8d2
GPU: physical index 5, GPU-f0bea51c-913e-fe29-1cdd-4cde2adb3ff3, NVIDIA A100-SXM4-80GB
Runtime: torch 2.5.1+cu121 / CUDA 12.1; TF32 disabled; equivalence FP32; profile BF16 autocast

Gates:
- CUDA backend gates: 4 passed.
- Affected regression: 15 passed.
- Real R2/R4 FP32 forward/gradient/AdamW parity: passed.
- Real R2/R4 8-step generation and codec decode: finite [16, 4194, 3], exact observed clamp.
- Bounded profile: 200 optimizer steps, 5 warmup + 20 measured for each of four cases and two backends.
- No production checkpoint, conditional-prior model, test split, or formal scientific training was created.

Profile timings are CUDA-event medians/p90s. The two backends share each case input and initial model state; end-to-end adds one frozen codec encode. N/B/M means total atoms / batch size / max blocks.

| case | backend | N/B/M | total median/p90 ms | forward ms | backward ms | optimizer ms | end-to-end ms | atom-frame tok/s | latent atom-time tok/s | allocated/reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 small | reference | 3869/3/264 | 7460/7751 | 1537 | 5823 | 4.71 | 8845 | 8.3k | 2.1k | 3879/22042 |
| R4 small | factorized_v2 | 3869/3/264 | 100/110 | 37 | 52 | 3.76 | 1485 | 628.0k | 157.0k | 4085/22042 |
| R4 median | reference | 4194/2/264 | 7651/8812 | 1695 | 6013 | 3.99 | 8400 | 8.4k | 2.1k | 4220/24398 |
| R4 median | factorized_v2 | 4194/2/264 | 108/120 | 41 | 55 | 4.37 | 856 | 627.6k | 156.9k | 4295/24398 |
| R4 large | reference | 4430/5/119 | 8700/8986 | 1857 | 6751 | 3.95 | 9543 | 8.1k | 2.0k | 4370/24782 |
| R4 large | factorized_v2 | 4430/5/119 | 86/98 | 33 | 48 | 3.39 | 929 | 807.9k | 202.0k | 4460/24782 |
| R2 median | reference | 4194/2/264 | 15105/15432 | 3044 | 11922 | 3.87 | 15899 | 4.4k | 2.2k | 8287/24452 |
| R2 median | factorized_v2 | 4194/2/264 | 106/111 | 42 | 64 | 0.86 | 900 | 623.7k | 311.9k | 8438/24452 |

Model-only median speedups are approximately 75x, 71x, 101x, and 142x for R4 small/median/large and R2 median. The reference bottleneck is the old per-time/per-sample/per-block Python spatial path. The optimized attention is explicit matmul/softmax with shared scalar weights; no SDPA or FlashAttention claim is made. Reserved memory includes the preloaded frozen codec; allocated peak was reset after warmup.

Reproduction commands were run inside enter-container with torch-ito and CUDA_VISIBLE_DEVICES=5:
- CUDA tests: CUDA_VISIBLE_DEVICES=5 DIT_RUN_CUDA_BACKEND_V2=1 PYTHONPATH=.:tests python -m pytest -q -p no:cacheprovider tests/test_dit_backend_v2.py
- Regression: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.:tests python -m pytest -q -p no:cacheprovider tests/test_molecular_dit.py tests/test_dit_pilot_runner.py tests/test_dit_trainer.py
- Stages: python scripts/profile_dit_backend.py --config config/dit_backend_profile_v2.yaml --stage preflight|equivalence|smoke|profile --device cuda:0 --output-root /tmp/molvid_dit_factorized_backend_v2

Raw runtime output was /tmp/molvid_dit_factorized_backend_v2/20260909_backend_v2. The compact JSON and this Markdown report are the committed evidence; no binary artifacts were copied.
