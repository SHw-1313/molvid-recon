# B implementation checkpoint

Integrated source candidate; NOT CUDA-validated. A common commit `eec76455a7d9a36723e7c1a2aad1043bba553198` was absorbed as `7270e54`. That release is an integration candidate, not formal approval.

Data flow: raw observed `[Th,N,3]` -> shared two-layer covalent scalar64/vector32 encoder -> residue tokens `[Th,R,64]` / `[Th,R,3,32]` -> ps quadrature/Haar analysis plus level/recent fallback -> K8 learned scalar128/vector64 slots -> DiT-width projection and local atom/group gather -> four-layer DiT with G -> per-frame h/v -> existing coordinate decoder.

- `prepare_frame_joint_batch` now dispatches learned_memory to separately supplied raw history, two anchors, and Q-only future. The old whole-clip teacher call is unreachable in B. Teacher receives two observed frames and Q targets at the same observed origin.
- `FrameJointModel.from_codec(history_backend='learned_memory', geometry_enabled=True, motion_enabled=False)` creates no legacy HistoryEncoder or MotionContextEncoder. Compatible DiT physical-time decay weights remain independently active.
- `FrameJointTrainer` owns all new modules through the history optimizer group. Warm-start report distinguishes loaded/new/retired tensors; no old history/M tensors are silently reused.
- Prepared memory remains differentiable and is computed once per prefix by the existing model call, shared across actual-source Euler branches. Inference uses `sample_learned_memory` with raw past, anchors and query times only.
- Raw data audit completed on 3 train systems / 9 replicas plus median/largest train topology profile inputs; report is under `runs/learned_memory_v1/raw_pilot_r01/data_card.json`.
- CUDA tests and bounded pilot entry exist. Pilot preflight refused to run because A resources.json assigns B no UUID. No GPU workload started; no CPU model fallback.

Remaining: CUDA smoke and full-teacher split parity, memory/DiT/decoder numerical gradients, real throughput/backward/graph-memory profile, 243-update pilot (full 27 history-span/dt/query-lag combinations x9 trajectories), fresh post-pilot review, corrections, and formal recipe integration. Current generic legacy train CLI does not ingest B raw banks; the bounded pilot uses the existing trainer directly. Formal exposure/config wiring waits for A's evaluated recipe and is not claimed complete.

M-index has configuration for a bounded pilot only. This work does not validate AF3 integration, complex generalization, mature motion quality, or long rollout.
