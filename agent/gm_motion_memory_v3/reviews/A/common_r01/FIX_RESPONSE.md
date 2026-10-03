# Writer response to common_r01

Reviewed commit: eec76455a7d9a36723e7c1a2aad1043bba553198.
Independent reviewer session: /root/common_review (fork_turns=none).
Original REVIEW.md is preserved verbatim.

R1: CLI now requires feature scales only for a positive/calibrated energy weight.
Observed-bond-only and later clash-only branches may omit unused ES scales.

R2: This task uses one CUDA process per arm. The new stratified CLI explicitly
rejects world_size != 1 before dataset/model loading, instead of claiming DDP
numerical equivalence from metadata sharding checks. The generic sampler still
supports sharding, but concurrent multi-rank numerical execution is not a
supported protocol for this release. A validated multi-rank auxiliary weighting
protocol remains future work, outside the one-arm-per-card run plan.

R3: Generated-bond calibration now modifies the already resolved loss_mapping,
preserving inherited clean/near/threshold terms and removing control-only keys.

No CUDA results are claimed. GPU availability remains a blocking prerequisite
for numerical smoke, calibration, pilot/profile and formal-candidate review.
