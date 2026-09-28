# Formal handoff gates

The GPU0 pilot is complete numerical/data evidence, not a selected M-short/M-long model. Do not promote pilot_model.pt to formal parent or use pilot loss to select history length.

Before formal integration, read the current A recipe_release and its evaluated immutable source/config/checkpoint identities. common_release is not recipe_release. Absorb only the declared fixed common commit; never edit A's worktree.

Resolve A's loss/sampler/metric/exposure policy once into both B arms. Keep the same anchor/future records, [-800,0]ps teacher anchors, parent, RNG streams, Q12 future and four-layer DiT+G. Only the authorized history extent differs. Native history dt is independent of query lag; 300ps is valid-only. Keep raw future outside conditioning and share the observed teacher origin.

The existing bounded pilot loop cycles27 conditions and is not the formal trajectory-capped sampler. Integrate raw-history windows with the released common sampler before formal launch; record paired-index hashes and optimizer module membership. Generic legacy train/inspect CLI is not yet a learned-memory artifact loader; do not point it at pilot checkpoints and assume compatibility.

Review the fixed integrated source with no implementation conversation and preserve the report. Resolve any findings and run targeted CUDA checks. Respect one B GPU slot and 48GPUh total / 8GPUh pilot budget, including failures/profile; GPU0 sharing was explicitly authorized but never permits preemption. Current costs are in train_manifest.json.

Then execute only M-short/M-long, preserving per-system motion and multi-lag evaluation, confidence intervals, teacher calls, peak memory and source/config identities. M-index is optional config/pilot only; no default third formal. No AF3 or long-rollout claim follows from this round.
