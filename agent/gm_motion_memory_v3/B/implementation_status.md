# B implementation checkpoint

GPU0 explicitly authorized by user; CUDA smoke6/6, real-data243-update pilot, and observed-only generation / differentiable K2 source checks passed. Actual artifacts: `runs/learned_memory_v1/cuda_pilot_r03/` and `execution_r01/`; source identities in report.md. No CPU model fallback.

Raw observed coordinates -> shared shallow local encoder -> chemical tokens -> physical ps scale analysis -> learned slots -> four-layer DiT+G -> per-frame future -> decoder. New backend skips the old whole-clip teacher path and retires old R4 history/M with explicit load report. Measured teacher2 observed +12 targets during training, teacher2 during inference; memory computed once per prefix with gradients preserved. All new modules are in history optimizer groups.

A fixed common commits eec7645 and5a745960 absorbed. Current generic legacy train CLI does not ingest B raw banks; the bounded pilot calls the existing trainer. Formal exposure/config wiring requires A evaluated immutable recipe and remains pending. Pilot weights are not formal parent weights.

Fresh independent review r03 and size/mixed-dt profile evidence are under the task report. M-index remains configuration/pilot only. No AF3, motion-quality, complex-generalization or long-rollout claim.
