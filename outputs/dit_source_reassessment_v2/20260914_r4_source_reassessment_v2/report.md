# R4 source checkpoint reassessment v2

Status: `A_SOURCE_REASSESSMENT_V2_PARTIAL_NO_CUDA_EVALUATION`

This is a descriptive `shared_adapter_legacy_diagnostic`; the Gaussian/conditional checkpoint difference is not an independent source ablation.
The frozen R4 codec/statistics, DiT/RF semantics, train/validation manifests, and test-sealed boundary were preserved.

## Inputs and selection

- Code HEAD: `f9a23b2c01e1c4ea39342bd114b3f43c9fedabaf`; required baseline: `5c2754fcce44ed77dad77db09db709408fee7634`.
- Legacy JSONL row check by arm: `True`; per-arm counts `{'conditional': {'main_batch': 52, 'subset_clip': 128}, 'gaussian': {'main_batch': 52, 'subset_clip': 128}}` vs expected `{'main_batch': 52, 'subset_clip': 128}`. Main batch rows were never split into fake systems.
- Validation selection: `8` systems, fixed R1/window30; train selection: `8` sorted systems, one preselected clip each.
- Test payload opened: `false`.

## Existing JSONL reaggregation

Legacy main rows are reported as batch-row means with `system_equal_not_available`. Legacy subset rows are averaged draw → clip → system, separately for H and Euler steps.

- `conditional`: {'main_batch': 52, 'subset_clip': 128} rows; checkpoint SHA256 `8a3f8803655406468b8d50ca5a316f4a7858be31366f74bc2d897ca96769b12c`.
- `gaussian`: {'main_batch': 52, 'subset_clip': 128} rows; checkpoint SHA256 `545a64f34ad6a2463a0a36e14a53c09f6c1e0fe6105ea6be9e30a34e84582c95`.

## New single-segment results

CUDA contract check: `None`.

No CUDA evaluation artifact exists. Numerical sampling was not run; no CPU fallback was used. See `preflight.json.gpu_inventory_at_preflight` for the confirmed occupied UUID/PID list.

## Short rollout

The required two-system H8 smoke was not run.
The eight-system rollout was not run.

## Reproduction

```bash
enter-container  # then: conda activate torch-ito
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage preflight
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage plan
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage reaggregate
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage cuda_check --device cuda:IDLE
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage evaluate --device cuda:IDLE
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage rollout_smoke --device cuda:IDLE
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage rollout --device cuda:IDLE
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2.yaml --stage summarize
```

Plot: `not generated`.
