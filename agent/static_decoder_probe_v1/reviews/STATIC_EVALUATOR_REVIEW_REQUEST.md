# Held-out static evaluator review request

- Parent commit: `e64bd525d87a85e3c53a6288cce061808b455210`
- Candidate commit: `8792b17` (`git rev-parse` supplies the full identity)
- Scope: `tools/evaluate_static_decoder.py` and
  `tests/test_static_decoder_evaluator.py` only.
- Reviewer mode: independent, read-only, no model-code or artifact edits.

## Intended protocol

Use the already-declared ordinary validation manifest, never sealed test data.
Select `anchor_window_300=0` exactly once for each of 8 systems x 3 replicas
from its declared source validation store, then convert the middle valid frame
to an explicit `T=1` static task. Evaluate clean reconstruction and Gaussian
denoising at per-axis sigma 0.05, 0.10 and 0.20 Angstrom.

Each record/sigma noise seed is a hash of the fixed base seed, source record,
frame and sigma. Independent arm invocations must therefore receive identical
input structures and perturbations. Clean coordinates stay loss targets only.
The diagnostic calls the same frozen teacher, shared decoder, coordinate MSE,
clean-target bond loss and released-A clash implementation used by training.
It reports raw losses plus the exact pre-gamma weighted training objective.

Rows aggregate replica -> system and then system-equal. Protocol output binds
the checkpoint, evaluator commit/file, model config, A physical recipe, static
loss config, store index, validation manifest and selected identities. Every
run must retain `test_opened=false` and have exactly `24 * 4 = 96` rows.

## Evidence before review

Inside `enter-container`, conda environment `torch-ito`:

```text
python -m compileall -q tools/evaluate_static_decoder.py tests/test_static_decoder_evaluator.py
python -m pytest -q tests/test_static_decoder_evaluator.py --disable-warnings
5 passed
```

The implementation owner requests a `PASS` or `FIX_REQUIRED` verdict and any
required bounded CUDA preflight before the three formal checkpoint runs.
