# Common repair candidate — 2026-09-28

Base: cc74a385577e57c64c9623111e0db86e54e48770.

Implemented packed-batch mixing, global stratified history/auxiliary execution
lists with cursor hash checks, explicit inherited loss overrides, Euler4/8
contracts, unused legacy auxiliary branch skipping, and metric family schema.
Fixed-history RMSF remains a diagnostic and is excluded from the main RMSF
summary, including Q12. The metric recomputer emits family-separated CSV rows.

Executed inside `enter-container`, `conda activate torch-ito`:

```
python -m pytest -q tests/test_gm_common_schedule.py --disable-warnings
2 passed in 6.47s
```

These are metadata/configuration tests, not CPU model execution. They verify
exact clip coverage, interleaved packed groups, H balance, stratified activation
counts, deterministic metadata resume, DDP list sharding, old schedule rejection,
fixed-history exclusion, endpoint-only release, and optional unused ES scales.
`git diff --check` passed. No whole-suite pytest was run.

CUDA model smoke, real CUDA resume and numerical tests are **not run**: all
eight GPUs have other users' active processes. No GPU is claimed or shared.
There is no independent review PASS or authorization artifact for formal
training yet. This common code release is an integration point for B, not a
scientific validation or capacity recipe release.

The model fixes dropout at zero; source/main draws use separate explicit torch
generators, and schedule generation uses a separate NumPy generator. There is
no overflow skip: nonfinite loss/gradient raises before optimizer/cursor advance.
After failure, resume restores the last saved checkpoint and RNG states; it does
not silently consume another clip or pick a different auxiliary mask.

Loss ledger verified from the actual P2-GM checkpoint:

| Term | Weight | Branch / gradient boundary |
|---|---:|---|
| flow | 1; h/v 0.5 each | target-interpolated latent; DiT/history/M, frozen teacher |
| clean_coordinate | 1 | clean teacher latent; decoder |
| clean_bond | 0.1 | clean teacher latent; decoder |
| near_coordinate | 0.1 | detached endpoint, s>=0.9; decoder |
| near_bond | 0.01 | detached endpoint, s>=0.9; decoder |
| generated_bond | 0.051707249134778976 | target-interpolated endpoint, s>=0.75; DiT/history/M/decoder |
| A actual-source terms | uncalibrated, not enabled | clash / motion-only ES / bond band pending A implementation |

G/M have no dedicated loss. Historical J0/J1 start at B0 with G/M disabled.
