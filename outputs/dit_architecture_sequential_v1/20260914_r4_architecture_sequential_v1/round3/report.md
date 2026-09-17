# Round 3: local versus cross-block temporal refinement

- Decision: `KEEP_PARENT`
- Selected arm: `parent`
- Parent checkpoint SHA256: `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34`
- Added refiner updates per arm: 5000

| H | T vs L E_boundary improvement | T direction | L vs P E_boundary improvement | L direction |
|---|---:|---:|---:|---:|
| H4 | -0.056095 | 0 | -0.000998 | 0 |
| H8 | -0.049682 | 0 | -0.001031 | 1 |

neither local nor cross-block refinement improved the free-generation boundary metric

Remaining risk: one training seed; target-coupled endpoint post-training is not a proof of autoregressive rollout stability
