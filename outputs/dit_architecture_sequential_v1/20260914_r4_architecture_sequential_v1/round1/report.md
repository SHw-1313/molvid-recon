# Round 1: pre-AdaLN vector magnitude

- Decision: `REJECT`
- Evidence scope: `quick`
- Selected arm: `control`
- Unique variable: ffn_norm_source: post_adaln versus pre_adaln
- Added successful updates per arm: 5250
- Future atom-frame tokens per arm: 194960744
- GPU-hours: 4.947884

| H | E_amp relative improvement | systems improved | bond relative change | contact F1 change |
|---|---:|---:|---:|---:|
| H4 | -0.002128 | 1 | -0.000441 | 0.000267 |
| H8 | -0.002144 | 1 | -0.000541 | 0.000201 |

pre-AdaLN amplitude input did not improve the paired primary metric

Remaining risk: one training seed; sequential continuation gain is not a scaling-law result
