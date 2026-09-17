# Round 4: error-corrupted observed-history training

- Decision: `REJECT`
- Selected arm: `control`
- Parent checkpoint SHA256: `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34`
- Primary E_roll_bond relative improvement: -0.028617
- Systems with lower extra generated-prefix bond degradation: 2/8

| H | clean bond relative change | contact F1 change | amplitude relative change |
|---|---:|---:|---:|
| H4 | -0.000298 | -0.000963 | -0.002356 |
| H8 | -0.002564 | -0.000454 | 0.000517 |

- True-prefix rollout bond relative change: -0.002173

error-corrupted history training did not reduce paired generated-prefix extra bond degradation

Remaining risk: one seed and a 32-frame diagnostic; this does not establish long-rollout stability or a scaling law
