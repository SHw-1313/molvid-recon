# Unmodified PVB_origin four-condition baseline

All four conditions use seed 20260810, the same 32-batch/48-clip validation selection, T=16 sequential rollout, and 10 SDE steps. PDB is excluded.

The structure/motion values below are PVB-style clip metrics computed by the same output-local evaluator used for A/B. They are not the original eval_prot.py TICA/MSM numbers because the current versioned clip stores do not carry the original gzip-JSON trajectory paths.

## Validation loss

| condition | initialization | mode | valid loss mean |
|---|---|---|---:|
| A | static | direct | — |
| B | dynamic | direct | — |
| C | static | retrain | 1.815242 |
| D | dynamic | retrain | 1.312822 |

## Future-frame comparison

| condition | bucket | RMSD | dRMSD | bond RMSE | contact error | velocity RMSE | acceleration RMSE | frequency retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | dt_100ps | 3.322737 | 2.363112 | 0.080970 | 8.428e-05 | 0.011231 | 1.795e-04 | 0.159592 |
| A | dt_80ps | 1.846390 | 1.404843 | 0.079719 | 1.500e-04 | 0.009699 | 1.994e-04 | 0.503229 |
| B | dt_100ps | 3.276637 | 2.313010 | 0.039170 | 7.434e-05 | 0.011144 | 1.784e-04 | 0.130838 |
| B | dt_80ps | 1.768896 | 1.336727 | 0.041824 | 1.592e-04 | 0.009566 | 1.973e-04 | 0.421775 |
| C | dt_100ps | 3.773854 | 2.926631 | 0.186918 | 1.942e-03 | 0.012049 | 1.917e-04 | 0.497170 |
| C | dt_80ps | 2.590685 | 2.270513 | 0.157229 | 1.920e-03 | 0.010787 | 2.187e-04 | 1.437826 |
| D | dt_100ps | 4.008358 | 3.032640 | 0.052082 | 4.937e-04 | 0.012219 | 1.938e-04 | 0.618140 |
| D | dt_80ps | 2.678679 | 2.271368 | 0.055475 | 2.715e-04 | 0.011259 | 2.274e-04 | 1.881288 |

## Artifacts

- loss_curves.png / loss_curves.pdf: retraining loss with validation-mean markers; direct conditions correctly have no training curve.
- eval_reconstruction.png / eval_reconstruction.pdf: one-step and future reconstruction metrics by native time bucket.
- eval_dynamics.png / eval_dynamics.pdf: motion, contact/clash, and frequency metrics by native time bucket.
- eval_metrics.csv, training_loss.csv, and summary.json: machine-readable records.
