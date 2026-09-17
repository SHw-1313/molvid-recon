# PVB codec round-trip evaluation (Train split)

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.738477 | 0.92368 | 1.32876 | 0.964085 | 0.00672182 | 0.000112805 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.332049 | 0.812389 | 0.842627 | 0.630928 | 0.00520169 | 8.68247e-05 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.426532 | 0.378307 | 0.926879 | 0.691459 | 0.00708774 | 0.000120524 |
