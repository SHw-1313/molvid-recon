# PVB codec round-trip evaluation

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.698359 | 0.476461 | 1.35523 | 1.00754 | 0.00566139 | 9.55481e-05 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.30045 | 0.144682 | 0.865605 | 0.652273 | 0.00418887 | 6.98217e-05 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.516587 | 0.0672037 | 1.10577 | 0.823784 | 0.00615658 | 0.00010406 |
