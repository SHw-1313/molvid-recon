# PVB codec round-trip evaluation (Train split)

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.768195 | 0.944104 | 1.36236 | 0.986531 | 0.0067908 | 0.00011396 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.315043 | 0.812106 | 0.81574 | 0.611587 | 0.00513803 | 8.58913e-05 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.417614 | 0.353831 | 0.915999 | 0.678755 | 0.00705834 | 0.000120152 |
