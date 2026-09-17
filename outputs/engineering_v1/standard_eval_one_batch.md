# PVB codec round-trip evaluation (Validation split)

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | — | 1.08943 | 1.68805 | 1.42035 | 0.00625793 | 0.000105438 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | — | 1.0196 | 1.62936 | 1.32422 | 0.00627522 | 0.000105936 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | — | 0.642505 | 1.46102 | 1.11586 | 0.00606147 | 0.000101813 |
