# PVB codec round-trip evaluation

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.335855 | 3.16708 | 2.23116 | 0.0102846 | 0.000164706 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.322061 | 1.58394 | 1.16544 | 0.00825704 | 0.000172189 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.15044 | 2.84457 | 2.00728 | 0.00884002 | 0.000140315 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.150532 | 1.2938 | 0.94535 | 0.00661954 | 0.000137559 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.639412 | 2.9615 | 2.08792 | 0.010822 | 0.000173658 |
| dt_80ps | 80.0 | 1200 | 320.0 | 0.561081 | 1.41642 | 1.04238 | 0.00908575 | 0.000188473 |
