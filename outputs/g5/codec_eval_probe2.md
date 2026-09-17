# PVB codec round-trip evaluation

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.337283 | 3.41536 | 2.45393 | 0.0103711 | 0.000163912 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.318 | 1.6097 | 1.18464 | 0.00799185 | 0.000166811 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.171622 | 3.08774 | 2.22268 | 0.00884426 | 0.000137441 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.170119 | 1.35301 | 0.99099 | 0.00653129 | 0.000135554 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.649014 | 3.20518 | 2.30875 | 0.0108965 | 0.000173022 |
| dt_80ps | 80.0 | 1200 | 320.0 | 0.511782 | 1.45131 | 1.0684 | 0.00877253 | 0.000182298 |
