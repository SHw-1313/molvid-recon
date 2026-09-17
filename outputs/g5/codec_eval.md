# PVB codec round-trip evaluation

Metrics are stratified by native time bucket; no cross-bucket mean is reported.

## ratio1_no_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.335856 | 3.16708 | 2.23116 | 0.0102846 | 0.000164706 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.322061 | 1.58394 | 1.16544 | 0.00825704 | 0.000172189 |

## ratio1_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 100.0 | 0.17088 | 2.84012 | 2.00466 | 0.00881832 | 0.000139972 |
| dt_80ps | 80.0 | 1200 | 80.0 | 0.171203 | 1.29534 | 0.94795 | 0.00662183 | 0.000137607 |

## ratio4_temporal

| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dt_100ps | 100.0 | 1500 | 400.0 | 0.624496 | 2.96257 | 2.08827 | 0.0108173 | 0.000173593 |
| dt_80ps | 80.0 | 1200 | 320.0 | 0.548376 | 1.41621 | 1.04178 | 0.00908035 | 0.000188396 |
