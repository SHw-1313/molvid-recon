# R4 source A/B v1

Common endpoint: 20000 optimizer updates

Gaussian and conditional arms share initialization, data schedule, H4/H8 history, tau/noise generator, codec and evaluation clips.

| arm | main rows | subset rows | main future aligned RMSD | main future bond RMSE |
|---|---:|---:|---:|---:|
| gaussian | 52 | 128 | 2.1646002084823737 | 0.8944938034357539 |
| conditional | 52 | 128 | 2.0776938387766766 | 0.7049312218917992 |

The table is descriptive; source arms are not selected by directly comparing RF losses. Test payloads were not opened.
