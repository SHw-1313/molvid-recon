# 实验结果整理版

先看 [关键实验结论](KEY_RESULTS.md)，再按实验组进入具体运行。关键表按
生成时间—数据设置—模型设置—训练设置—结果组织。时间取自路径中的运行标识，
不是 Git 检出后的文件 mtime；未记录的时间不推断。所有数值来自列出的原始结果文件，
未重训、未打开 test clip。完整历史输出已从 Git 历史清除；本目录保留清除前校验过的结果包。不同运行或 quick/final 评估不混作同一结论。

逐 epoch/step loss 见 figures/ 中的曲线；最终或单次 checkpoint 指标仅保留在
[evaluation_metrics.csv](evaluation_metrics.csv)，按 scope 区分 final/quick/未注明，不为其造图。

原始 JSON/JSONL/CSV/已有图片被语义化命名后装入 [raw_results.tar.gz](raw_results.tar.gz)。
[raw_manifest.csv](raw_manifest.csv) 记录原路径、新名称、SHA-256、大小与运行标识。
训练数据、checkpoint、日志和 test split 不在整理包内；保留的原始结果文件见 [raw_results.tar.gz](raw_results.tar.gz)，名称和 SHA-256 见清单。

| 实验组 | 生成时间（路径） | 运行数 | 图数 |
| --- | --- | ---: | ---: |
| [atlas_only_modified](runs/atlas_only_modified.md) | 未记录 | 5 | 7 |
| [atlas_quarter_50epoch](runs/atlas_quarter_50epoch.md) | 未记录 | 1 | 1 |
| [atlas_selected_trajectories](runs/atlas_selected_trajectories.md) | 2026-08-26 09:44:44～2026-08-26 10:56:36 | 8 | 22 |
| [dit_architecture_sequential_v1](runs/dit_architecture_sequential_v1.md) | 2026-09-14 | 6 | 0 |
| [dit_source_ab_v1](runs/dit_source_ab_v1.md) | 2026-09-10 | 3 | 5 |
| [dit_source_reassessment_v2](runs/dit_source_reassessment_v2.md) | 2026-09-14 | 2 | 0 |
| [dit_state_detail_pilot_v1](runs/dit_state_detail_pilot_v1.md) | 2026-09-08 | 5 | 4 |
| [dit_state_detail_probe_v1](runs/dit_state_detail_probe_v1.md) | 未记录 | 2 | 0 |
| [engineering_v1](runs/engineering_v1.md) | 未记录 | 5 | 3 |
| [g5](runs/g5.md) | 未记录 | 5 | 6 |
| [g5_logged](runs/g5_logged.md) | 未记录 | 2 | 3 |
| [origin_baselines](runs/origin_baselines.md) | 未记录 | 8 | 4 |
| [state_detail_codec_v2](runs/state_detail_codec_v2.md) | 2026-09-03 21:38:07 | 2 | 16 |
| [visnet_spatial_v1](runs/visnet_spatial_v1.md) | 2026-09-01 18:12:40～2026-09-01 19:25:57 | 4 | 16 |
| [visnet_spatial_v2](runs/visnet_spatial_v2.md) | 2026-09-02 17:04:56～2026-09-02 22:02:53 | 4 | 11 |

归档共 388 份结构化文件/原有图，新增 55 张训练曲线，
提取 4192 个评估标量。若某个设置或结果显示“未记录”，请查该运行的原始记录；不要把缺失值视为零。
