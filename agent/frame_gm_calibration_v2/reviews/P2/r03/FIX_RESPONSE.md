# P2 r03 选择器修复响应

## R1 — formal provenance 链

已修复。选择器现在在 bootstrap 前逐项验证：

- 冻结 selection rule 的完整内容，而非仅版本号；
- formal run manifest、candidate commit、parent、四臂 config hash、完成状态、最终 checkpoint hash、exposure、sampler sealed-test、target code commit 与 resolved config；
- 四臂 profile JSON 的 schema、arm、parent/data identity、step time 及文件 hash；
- metrics-v3 manifest schema，并重新计算 rows/metrics/per-system 三个 SHA-256；
- metrics 到 source evaluation protocol 的 SHA，protocol 到当前 formal checkpoint 的 SHA，以及 Euler16、seeds、paths、行数和 paired store；
- legacy/fixed protocol identity 在四臂间完全相同。

输出的 `selection.json` 保存 resolved experiment、formal manifest、每臂 config/exposure/sampler/target/resolved-config、profile、metric manifest 与 evaluation protocol 的路径和 SHA-256。

## R2 — 精确视图网格与配对目标

已修复。每臂要求 8 个 system，并逐 system 验证唯一且完整的：

- legacy `(lag 100/200/300/400 ps) × (H4/H8)`；
- fixed `(lag 100/200/300/400 ps) × H4`；
- 每行 `replica_count == 3`；
- legacy/fixed system 集合一致。

聚合前还逐 `(system, lag, history)` 在四臂核对 RMSF target，不能再用相同体系均值掩盖错配。

## R3 — profile cost 与最终平局

已修复。成本 mapping 必须恰有 B0/G/M/GM，且每项有限、严格为正；值还必须与 hash-bound profile JSON 精确相等。若合格候选的平均名次与冻结成本同时完全相同，选择器 fail closed，不再使用未冻结的 arm 字母序。

## R4 — 回归测试

`tests/test_frame_gm_selection.py` 已增加：

- per-system CSV 篡改与 evaluation checkpoint 错接；
- 重复/缺失 view、错误 replica count；
- Pearson 安全门；
- 双候选平均名次后按 cost 决胜及最终同值 fail closed；
- NaN、Inf、负数和零成本。

容器内命令：

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/workspace/molvid-recon-gm-calibration-v2 \
  pytest -q tests/test_frame_gm_selection.py tests/test_frame_gm_distribution.py
```

结果：`15 passed in 2.17s`。其中选择器定向测试全部通过；distribution 测试属于并行开发中的 P3 未提交文件，不作为本次 P2 复核依据。

正式选择仍将在四臂训练、legacy/fixed 评估和 metrics-v3 全部完成并把 formal manifest 标为 `COMPLETE` 后运行；当前不会用不完整 artifact 产生 parent。
