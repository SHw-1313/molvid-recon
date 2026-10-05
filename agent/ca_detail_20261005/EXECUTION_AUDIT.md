# 运行真实性与耗时口径

2026-10-05。用户质疑“为何所有初步结果这么快，是否真的在 GPU 上训练/测试”。
本文件将训练、冻结模型推理、已有结果汇总分开；不是第三方独立审计。

## 前一轮究竟执行了什么

| 项目 | 实際操作 | 原记录耗时 | 不能声称的内容 |
| --- | --- | --- | --- |
| N0 | CUDA 定向测试和旧新 A1 固定输入推理 | 各测试日志记录 | 不是完整模型重新训练 |
| N1 | 冻结 A1 的生成/评分；48+16+8 draws | 123.31+71.93+23.72 秒，计时不含全部数据准备/加载 | 没有更新 A1 参数 |
| N2 | 读取已完成 S 三臂 checkpoint/指标，重聚合 | 汇总可很快；旧 S0/Sdec/Strunk 训练记录分别 1415.85/2965.62/2702.85 秒 | 不能把旧训练称为本轮几分钟内新训完 |
| N3 U | GPU 上固定特征的小 MLP，18 heads×2000 steps | 171.51 秒，含该 runner 内评分/保存 | 不是18个完整 DiT |
| N3 K | GPU 上固定特征的小 head，9 heads×2000 steps | 73.49 秒 | 不是9个完整生成器 |
| N3 ridge | CUDA 线性方程求解 | 非迭代训练 | 不是数千次神经网络更新 |
| N4 | CPU 元数据、XTC 读取及相关性统计 | 没有 GPU 训练 | 不是新 MD 或热力学验收 |
| 旧 Mamba | 新 history 替换后的短程 continuation，每臂128步 | M-history 955.42 秒，D0 970.36 秒 | 不是完整时间模块替换，也未证明质量改善 |

旧运行根目录：`/data4/users/sihao/workspace/molvid-review-followup-20261005/runs/review_20261005/`。
S 训练原目录：`molvid-static-response-20261004/runs/static_response_20261004/pilot512/`。
Mamba 原目录：`molvid-integration-abcd-20261004/runs/d2_mamba_20261004/paired_profile_v3/`。

因此，前面笼统的“实验完成”应理解为当时列明的有界任务完成，不能理解为所有规划架构都已经充分训练。此前 Cα/残基分层方案没有实现/训练；本分支才是首次实现。

## 这次实际核验的证据

`tools/audit_previous_execution.py` 在 `enter-container` / `torch-ito` / A100 上执行：

- 读取 27 个 U/K 模型 checkpoint，所有模型保存 step=2000，全部 AdamW 状态 step=2000，第一阶矩非零；逐文件 SHA 和矩范数已保存。
- 严格重建 `history_muvar_101.pt`，加载原 fit-only 归一化及 calibration 温度，在 GPU 上重新预测完整原 valid 缓存。
- 新算 MSE/NLL/coverage/width 各 horizon 的均值与原 JSON **最大绝对差为 0**；新推理/评分 CUDA stream 计时 232.68 ms。
- 产物：[核验 summary](../../runs/ca_detail_20261005/previous_execution_audit_v2/summary.json)，同目录有 execution 身份、GPU 监控与输入 SHA。

第一次核验脚本把 K 的 `regions.pt` 字典也当成模型，因无 optimizer 字段失败；随后按 `K_*.pt` 限定真实 head 文件重试通过。第一次失败目录/日志保留，不写成通过。

这些是 checkpoint、优化器和重新执行的相互印证；不能凭此补造过去没有留下的 GPU 利用率曲线，也不是对所有历史训练逐步重放。新生成的 telemetry 只属于本次核验。

## 本次 Cα 实验为什么也不需要数小时小 head 训练

本分支冻结 A1，以实际生成的 CA 作为中间条件；新训练模块只有 **92,032 参数**，三层局部图消息，未重跑 DiT backward 或 teacher 编码。
初始分层输出是明确的刚性残基平移基线，不是从随机坐标开始学习整个蛋白质。

- A1 cache：32 train体系×2窗口×1seed + 8 valid体系×2窗口×2seed，共96次生成，阶段 wall time **282.03 秒**；保存80个记录文件及完整生成坐标。
- clean/mixed 两训练臂，各 seed71/72、512 updates。四模型训练及保存阶段分别15.28/14.09/13.06/13.98秒；对应 CUDA stream 累计14.82/13.70/12.69/13.58秒。
- 全部512步记录保留，不只保留“最后一步完成”的标志。参数变化 L2 约6.86–7.11；checkpoint严格重载输出完全一致。
- GPU 为本机 A100-SXM4-80GB。cache 在 GPU5；clean 在GPU6，mixed在GPU7。UUID记录在各 `execution.json`，容器 PID 与 host监控命名空间可能不同。
- `gpu_telemetry.jsonl` 每2秒实际查询 nvidia-smi；已记录训练显存约2.99GB（nvidia-smi口径），Torch峰值分配约1.07GB。两者口径不同，不能混用。
- 历史原子/时间缓存复用使训练快，但也限制结论：32个训练体系、固定100ps、确定性小恢复器、冻结 coarse generator，不是正式多源混合预训练。

“真的更新了参数”与“训练足够、模型有效”是两件事。效果只由配对验证给出，不由 step 数、GPU利用率或完成标记保证。
