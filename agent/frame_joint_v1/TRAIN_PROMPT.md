# 实现完成后的训练启动文本

使用前提：实现 session 已提供可运行的新架构、必要 CUDA 检查、原 3/9 tiny 结果和真实启动命令。推荐固定配方执行使用 Luna / high 或 Sol / medium；涉及新科学决策、梯度/恢复 bug 时回到 Sol / xhigh。

用户发送本文件即授权按以下范围使用最多 4 张分配给本任务的 GPU 进行训练与评估；实际可用 2 卡时顺序执行，不占用他人的 GPU。不 push、不改依赖、不打开封存 test。不为已明确授权的同一预算反复要求确认。

1. 先读根 AGENTS、本任务 HANDOFF、TRAINING_EVAL、experiment_spec.json。使用实现者验证过的实际 CLI 命令和 resolved 配置；计划 JSON 本身不是现有训练程序的可执行配置。
2. 确认已有实现和检查完成。若仍有已知缺陷，先修复具体缺陷并重跑相关短检查；不要跳过修复直接 pytest。若 tiny/检查已有同 commit 的有效结果，不重复全量运行。
3. 记录训练 commit，使用互不覆盖的输出目录。运行期间不热改同一 run 的科学代码/配置；必要修复创建新记录并说明是否需要重启。
4. 完成 48 体系 pilot 与短 frozen-decoder 对照。已完成者直接复用，保留共同起点关系。
5. 优先完成用户新增的 bond_keep / bond_release 续训：同 parent、同数据、同 H/s/noise 调度、同低 LR schedule；release 所有显式 bond 项为零。每臂 5 有效 epoch / 最多 16 GPU-hours，比较共同更新步数的 checkpoint。
6. 完成 192 体系旧 A / 新 B 主比较，每臂 20 有效 epoch / 最多 64 GPU-hours。192 manifest 缺失则报告缺失，继续可做的 48 体系工作，不能改 split 凑数。
7. 每阶段按计划评估并保存 parent/keep/release 的几何—运动对照。若两臂不同速度触及预算，主比较使用共同曝光量的 checkpoint，分别报告最终预算点；不掩盖欠优化。
8. multi-time 的代码和配置本轮必须准备好。是否启动由本指令中的总预算余额决定：最多额外 24 GPU-hours、5 有效 epoch，固定 train 100/200/400 ps 与 held-out 300 ps；不同时改变已选择的 bond policy。若 bond 结论仍存在明显 tradeoff，分别报告，不擅自宣布一个赢家或扩增组合实验。

预算上限：pilot（含 decoder 预热与 flow 起步）16 GPU-hours；短 frozen-decoder 对照 8；bond 两臂合计 32；192 主臂合计 128；multi-time 最多 24。总计最多 208 GPU-hours，已完成的有效 run 复用，不重复消耗。不同阶段到达预算时如实标记未收敛/未执行，不自动加预算。

不要承诺这些预算一定收敛或两天内全部跑完。4 卡全程满载仅计算预算也约 52 小时，实际还包含数据和评估开销。以实测吞吐安排，优先交付正确模型、48 体系结果和 bond 成对实验。

训练中遇到 NaN/OOM/梯度异常：保留失败证据，定位后做最小修复；不要降低评估标准、换数据、偷偷冻结更多模块或让 CPU 代替 CUDA 继续跑。常规断点恢复可自主完成。

交付：实际命令/config/commit/hash、成功更新与数据曝光量、每组算力、几何与运动曲线、同一批体系的表、短 rollout、未解决问题。必须区分 parent、继续加 bond、去掉 bond 三者，不能仅报告 release 优于 parent 就归因为去 bond。
