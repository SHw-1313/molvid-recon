# 必要验证与执行条件

仅运行回答具体剩余风险的检查。成功顺序：实现 → 定向 CUDA 数值检查 → 真实 smoke → 直接受影响回归 → source/profile → 两臂训练。不要先全仓 pytest。

## A. 后端和整合

- [ ] 固定 A 文件导入来源；B 的 backend 代码和用户修改保留；没有整体 merge。
- [ ] 新 trainer/evaluator/sampler/resume 的 resolved backend 都为 factorized_v2；真实 forward 能证明走到该分支。
- [ ] 真实 R2/R4 checkpoint 的 FP32 fields、RF loss、全参数 gradient/optimizer 更新通过 FIXES 的元素级容差；真实非零 modulation；缺失梯度模式一致。
- [ ] 真实 finite/clamp 失败会使检查退出非零；测试人工错误输出能触发失败，防止固定 PASS 回归。
- [ ] 同一个真实 R4 clip、H4/H8、seed、16 Euler steps，新旧 backend 的 final raw latent/坐标比较。FP32 latent沿用组合容差；坐标 allclose atol=1e-3 Å、rtol=1e-4。BF16以旧后端相同精度为参照，报告误差，若明显超出FP32及声明BF16误差尺度则调查，不直接扩大容差。
- [ ] R2 只补真实后端检查，不运行科学训练。

## B. Source 正确性

- [ ] Gaussian 的 source/path/target 与旧实现相同；tau=0/1 端点正确。
- [ ] Conditional 的两臂 source 差等于 m，target 差为 -m；四 field masks 和 observed clamp 正确。
- [ ] raw-zero detail 标准化和反标准化正确；只有一次 normalize/inverse。
- [ ] 改变或替换全部 future 坐标/latent 后，observed 条件和两个中心都不变。只保证 source 构造无泄漏；监督 target 当然可以改变。
- [ ] 采样时对未观测输入 fields 做 mutation，在相同 observed/source/noise 下结果不变，验证 clean future 没有被模型偷读。
- [ ] 同一旋转作用于 source center、latent、eps 后验证 SO(3)；translation 仅按原始 origin 协议处理。
- [ ] 验证 future 可以离开 center，只有 observed 固定；没有把 state 全部 clamp。
- [ ] 真实 repeat center 模板检查达到 EXPERIMENT 条件，或有明确替代依据；不使用真实未来好坏选择 source。
- [ ] source 构造的 template coords/latent 留在CUDA；CPU拓扑注册不要求整段数值数据回传。

## C. 指标与缓存

- [ ] 零运动 RMSF 值有效，但退化 ACF/Pearson 返回null/reason；有运动样本仍产生有效值。
- [ ] occupancy 的时间平均先于预测/目标差值；用接触发生时间不同但占比相同的简单例子验证其与逐帧 disagreement 不同。
- [ ] 噪声 per-coefficient RMS 的分母含C/3C；零尺度为0，非零尺度在大样本上接近声明值。
- [ ] rows→draw→sample→system 的统计正确；不等 draw 数/重新分batch不会改变定义下的权重。8/16步和perturb scales分别分组。
- [ ] future-only diversity 和 latent summaries 不含 observed；raw rows保存完整。
- [ ] 如启用cache，真实ragged batch的cached/direct fields、origin/masks/identity一致；改变codec/H/center/time/topology内容会miss或拒绝旧entry；双进程不读取半写文件。
- [ ] 若不启用cache，仍报告一次target encode、source准备时间；不声称获得cache加速。

## D. Runner smoke 与恢复

- [ ] 两臂 init_hash、参数量、训练schedule/H/tau/eps一致；source和optimizer独立。
- [ ] 每臂一个真实 train batch 的forward/backward/optimizer/16步sampling/decode全部CUDA有限。
- [ ] 一个可丢弃的短resume检查：连续路径与保存恢复后的下一步，batch cursor、tau/eps一致，参数更新满足声明数值容差；source不兼容checkpoint被拒绝。
- [ ] codec/statistics哈希前后不变。哈希仅在输入、checkpoint/验证边界执行，不每层/每step搬整模型到CPU。
- [ ] 旧probe/t1_pilot步数限制仍生效，新source_ab_v1上限20000；不能用旧CLI偷偷超额。
- [ ] budget和source合同冻结后才能启动科学训练；完成检查后无需再次请求开跑许可。

## E. 执行规模与停止

以上按现有测试文件组织，可合并多个断言到少量CUDA fixtures。直接受影响回归只覆盖molecular_dit、RF、adapter、trainer、diagnostics、新runner；没有公共数值语义改动时，不重跑codec训练、VisNet或全T1。

不要无选择运行这些文件里的旧CPU数值用例。对本次必要的数值case使用CUDA参数化或新增聚焦CUDA覆盖；纯metadata/I/O用例可CPU。backend CUDA测试需显式设置现有 `DIT_RUN_CUDA_BACKEND_V2=1`，记录实际执行数和skip原因，不能因默认skip获得绿色结果。

默认不重跑完整A主/深诊断、不重做B四档性能。补测限制为定位已确认缺口及接通新实验所需的R4样本、R2 parity和median profile。

本轮“完成”表示合同内实现和实验已经完成，不表示模型质量成功。科学结果不理想不是跳过实验或改阈值的理由。

## 执行者填写

| 检查组 | 实际设备/精度 | 命令/证据路径 | 结果 |
|---|---|---|---|
| A 后端 | 待执行 | 待执行 | NOT_RUN |
| B source | 待执行 | 待执行 | NOT_RUN |
| C 指标/cache | 待执行 | 待执行 | NOT_RUN |
| D runner/resume | 待执行 | 待执行 | NOT_RUN |

最终状态见根提示；遇到实现失败准确写FAIL，不把NOT_RUN写成PASS。
