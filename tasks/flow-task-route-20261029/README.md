<!-- TASK_ROUTE_IMPLEMENTATION_BEGIN -->
## 实现与复现入口

代码基底为 kernel `eca03dd`，权重父为已完成的
`/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt`。
这是新的 weights-only 阶段；不继承父优化器、数据游标或阶段学习率。

新增 `molvid/flow/task_adapter.py`，仅在所有 temporal 模块之后、原
Collective 之前应用逐原子等变残差。两支各用 64 维瓶颈：标量输入为
`LayerNorm(concat(h, norm_XYZ(v)))`，经 Linear/SiLU；向量仅沿通道做
无偏置线性映射，使用标量的 `2*sigmoid` 门控。两个 up 输出映射零初始化，
down 映射和门非零；初始化函数保持原模型，首步 up 可获得梯度，down
首步零梯度是该初始化的预期。新增构造位于 CPU `fork_rng(devices=[])`
内，不重新播种，不推进共有 CPU/CUDA 随机流。

两臂都每次计算两支，总参数 **10,175,884**，底座冻结 **10,009,356**，
可训练 **166,528**（每支 **83,264**）。这里“每次活跃容量”指两支完整
参与前向，并不声称零初始化首步所有参数均有非零梯度。`shared` 固定
混合权重 `(1/2,1/2)`；`routed` 在 joint 使用 `(3/4,1/4)`，在动态 Q1
使用 `(1/4,3/4)`。两臂总参数、每次执行的参数和乘加结构一致；这是软路由，
不是任务间梯度完全隔离。共享臂退化为任务无关的两支均匀混合。

`settings.task_adapter={"mode":"shared"|"routed","bottleneck":64}`。
`prepare_condition(..., route_task="joint"|"q1")` 要求显式任务身份，
不从 Q 推断、不把静态 Q1 当动态 Q1。`sample_truncated` 及
`LigandSourceGeometry.backward` 同样传递此标签；推理入口也必须明确指定。
关闭适配器时原模型 keys/参数/数学路径保持兼容；warm-start 只允许
`task_adapters.*` 为新增 keys，其余继承严格匹配。

原参数 `requires_grad=False`，优化器只接收 adapter 参数；底座后半段仍
保留对 adapter 输出的输入梯度，绝不 `no_grad` 截断辅助反传。保留原
`train()`/checkpoint 重计算；本模型无 dropout 或运行均值统计。
冻结底座权重不代表生成几何必然不变，仍用原配体辅助及完整评测检查保护项。

两配置 `configs/shared.json` 与 `configs/routed.json` 仅 mode 不同，复用
旧固定 history_mix 数据计划：512 更新=256 joint+256 Q1；共同 seed71、
原时钟/H/source/rho、每组第8次配体原生35NFE末4区间辅助及同16帧正常带。
AdamW/LR/clip 与原配方保持；没有新增 loss。

`tests/test_task_route.py` 的四项 CUDA 检查覆盖初始化/共享 RNG 与默认
兼容、双支激活与等变/隔离、冻结底座/梯度/保存恢复，以及真实
3M35/L4096 的 H1Q1 原生35NFE、Q1 RF 与配体辅助反传。
实际检查耗时和通过状态由 run 的执行回执记录，不在实现段重复维护。

`profile_route.py` 只执行7次弃用的 RF+AdamW 更新：固定计划最大11955原子
H4Q1一次，seed71首 joint 三次，计划中 H4原子数中位 Q1三次；后三次
各列首次及两次稳态，保留完整原子、不做大辅助采样。它只给这些具体窗口的
成本与峰值，不能把最大或中位窗口外推为整个训练池均值。

```bash
PYTHONPATH=$PWD python -m pytest -q -s tests/test_task_route.py
PYTHONPATH=$PWD python tasks/flow-task-route-20261029/profile_route.py --allocator-GiB 32
# 仅在工程师冻结预算后，由其指定每臂完整时间上限：
python tasks/flow-task-route-20261029/run_train.py --arm shared --budget-seconds SECONDS --phase first32
CUDA_VISIBLE_DEVICES='' python tasks/flow-task-route-20261029/check_pair.py --step 32
python tasks/flow-task-route-20261029/run_train.py --arm shared --budget-seconds SECONDS --phase remaining
```

`routed` 使用相同入口；`--phase all` 可按已有授权连续执行。
有限 runner 不重试，首32另存 `checkpoint_step32.pt`，完整512日程不缩短；
remaining 的同臂总额度扣除 first32 实耗。CPU `check_pair.py --step 32|512`
读取两臂实际日志/checkpoint，核对数据/源噪声/progress/RNG/曝光/辅助次数、
模型和AdamW有限性、L4096底座逐位未改及 adapter 输出参数已更新；
训练后 adapter 权重和 loss 不要求相等。

推理薄入口为本任务 `sample_entry.py --manifest ... --entry ... --arm ...
--checkpoint ... --output ...`。它只沿显式 entry 的 family/mode 选择 route：
endpoint、Q1 rollout/小步组合为 `q1`；joint、Q12/Q4、原纯clamp粗细为
`joint`，不根据请求帧数猜任务。Q12/Q4 每次 generated-history 重编码继续
传原 joint 标签。`molvid/flow/kernel_sampling.py` 仅增加可选 `route_task`
透传，None 保留旧调用。原验证过的有限 solver 在本任务内复用副本，只补
标签，不改采样数值、source抽取或时间网格；不支持 clean-anchor 端点条件。
<!-- TASK_ROUTE_IMPLEMENTATION_END -->
