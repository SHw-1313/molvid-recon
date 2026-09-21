# 数据、训练、bond 续训与评估

本轮验证整体新架构与联合训练，随后检验“先用 bond 学几何，再移除 bond 是否恢复运动”。用户与同事已多次观察到 bond 改善几何但恶化运动；不得把 bond 常开写成既定最优，也不能仅按 bond RMSE 选 checkpoint。

## 数据边界

| 阶段 | 数据 | 时间 | 窗口与观察 |
| --- | --- | --- | --- |
| tiny | 原 3 体系/9 轨迹 | 100 ps | 16 总帧，H4/H8 |
| pilot | 原 48 train / 8 valid 体系 | 100 ps | 同上，H4/H8 各 50% |
| main | 从原 train split 扩到 192，原 8 valid 不变 | 先 100 ps | 同上 |
| multi-time | 新候选继续训练；不与 bond 分叉同时改时间 | train 100/200/400 ps；valid 增加 300 ps | 同上 |

原 8 test 体系保持封存。所有 replica、重采样 view 和重叠片段跟随 system split。192 个体系若尚未 materialize，先完成 48 的实现与实验，报告欠缺的数据，不从 valid/test 凑数。

每轨迹每有效 epoch 抽 24 个窗口起点；若合法窗口不足则记录实际数，不能通过重复计数宣称扩大独立数据。100/200/400 ps 视图均来自真实连续轨迹；16 帧 400 ps 在 100 ps 底层网格需 61 个帧位置的跨度。重新读取 source dt 与真实 time_ps，不能只改现成 clip 标签。每轨迹的 24 个窗口预算在多时间 bucket 间分配，不默认变为三倍。

T=1 静态输入必须可用，时间交互 mask 正确，仅有适用的几何/重建目标。本轮用小样本验证这条路径，不加入全部 40 万静态结构，不把单帧重复成动态标签。不同时扩入 MISATO、配体、80 ps 新来源和 AF3。

## 梯度和 loss

冻结逐帧目标 encoder+stem。历史 encoder、DiT 和 decoder 是联合训练范围。flow s 每个样本采 Uniform(0,1)，同一样本各 query 帧共享本次 s，噪声按有效逐帧 latent 采样。

flow loss 为 h/v 两字段 MSE 等权平均；先每个样本有效元素平均，再对样本平均，不让大蛋白因原子更多自动占更大权重。padding、观测位置不作为未来 flow target。

| 分量 | 默认权重/作用 | 梯度 |
| --- | --- | --- |
| flow_h / flow_v | 各 0.5 | history、DiT |
| generated bond | 对 s≥0.75 的 endpoint 解码，系数 λb | history、DiT、decoder |
| clean coordinate reconstruction | 1.0，真实逐帧 latent 重建 | decoder |
| clean bond reconstruction | 0.1 | decoder |
| near-endpoint coordinate reconstruction | 0.1，对 s≥0.9 的预测 endpoint | decoder only |
| near-endpoint bond reconstruction | 0.01，相当于 0.1×0.1 | decoder only |
| velocity/acceleration/contact-distance training loss | 0 | 本轮不开 |

near-endpoint 分支 detach 预测 latent 及其他可训练条件，不能把配对坐标回归通过它传回 DiT/history。clean 分支目标 latent 冻结。generated bond 分支保留完整梯度；不能因为 decoder 曾经冻结，就沿用 no_grad/inference_mode 切断坐标辅助梯度。

λb 在 flow 起步后用固定的 16 个训练 mini-batch 校准为 generated-bond 对 DiT 的梯度范数约占 flow 梯度的 10%，记录后固定。零/非有限梯度时先检查计算路径，不伪造系数。记录各项 raw loss 和 weighted loss；仅看总 loss 不能解释运动退化。校准不看验证/测试指标。

所有显式 bond loss 从一个可枚举的 loss 配置入口取得有效权重。`bond.enabled=false` 应令上表三个 bond 权重全部为 0；若设为 false 时仍给出非零子权重，配置解析应报告冲突或明确生成全零 resolved config，不能悄悄沿用旧默认。

## 训练阶段

1. decoder 预热：默认 3 个有效 epoch，若仍明显未收敛，可在已给预算内延长到 5 个。优化新 decoder 的 clean coordinate/bond；暂不训 DiT。检查真实 latent 重建的 RMSF，而不只检查坐标。
2. flow 起步：1000 successful updates，训练 history/DiT，decoder 冻结；以 flow 为主。完成后校准 λb，并保存共同起点以便做 joint/frozen 的短对照。
3. joint pilot：48 train，默认 6 个有效 epoch。开启上表 loss，训练 history/DiT/decoder。短 frozen-decoder 对照从第 2 阶段的同 checkpoint 分叉，相同数据与步数；关闭只会产生常数的 decoder-only 重建优化项，保留经冻结 decoder 回传的 generated bond。
4. main：192 train，目标 20 个有效 epoch，每主臂最多 64 GPU-hours，以先达到者为止。记录真实曝光量；若预算到而尚在改善，标记 optimization-incomplete，不据此宣称架构失败。
5. bond continuation：从已训练 joint parent 分叉成 keep/release，详见下节。优先在 48 体系 pilot parent 做一次；若主模型已有完整结果，可用该 parent 替代，不默认两套全跑。
6. multi-time：在完成 bond 对照判读后选择明确候选，以独立 experiment ID 混合 100/200/400 ps；这一阶段不再同步改变 bond policy、架构或数据组成。

默认 LR：DiT 2e-4、history 1e-4、decoder 预热 1e-4 / joint 5e-5。AdamW wd=0.01，norm/bias 不 decay；grad clip=1；5% warmup 后 cosine。attention/FFN 可 BF16，冻结 geometry 先保持 artifact 的已验证精度，关键几何和 norm/loss FP32。deterministic correctness 检查用 FP32，记录训练精度与确定性选项。

每卡先用 16k atom-frame 的 microbatch 目标做 profile；单个样本更大时允许 singleton，不丢大体系。依据最大体系和显存测量调整到可运行配置，effective global atom-frame 目标 128k，可 gradient accumulation。记录真正 batch 大小和样本权重；DDP 不同 rank 有效样本数不同需正确全局加权，不能简单平均各 rank 的均值。

DDP 要实现进程初始化、sampler 分片、梯度同步、每 rank RNG、rank0 保存与评估协调。所有可训练模块受同步管理；不能只包装 DiT 而漏掉 decoder/history。单次训练 forward 的组件调用与 DDP 边界清楚。先验证单卡/两卡同一小 global batch 的更新一致性，再跑长训。

## bond-on / bond-off 成对 continuation

这是一项新实验，含两个对照臂，不是两项不同目的的大训练。

| 项目 | bond_keep | bond_release |
| --- | --- | --- |
| parent | 同一已训练 joint checkpoint | 同左 |
| architecture / teacher / stats | 不变 | 不变 |
| trainable modules | history + DiT + decoder | 同左 |
| generated bond | 保持 parent λb | 0 |
| clean / near-endpoint bond | 保持 parent 权重 | 全部 0 |
| coordinate reconstruction / RF | 保持 parent 权重 | 同左 |
| topology message passing | 保留 | 保留 |
| 数据、时间、H、batch schedule | 相同，仍 100 ps | 同左 |
| optimizer moments / RNG / cursor | 从 parent 复制 | 同左 |
| 续训学习率 | 两臂共同的新低 LR schedule | 同左 |

分叉 parent 默认是预定 joint 阶段的最后一个完整 checkpoint，不按最小 bond 或某一次采样的运动指标挑选。parent 本身先做同一套评估，后续两臂以相同更新步数/曝光量取 checkpoint。

续训设置：目标 5 个有效 epoch，每臂上限 16 GPU-hours。两臂共同将 LR 设置为原 joint base 的 0.25 倍：DiT 5e-5、history 2.5e-5、decoder 1.25e-5；5% warmup，cosine 到本轮 base 的 0.1 倍。保留 optimizer moments，在两个臂中都重建相同 continuation scheduler；不要让一臂沿用已接近零的旧 scheduler、另一臂重新 warmup。

使用同一训练 seed、每 rank RNG 和数据游标；H、s、epsilon 的抽样顺序一致。关 bond 不应改变其他随机调用次序。评估随机 generator 独立，不能改变训练 RNG。关闭的 bond 可以在 no_grad 下记录诊断值，但不能参与任何梯度。

必须做一次“总 loss 及各项梯度归因”的检查：release 的三个 bond weighted loss 为零，且与完全移除 bond 项的总梯度一致；coordinate loss 等仍能间接改善键长，这不算 bond loss 未关闭。

该实验回答去掉 bond 后的可恢复性/几何遗忘；不证明从头不加 bond 的模型会怎样。如果运动没有恢复，可能仍有表示/优化/采样限制，不能自动判为四帧块问题复发。

## 主比较和预算管理

- A：旧块生成 + 原已选 bond 配方，在相同数据范围继续训练。保留历史 weights/stats 身份；缺失旧产物只报告缺失，不能从模板 λbond=0 假装重现。
- B：新 frame joint 候选。
- C：48 体系上的新架构 frozen-decoder 短对照。其 joint 对照复用 B 的同起点 pilot，避免重复跑。
- D：上面的 keep/release 成对续训，优先于大规模 multi-time。

A/B 既报告相同数据曝光量的结果，又报告耗时和已有预训练历史；它是工程候选比较，不能把不同 warm start 的全部差异归因于某个 block。C/D 才是明确的同源分叉比较。

4 卡时可两个 2 卡 run 并行；2 卡时顺序运行。同一 run 独占输出目录，GPU 不重叠。D 每臂单独预算；到达配对比较长度后先评估，不用无限续训替代判断。

## 评估

quick：固定每 valid 体系 2 个窗口，1 个噪声 seed；完整阶段评估：每体系 8 个窗口、4 个噪声 seed，H4/H8，Euler 16 steps。固定窗口和 seed，不 best-of-N，不把 quick 与 full 混成一条曲线。每个 epoch quick，阶段末完整评估；不每个 step 跑昂贵评估。

报告下列组，按 system 等权并给每体系数值：
1. clean latent decoder oracle：coordinate、bond、RMSF，分辨表示/解码问题。
2. 几何：bond RMSE、clash、局部距离、contact F1/occupancy。
3. 运动：相对真实 MD 的 RMSF 幅度误差、逐原子相关、物理 lag 的位移与 ACF；零方差/太短窗口显式标不可用，不填零冒充好分数。
4. 构象与生成：pair-distance/contact 分布，多噪声样本覆盖；参考轨迹 RMSD 作为诊断而非唯一排名。
5. 周期性：所有相邻步位移按索引 mod 4 分组，与 MD 同样分组；不只优化边界变平滑。
6. rollout：H8，每段生成 8 帧，连续 4 段；区分真实历史与生成历史。时间、坐标参考跨段一致，不用真实未来重置。短窗口统计不宣称慢动力学/平衡自由能已经学会。

如要声称 H8 比 H4 有效，另用相同未来最后 8 帧：H4 只看索引 4..7，H8 看 0..7；H4 条件及对齐不许借用 0..3。不能比较“12 个未来帧”与“8 个未来帧”就归因为更多历史。

bond 对照必须给 parent/keep/release 三组同范围结果和曲线：
- release 运动更接近 MD且几何保持：支持分阶段 bond 策略；
- 运动恢复但键长/clash 明显恶化：报告 tradeoff，不宣称胜出；
- 两臂都改善：增加训练本身可能重要；
- 几何保持但运动无改善：去掉 bond 不足以解决问题。

无需预设一个所有体系统一的 RMSF 硬阈值；先报告对 MD 的误差、体系间一致性和几何代价。不得用旧模型的过度运动作为“保留 50% 幅度即通过”的参照。
