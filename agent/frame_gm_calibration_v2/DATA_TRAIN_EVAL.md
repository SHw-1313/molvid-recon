# 数据、训练、评价与自动选择

## 1. 真实数据身份

训练 parent：`runs/frame_joint_v1_multitime_260921/frame_joint_step_00022272.pt`，SHA `b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`。

codec：`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`，SHA `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`。

旧多时间 data root：原 repo 的 `runs/frame_joint_v1_multitime_data_260921`；data hash `54bbdc198373ae9f477bf5803643aab77f960fe4498da8db1fcf9c5bfdd770c0`。train 100/200/400 ps 共 35,712/17,856/8,640 clips；同一 192 体系、replica 的视图不跨 split。

raw ATLAS 已用路径：`/data1/repo/BioKinema/data_atlas`，native 10 ps。从原始帧精确索引构建视图，不插值伪造细轨迹。不因现有 clip-store 没有某个 view 就宣称 raw data 不可用。

statistics：旧 repo `runs/frame_joint_v1_multitime/frame_statistics.pt`；file SHA `117dbcb674f9932c40216c4e2c3ee69153dacc3d3efebcca6e67c876a3b6bee9`；statistics hash `7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`。

新 worktree 的 relative runs/ 不会自动含有旧 run。解析 verified absolute parent/data/statistics 路径，读旧文件、写新输出；不能把整个旧 runs 当作可写输出目录复用。

## 2. P1 必要工作

### 指标修复

旧 `velocity_rmse` 实际是帧间位移误差，保留为 `displacement_increment_rmse_A`。正确有限差分速度用真实 Δt，输出 Å/ps；不是瞬时原子速度。跨 lag 比较必须报告 Δt。

所有轨迹内动态量使用相同对齐定义：每个生成/真实 frame 各自对齐至同一个最后 observed frame 的固定 backbone/有效原子集合，不能把预测先对齐真实 future、再拿去与自对齐的 MD 求动态相关。alignment definition 写入 schema。对于复合物使用蛋白参考集合，不把配体自身运动拟合掉；本轮数据仍是 ATLAS 蛋白。

同时优先计算无需坐标对齐的内部距离/扭转增量相关。零方差、frame 太少、缺拓扑分别返回 availability=false 与 reason，不把 NaN 静默当 0。

RMSF report 分开：每 atom/residue correlation/MAE；每 system 的平均 RMSF；system means 的 Pearson/Spearman、MAE、std(pred)/std(MD)。旧 `rmsf_flattening_ratio` 可留为明确命名的 mean-amplitude-ratio，不能继续用它代替系统间 spread ratio。Pearson 与 spread 同时看，随机扩大差异不能算正确。

从已有坐标重算所有受影响字段，含 Spearman/ratio/summary/per-system；不能只替换 RMSF 主字段留下旧衍生量。新路径 `runs/frame_gm_calibration_v2/P1/metrics_v3/`；历史归档不覆盖。

### 时间视图

保留 legacy uniform-grid H4/H8×100/200/300/400 ps，以便与旧结果比较。新增 fixed-history H4：observed times `[-300,-200,-100,0] ps`，未来 horizon=1200 ps，Δt 100/200/300/400 各有 12/6/4/3 个有效 future，padding 到最多16总位置。

相同 anchor、observed 坐标、atom identity 对齐，仅 query grid 改变；300 ps 只用于 valid。查询未满16的尾部不得把重复坐标当真实静态帧，所有层和统计只看 mask。分子平移旋转 augmentation 对同组所有视图一致。

每条记录写 raw indices、absolute time、relative time、native_dt、anchor、system/replica、split、view kind 和 hash。所有派生视图继承原 system split。H8 legacy view 不宣称在四种 lag 下都存在相同 1200 ps 终点；fixed-history 主检验只用上述 H4。

### Latent 诊断

复用已有 endpoint diagnosis，补 train-only Δz=z_future−z_last 的 h/v 分布、单位 source noise 和实际生成误差的范数分位数；按 lag/体系分组。做少量等范数扰动→decoder 输出敏感性比较；真实 future 参与的诊断明确 future-informed，不能并入生成质量主表。

模型相关数值在 CUDA；汇总文件/画图准备可 CPU。仅验证时间单位、刚体对齐、mask、常数轨迹 N/A 等关键性质；不全量回归，不另开 reviewer。

## 3. P2 数据与训练

四臂 B0/G/M/GM，使用同一个已验证 parent、同一条 sampler 清单和主随机数流。B0 是同预算续训，不拿未经续训的旧 checkpoint 代替 B0。

数据配方：75% legacy uniform clips（H4/H8 等比例），25% fixed-history H4 matched views。每类内部训练 query lag 100/200/400 等权；300 不进入训练/归一化/校准。该配方的总 H4/H8 比例约62.5%/37.5%，明确记录，不假装仍50/50。所有臂相同。

每 physical trajectory 每 effective epoch 默认 cap=24，跨 lag/view 共享 cap。按 system/replica 等权尽可能均衡抽取，不让长轨迹或较细 lag 仅靠 clip 多而占优势。按 H 和 view 组织同批是允许的；顺序/packing 必须在各臂一致，且实际曝光逐 bucket 统计。

保留 parent 的目标统计空间。新 view manifest 与旧 normalization source manifest 是不同身份：显式记录派生关系与 normalization source hash，允许授权的 train-only 派生视图复用旧统计。不要改旧 stats 文件/provenance/SHA 来骗过 `_load_statistics`；为 changed-data warm start 增加明确兼容规则。普通 resume 仍严格校验当前 data/normalization 双身份。

### 初始化和 optimizer

- 旧权重兼容加载，新增/不匹配项逐项输出报告。开关全 off 时旧 uniform view 结果应与 parent 一致（容许 dtype 数值误差）。
- P2 四臂均作为新实验 warm start，optimizer/scheduler/cursor 对称地从头建；不让 B0 继承成熟 optimizer 而新模块臂重置。
- 使用显式 warm-start 接口；现有 `--continuation-parent` 的严格契约不适用于所有架构变化。不要静默 strict=False 或把改架构称为 exact resume。
- 学习率默认 history 1e-4、DiT 2e-4、decoder 5e-5；BF16，AdamW、weight decay .01、clip1；5% warmup+cosine，teacher frozen。
- parent 已解析 generated-bond λ=`0.05170724987983704`，以 parent contract 核验值为准。所有臂共用，clean/near 权重沿用；本轮不重复四次梯度校准。

### Pilot 与正式预算

先在原3体系/9轨迹训练子集完成每臂128个成功update的 pilot，包含H4/H8和matched masks；另用实际192 train最大样本做 forward/backward profile。pilot 起点也来自正式 parent，但 pilot 子checkpoint 不作为正式训练起点。

显存/速度 profile 之后、正式训练之前冻结预算：首选四臂各1个 effective epoch；若预计四臂正式训练显著低于96 GPU-hours，可统一选2个 epoch。pilot/评估另计入共同32 GPU-hours，避免重复记账。不得只延长“看起来更好”的臂。

effective epoch 按预先生成的 trajectory/view曝光清单定义，不按不同GPU数量的steps相等。记录 valid atom-frames、clips、system exposure、GPU-hours。若预算要求不足1 epoch，可以统一选择清单的确定性前缀，记录覆盖率并将结论标成短预算，不冒充完整epoch。

微批调整依赖最大样本 profile；用 accumulation 保持相同 global sample exposure与损失计数。不得为GM单独减少帧数、原子数或通道。4卡允许一臂一卡并行，OOM时用同样的卡数/策略分批跑各臂。不要抢占现有作业。

## 4. P2 评估与 P3 起点选择

正式比较使用固定 valid 8 systems×3 replicas×2 anchors，legacy H4/H8 和 fixed-history H4，Euler16、seeds[0,1,2]，不得 best-of-N。pilot 仅用较小 valid 子集/seed0验证数据流。

每组含 persistence、当前模型 clean-latent decoder oracle、真实生成。oracle 随 decoder 变化需重新算，不能拿旧 oracle替代。persistence 可按同协议缓存一次。

主要指标：

1. Geometry：bond RMSE，angle error/validity，clash rate；aligned RMSD/dRMSD/contact F1作辅证。
2. Motion：atom/residue RMSF MAE/correlation；system mean RMSF的MAE/correlation/spread ratio。
3. Time：fixed-history公共时刻(400/800/1200 ps适用于100/200/400；四lag共同1200 ps)的MSD误差；各 grid自有的MSD曲线；内部距离/扭转增量相关；correct/wrong-clock配对诊断。

fixed-history 300ps grid 在400/800无采样点，不插值凑出真值；全部grid共同终点为1200ps。correct/wrong clock 只作诊断，不把错误标签部署为改进。

RMSF main values按同一窗口定义比较；不同Q的估计偏差同时存在于MD/生成，跨grid的主要一致性判断使用公共时刻MSD/分布，不要求各grid RMSF数值完全相等。

聚合：seed均值→anchor均值→replica均值→system等权；bootstrap以system为顶层，不能把原子当独立样本制造置信度。保存每个体系的原始指标和 availability计数。

### 自动选择规则（正式训练前冻结）

比较相对B0的逐体系配对差值；三组主误差分别为 bond RMSE、residue RMSF MAE、fixed-history MSD curve error。软件无效/泄漏/缺关键指标的臂不可选。

- 若一个臂在三组误差中至少一组的配对95% bootstrap CI明确改善，其他组没有明确恶化，同时clash和体系RMSF相关性/离散度没有反向崩坏，保留为候选。
- 多个候选采用三组误差的平均名次选起点，平局选择GPU成本低者；保存rank表，不把探索性选优称为统计确证。
- 无这样的候选，P3使用B0，并报告P2没有取得跨目标优势。允许训练策略研究继续，不自动强推GM。
- 所有结果先记录，再选择；禁止事后改metric、split或阈值。计算spread ratio接近1只作辅证，不单独排名。

输出 `selection.json`：所用规则版本、逐臂三组指标、CI、选择原因、parent路径/SHA、未解决问题。

## 5. P3 双臂训练和可选 bond 分叉

J0/J1 同一个 P2 parent、对称重置optimizer与scheduler，相同主sample顺序、主noise/flow RNG、LR和暴露量。sampled auxiliary 使用独立 RNG，不扰动主随机流。

先64-update两臂pilot（J1必须触发多个sampled steps）及梯度/恢复验证，然后独立review。正式训练预算优先1个effective epoch，两臂同步决定是否2个，总阶段不超过96 GPU-hours；其中为条件触发的bond分叉保留最多20%。K=2、4-step sampled branch 每8步，最终主评估仍Euler16。

若J1相对J0在几何上明确改善而RMSF/时间组明确恶化（按上述配对CI），从同一个J1 checkpoint启动bond keep/release各0.25 epoch，比较关闭显式bond后是否恢复运动。两臂保留其余J1训练机制、使用同预算；无此现象不追加。已处于bond-off状态不得再伪造bond-off对照。

测试集全程不打开。训练规模维持192；不加入40万静态、AF3或更多数据源。

## 6. 交付文件

每阶段生成 `resolved_experiment.json`、`commands.sh`（实际运行过的精确命令）、`run_manifest.json`、`metrics.json`、`per_system.csv`、`report.md`。

主图：geometry–motion散点；体系RMSF预测vsMD；每lag RMSF分布；固定历史MSD–真实时间；局部相关函数；参数/吞吐/显存；训练raw/weighted losses。使用已有绘图依赖，不安装新包。

保存选择后的代表体系生成坐标；完整数据/checkpoint留在runs，提交可审阅的小表格/配置/图到本任务结果归档。更新模型数据流图和shape表，让人能从raw history一路跟到generated coordinates。

## 7. 预算与故障处理

总额224 GPU-hours；P2/P3各96，其他32。首次 profile 后写分配和预估；worker按实际GPU占用累计。重复失败自动修复重试最多2次同类故障，仍失败先定位，不无限重启烧卡。预算接近上限时在checkpoint边界停并评估现有结果，报告明确剩余项。

同代码run中断可strict resume；代码/数据/目标改变则新run记录parent。已经完成的run不因轻微日志修复重跑；修改指标可从saved coordinates重算。review期间可以准备数据/图表，但不能抢跑尚未审阅的正式科学训练。
