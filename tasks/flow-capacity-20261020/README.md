# Flow A：六个增容位置与共享控制（2026-10-20）

当前必要验证已通过，已冻结科学源码，准备正式六路启动。4项原针对CUDA检查及2项激活重算输出/梯度/RNG对照通过。真实17,002原子H4Q12最大窗：wide峰25.11GiB/44.44秒单步；Collective峰17.26GiB/33.72秒单步。两者loss/梯度/更新后参数及Adam状态均finite。以上为最大窗，不是平均吞吐；首次OOM、入口修复、受影响检查失败与调试全部保留计费。中央管理仍只在molvid_recon/AGENTS、HANDOFF、TASKS；此处是单一工程结果入口。

## 固定实验

基于group-QK/逐原子Value/原Collective/固定rho=1.1397942064485225 Å/原RF loss，独立分支research/flow-capacity-20261020，基点bee4312。原source、physical time、35NFE采样及评分不变，不带noise、GroupV、pair-message、几何loss。全部seed71，从头静态4096更新，再各自静态final权重暖启动态4096；不把小模型强加载到wide。AdamW lr1e-4、warmup128、clip1、bf16。第一32步保存/原4096配置exact resume，scheduler不改。

| 臂 | 增容位置 | 参数 | 分配GPUh（含本臂12Q50） |
|---|---|---:|---:|
| control | 原256宽、6空间层 | 6,701,836 | 9 |
| observed | 观测编码后4空间层 | 10,009,356 | 11 |
| query_pre | 当前query前段4空间层 | 10,009,356 | 14 |
| temporal | 原时序模块后3同构模块 | 9,671,692 | 11 |
| collective | 保留原128通道块，再3个512通道组级块 | 9,658,072 | 14 |
| query_post | 原末两层后再4空间层 | 10,009,356 | 14 |
| wide | 全局384宽、8空间层 | 17,890,572 | 21 |

五位置新增约2.96–3.31M参数；wide是更多参数的全局扩容参考，不当作等参位置对照。原小模型全部共享参数按同seed保持初始化，新增模块在独立RNG scope创建，初始为残差恒等映射且可获得梯度。wide使用同初始化规则但不是同函数/同参数。静态时没有历史预测：observed新增层在当前受噪几何路径上预训，动态时才移至observed路径；因此observed/query_pre的静态阶段可为同函数，位置差在MD阶段。所有臂的时序模块在静态阶段均沿旧规则旁路，不能声称其已静态预训。

全部七臂统一checkpoint_query_blocks=true；仅训练激活重算，避免为wide删大体系。默认false兼容旧路径；同权重输出/梯度已验证，采样路径不改。自连接方向在数学上恒零，现显式断开原实现中±1e8中间导数的浮点抵消；非自身边和forward数值不变，参数梯度差在重复底噪内。所有子命令强制本工作区PYTHONPATH，记录并校验molvid.__file__。

## 本轮真正可训练数据

详见[data_ready.json](data_ready.json)，不是磁盘原始全集或NAS全集。

- ATLAS：512个TRAIN体系、165,888个普通16帧窗口，100/200/400ps；最大17,002原子。所有体系具备3时钟。
- MISATO：复用旧5Å接触筛选后1022体系、10,152窗口，80/160/320ps；最大16,999原子。定义为参考片段每帧蛋白—配体最近重原子距离≤5Å，不按生成输出筛失败。8体系因筛选缺部分时钟，原sampler只循环实际可用桶，不补回坏窗口。
- 静态PDB：已注册的2569个protein来源，172–5973原子；这是筛选的mmCIF子集，不是全PDB/复合物全集。4096更新保证全部来源至少一访，再1527次第二轮。
- 动态4096步两域等权：ATLAS每体系4访、MISATO每体系至少2访（4个额外第三访），覆盖全体系和H4/H8，**不等于每体系所有时钟×history组合全遍历**。
- 新MISATO12800原始训练数据当前不是ClipMMapDataset、容器未挂/data5，本轮不搬812GB/改挂载/启动大数据转换。
- 沿用历史1536窗口/256体系的固定rho，不把旧统计冒称新ready池校准。

项目development48、confirmation_reserved、excluded_sealed和六评测ID与动态TRAIN均零交集；PDB池与已有ATLAS保护ID、六评测ID零交集。本轮仅作开发实验：ATLAS ready与外部BioKinema/PVB test分别8/4体系重叠；旧PDB静态注册池与MISATO官方test有6个PDB重叠（1B74、3O5N、4FYH、4KYK、4O1L、5MTE）。这些事实不被管理迁移抹除，不能宣称所有外部测试独立。没有打开封存测试坐标。

## 运行与有限资源

总96 A100 GPUh，七臂94h＋共享检查/profile/失败2h，非每卡96h。六个扩增先占GPU1/2/3/5/6/7；control在observed同lane完成后顺序执行，不占第7卡。不动外部GPU0/4进程。

[launch_lanes.sh](launch_lanes.sh) 只执行本轮固定六lane命令与wait，接现有CPU评分；[run_arm.py](run_arm.py) 逐命令累计墙时并以剩余分配作timeout，无自动重试或跨臂挪额度。每lane静态→动态→6条件各2draw Q50连续执行。输出在`runs/flow_capacity_20261020/{arm}/{static,md}`，生成在`generation/{arm}/{case}`；命令、PID、退出码、秒数在`{arm}_execution.json`和`process_logs/`。异常保留结果并报告；不自动重启、改数据或降低单臂步数。

全部84新Q50：1whz/3vnb/6e5y与4J8T/2XSB/5TV1，seed17/29，H4，50未来帧，100/80ps，18Heun区间35NFE。旧512控制不复用，需当前同数据/同4096静态+动态配方控制。七run内部比较是位置/容量证据；相对旧512的差不能归因单独的数据扩量。

[compare_capacity.py](compare_capacity.py)复用MAIN common_panel/ComplexMD0.3.1、现有五类几何/主侧链拆分、CA与残基质心幅度、固定MD柔区、RMSF Pearson、CA/PCA/接触ACF和原始3Dmol轨迹导出；不靠压幅判成功。全部CPU评分在GPU命令完成后自动顺序执行，手工WebGL播放如无浏览器仍明确未验证。

启动估算依据：旧group512平均4.763秒/MD更新；新ready池两域等权mean atoms2579.9，旧子集2585.9，规模均值近似相同，最大窗验证用于显存而非平均ETA。考虑容量与统一重算开销，粗估control6–8h、observed/temporal各7–10h、query_pre/query_post/collective各8–13h、wide12–19h，包含静态/动态与12Q50；非完成保证。总分配仍94h+2h，真实进度按日志修正，不单臂减少曝光。共享验证实际155.919秒（含全部失败），见run/shared_resource_summary.json。
