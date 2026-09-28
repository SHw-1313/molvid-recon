# B learned history memory：实现与 raw 数据阶段报告

状态：blocked（CUDA 槽位），2026-09-28。集成候选 `43bbadf0491c8b3815be03ae1f3ad0114484227a`；独立静态审阅进行中，尚无数值或正式训练放行。环境为 enter-container / torch-ito。没有 CPU 模型 fallback。

## 问题与方法

固定四层 DiT+G，通过可学习的 raw 长历史表示改善运动与 multi-lag，而不把完整 geometry teacher 的历史调用成本扩展到129帧。该假设尚未通过训练/评估验证。

数据流：raw observed `[Th,N,3] Å` → 两层共享共价局部原子网络（64 scalar / 32 vector）→ 蛋白 residue tokens → 真实 ps 多尺度核、level、近期有序路径 → 每 token 8 个可学习 slots（128/64）→ 256/128 投影、所属 token gather → 四层 DiT+G → per-frame future h/v → 现有 decoder。

旧 R4 history 和旧 MotionContextEncoder 在 B backend 中不实例化。保留兼容的 DiT 时间衰减参数；退出旧 history/M 权重在 warm-start report 的 retired 字段列明。模型级接入已经实现，实际加载/参数梯度报告需要 CUDA pilot。

## 数据

沿用冻结192 train/8 valid体系划分。本次只读 train raw：1bkp_A（2307 atoms）、1by2_A（886）、1cpq_A（920）各 R1/R2/R3。另读 train 中位体系3gwi_A（1374）与最大体系4yal_A（4975）的R1，作为后续profile输入。sealed test未打开。

每条轨迹真实读取并验证 native10ps 时钟，覆盖 raw indices 0..1760；anchor=1280，即12.8ns真实past和4.8ns真实future。没有审核更后的轨迹尾部，也不宣称已审计全部576条train轨迹。

history span=800/3200/12800ps；observed dt=100/200/400ps，与 query lag=100/200/400ps独立构造。Q12；两个完整teacher anchors固定[-800,0]ps。最长history分别129/65/33帧，含两端。300ps只valid，本次未读取。

所有帧对齐同一个last-observed backbone reference；future只独立变换到该observed reference，不影响history/anchors。短长臂使用完全相同anchor和target切片。

实际文件、split/manifest身份和每条raw索引见[data_card.json](data_card.json)。原始轨迹bank本轮重新写入本工作区runs，无共享可写symlink。

## Loss与训练

teacher、P2 GM parent和statistics按指定SHA核验；从实际parent `contracts.loss`展开账本见[loss_manifest.json](loss_manifest.json)。planned base-only pilot使用flow1（h/v各0.5）、clean coordinate1、clean bond0.1、near coordinate0.1、near bond0.01、endpoint/generated bond0.051707249134778976；near s≥0.9且endpoint detach，generated bond s≥0.75且保留flow图。无actual-source辅助权重自拟。正式loss等待A已评估recipe。

默认pilot配置243 updates=27种span/dt/query组合×9条轨迹；AdamW，history1e-4/DiT2e-4/decoder5e-5，BF16，teacher frozen。复用现有FrameJointTrainer，不引入并行trainer。pilot权重不会作为formal起点。

## 结果与成本

raw读取/对齐/压缩写出每条约0.46–2.88秒；这是CPU数据IO，不是模型吞吐。语法编译已通过。CUDA preflight按resources.json拒绝启动：A记录8卡均被其他用户占用，B UUID分配为空。

成功训练更新0，GPU用量0/48GPUh。teacher实际调用数、分段吞吐、峰值显存、CUDA等变/数值/梯度、generated运动、多lag指标、逐体系CI、逐残基RMSF和MSD曲线均未运行，不能填成0或通过。代码契约为训练observed2+targetQ、推理observed2+target0；这尚不是实测结果。

无新checkpoint，无短长优胜选择。M-index仅配置/pilot入口。没有AF3融合、复合物泛化或长rollout验证。

## 阻塞与下一步

已固定吸收A公共候选`eec76455a7d9a36723e7c1a2aad1043bba553198`，本地cherry-pick `7270e54`。A公共代码本身尚待CUDA验证；不是recipe release。

待B资源UUID分配后，先targeted CUDA smoke，修复后真实teacher parity/2+Q调用审计、3体系pilot和大小体系profile，再按最新固定commit做无实现上下文独立review并修复。正式M-short/M-long另外等待A已完成评估的immutable recipe。普通检查通过不需重复请求许可。
