# 本任务执行约束

适用范围：`exp/dit-architecture-sequential-v1` 新 worktree 中，本 prompt 明确授权的实现和实验。执行前阅读根 AGENTS、入口 prompt、同目录 PLAN 与 EVALUATION。

- 只按 1→2→3→4 顺序实现候选；公共数据/评估/独立加载修复可先完成。
- 不动 A、B 工作区或正在运行的作业。历史 shared-adapter 模型不续训。
- 沿用 TorchMDNet、R4、16 帧、H4/H8、100 ps、原 train/valid split、冻结 codec 与 statistics。第 3 轮新增独立 refiner，不解冻原 codec。
- 第 1/2/4 轮严格同父模型、独立参数 storage、同训练暴露量对照。第 3 轮两 refiner 参数和训练配方相同，只变跨块连接 mask。
- 数值检查和科学计算用真实 CUDA；实现先于测试，针对性测试先于真实 smoke，再做受影响回归。不做 CPU 数值替代或无关全仓 pytest。
- 不读取 test 坐标；不得以 valid/test future 生成推理条件。训练目标可以使用训练 future，但 inference API 不接收未知 future。
- 已观测帧的 observation clamp、坐标输出 mask、time/gauge/molecule 隔离贯穿新模块；统一 tokenizer 不改为严格 history/future 两套 codec。
- 不引入 AF3/MSA、静态混合、VAE、额外几何 encoder、变 dt 数据或全量 scaling。
- 不启额外 coding agents。GPU 使用按空闲状态决定，同卡不竞争，不抢占。
- 保留每轮失败和不确定结果，不能为通过改阈值、偷偷增训单臂或用旧污染结果排名。
- 新建并持续更新本目录 `HANDOFF.md`；只追加实测证据。不改写 PLAN 中科学问题来迁就结果。
