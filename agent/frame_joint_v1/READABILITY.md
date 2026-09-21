# 可读性是本版交付要求

用户刚完成重构；新功能不能恢复为多层 wrapper、动态调度和混合职责的旧形态。

## 模块和入口

- 继续根包 `molvid/`；不创建 `src/`、`molvid/models/`、`legacy/` 或旧源代码 archive。
- `codec/frame.py` 只负责逐帧几何表示；`codec/history.py` 负责已观测历史；`codec/decoder.py` 负责 latent→坐标。
- 若需要组合类，可用根包 `molvid/model.py::FrameJointModel`，清楚持有 teacher/history/DiT/decoder；只做组合，不塞数据加载、loss、optimizer、评估和写盘。
- `dit/` 是可训练 flow 场；`flow/` 是插值、source、积分；`losses/` 是损失计算；`training/` 管优化；`generation.py` 管推理数据流；CLI 仅解析、构造、调用。
- 新旧模型仅在工厂/构造入口分流。不能把 frame_joint、R2、R4、refiner、各种阶段全部塞进一个万能 forward。
- 保持 composition；不为两个实现创建多层继承、registry/plugin 框架或支持尚未计划的十种后端。

## 函数和 tensor

- 一个函数一个主要职责；一屏看不清数据流时按业务操作拆分，不按任意行数拆成十层透传。
- 公开方法写清 tensor shape、单位、mask 含义、返回值；标出 packed N、sample index、vector xyz 轴。
- 使用 `observed_*`、`query_*`、`target_*`、`flow_time`、`time_ps`、`history_memory` 等语义命名。数学局部允许 h/v/s，不用 x1/x2/z0 同时指结构首帧和 flow 源端点。
- 显式区分 residual、position、vector feature；只许通道线性变换的地方不混 xyz。
- 核心数据契约使用少量 dataclass；禁止在主要前向路径传递无定义、不断长字段的 dict，也不为每个临时值创建类。
- 不 monkeypatch forward，不靠 hasattr/getattr 回退选择模型行为，不捕获广义异常后静默跑旧实现或 CPU。

## 配置和训练

- 配置在边界解析并验证一次。类型化设置传入模块；模块内部不重新打开 YAML，不通过环境变量悄悄改变科学行为。
- `observation.history_lengths` 与 `observation.history_weights` 表示 H 采样；历史坐标扰动若保留接口，另叫 `history_corruption`，默认关闭。
- loss 权重只有一个 resolved 配置来源；bond-off 必须覆盖所有 bond 项。不要靠搜索整个项目手动改三处常数。
- 梯度边界用正常 no_grad/detach 和显式参数分组表达；不靠 module.train()/eval() 冒充冻结，也不靠 callback 神秘切换 requires_grad。
- 一次 forward 的 batch→condition/target→flow→endpoint→decode→loss 在 trainer 中能顺着读下来。日志、校验和实验编号不淹没这条主线。
- GPU hot path 不做 .cpu()/.numpy()/逐 tensor .item()；必要 I/O、RNG 恢复和低频日志不是“全项目禁止 CPU”的理由。

## 人类可审查的交付

在 `tools/inspect_model.py` 扩展现有工具，输出真实组件树、参数量/可训练量、输入输出 shape、设备/精度、teacher hash 与梯度去向。不要输出一堆无关的大 tensor。

新增简短 `docs/frame_joint_v1.md`（由实现者按真实代码生成）包含：一张数据流图、组件到源码入口的表、一个训练 step 的调用顺序、实际 tensor shape、checkpoint 初始化关系、三条常用命令。该文档描述实现现状，不复制整份实验计划。

独立审阅者应能从这份说明和 model/trainer 主入口回答：条件来自哪、目标来自哪、噪声加在哪、预测什么、怎样变坐标、哪些参数拿到哪项 loss 的梯度。答不清要修代码/文档，不是再加十个测试掩盖。

根 AGENTS 保留短而长期的规则；实验日期、阶段参数、未完成项放任务目录。根 TASKS/HANDOFF 仅新增本轮摘要与链接，不复制整包。
