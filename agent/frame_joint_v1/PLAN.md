# 执行导航

## 阅读顺序

| 文件 | 负责的内容 |
| --- | --- |
| `ARCHITECTURE.md` | 数据流、shape、模块去留、物理时间、梯度职责 |
| `CHECKPOINTS.md` | 指定旧 codec、提取规则、warm start 与 continuation |
| `TRAINING_EVAL.md` | 数据、loss、预算、成对 bond 续训与判读 |
| `experiment_spec.json` | 本轮统一的数值设置；这是计划规范，不是现有 CLI 配置 |
| `READABILITY.md` | 代码组织和人类审查要求 |
| `TASKS.md` | 实现顺序与交付清单 |
| `HANDOFF.md` | 实际完成状态、命令、结果；持续更新 |

`REVIEW_PROMPT.md` 给独立审阅 session；`TRAIN_PROMPT.md` 在实现可用后启动长训练；`ROOT_AGENTS_APPEND.md` 只合并长期规则，不直接覆盖根文件。

## 按任务阅读现有源码

| 工作 | 首先阅读 |
| --- | --- |
| 逐帧 teacher | `molvid/codec/model.py`, `codec/heads.py`, `spatial/encoder.py`, `checkpoints.py`, `geometry/coordinates.py` |
| 历史 memory | `codec/state_detail.py`, `codec/haar.py`, `codec/types.py`, `equivariant.py` |
| 新 latent 和时间 | `latent/types.py`, `latent/adapter.py`, `latent/statistics.py`, `latent/conditioning.py` |
| Frame DiT / decoder | `dit/model.py`, `dit/blocks.py`, `dit/backend.py`, `codec/heads.py` |
| flow / 生成 | `flow/objective.py`, `flow/source.py`, `flow/sampling.py`, `generation.py` |
| 联合训练 | `training/batches.py`, `training/dit.py`, `training/codec.py`, `losses/geometry.py`, `losses/reconstruction.py` |
| 数据与 DDP | `data/preprocess.py`, `data/manifest.py`, `data/sampling.py`, `data/batch.py`, `cli/train_dit.py`, `runtime.py` |
| 评估 / 性能 | `evaluation/runner.py`, `evaluation/motion.py`, `evaluation/geometry.py`, `tools/inspect_model.py`, `benchmarks/profile.py` |

源码路径以上均相对于仓库根；只读相关定义与调用链，不要求先逐字读整个仓库。结果背景读取 `results_archive/KEY_RESULTS.md` 和 `results_archive/runs/dit_architecture_sequential_v1.md`。不要恢复被清理的旧测试或源代码大目录。

## 已确定与未确定

已确定：TorchMD、历史压缩/未来逐帧解耦、真实时间条件、新 decoder、联合训练、bond 续训双臂、固定数据 split、根包浅层布局。

需现场解析：真实仓库/容器路径、指定 checkpoint 的实际挂载位置、192 体系的训练集 manifest、GPU ID、最大体系的实际吞吐、校准后的 bond 系数。记录这些值即可，不重新做架构选择。

旧代码保留为可运行对照；新旧分流发生在构造入口与显式类型边界，不允许每个 block 都充满 `if old/new/ratio`。只对这轮需要的旧 baseline 路径做兼容，不以新架构开发为由重做整个历史迁移工程。

## 模型选择依据

主实现建议 Sol / xhigh；复杂梯度、泄露与 checkpoint 审阅建议 Astra / high；固定配方训练与日志汇总可用 Luna / high。这里的任务分工是工程判断，不是这些模型在 molvid 上的实测排名。

官方模型定位与设置说明，核对日期 2026-09-20：
- https://learn.chatgpt.com/docs/models
- https://learn.chatgpt.com/docs/config-file/config-reference

CLI 可在本机实际可用时使用：`codex -m gpt-5.6-sol -c model_reasoning_effort='"xhigh"'`。也可在 `/model` 中选择。不要为本任务修改全局模型配置、权限或沙箱设置。具体可用型号以用户客户端为准。
