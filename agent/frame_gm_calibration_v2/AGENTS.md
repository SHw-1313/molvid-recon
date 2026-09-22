# 本任务目录约定

当前任务授权由根 `CODEX_FRAME_GM_CALIBRATION_PROMPT_260922.md` 给出。前三阶段包含训练，不沿用旧 tiny 后暂停。

- 根 `molvid/` 保持浅层组织；在现有类型、训练器和评估路径中扩展，不另造平行框架。
- 只读 reviewer 与实现 writer 职责分开。审阅者读 `REVIEW_PROMPT.md`，不要执行主实现 prompt。
- 本目录保存任务状态、实验配置说明和审阅记录；训练大文件仍放 `runs/`。
- 这是目录级文件，不会自动约束所有 `molvid/` 源码。主执行 prompt 要求显式读取；长期原则按 ROOT_AGENTS_APPEND 合并到根。
- 科学模型和数值检查走 CUDA；元数据、git、文件汇总可以 CPU。不要把“CUDA 模型执行”误解成禁止 checkpoint 的正常 CPU 序列化。
- stage / commit / config / manifest / checkpoint 的身份必须记录，未知写 unknown，不编造。
- 审阅结论、修复回应、实际训练结果分文件保存；保留原审阅，不能覆盖它来消除问题。
