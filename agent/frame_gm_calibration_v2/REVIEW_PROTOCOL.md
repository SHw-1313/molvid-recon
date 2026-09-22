# 独立审阅、修复、训练闭环

## 角色与隔离

主 session：唯一模型代码写入者，实现/跑pilot/修复/训练/评估，负责流程不断档。
reviewer：新建 session，不继承主 session 聊天、不读取主 session 日志或其思维过程；只读固定代码、已确认规格和可核验pilot证据。最终输出审阅文件。

独立是指独立会话和自行读代码，不要求 reviewer 换模型或另开账号；沿用本机可用模型即可。不要为了“更聪明”擅自修改账号/模型访问权限。根AGENTS正常读取，但review角色指令优先于任务目录中的实现步骤。

P1无需独立审阅。P2/P3各在实现+pilot后审阅一次；有实质性修复时追加定向复核。不要对每个函数或配置开一轮review。

## 主 session 发起审阅

1. 写实现摘要和真实pilot证据；不向reviewer灌输“已经正确”的判断。事实包括命令、退出码、代码SHA、配置SHA、数据来源、loss/梯度、CUDA设备、profile、已知未执行项。
2. 将本轮源码、规格、配置提交到本任务分支；仅提交本任务文件，避免混入用户改动。记录 `candidate_commit` 和 `base_commit`。
3. 建立独立detached worktree：`git worktree add --detach REVIEW_WORKTREE CANDIDATE_SHA`。源码须干净，review期间不修改它。
4. 在实现worktree的 `agent/frame_gm_calibration_v2/reviews/P2/r01/`（或P3）写 `review_request.json`。路径必须是reviewer进程可见的绝对路径，尤其注意容器与宿主映射。
5. 运行本包 `tools/run_fresh_review.py --request ABS_REQUEST_PATH`。调用Python也遵守当地repo的enter-container/torch-ito约定；若Codex只装宿主机，使用下方同义shell命令直接在宿主运行，不为了脚本再装Codex。
6. 读取生成的 `REVIEW.md`，按issue ID修复，写 `FIX_RESPONSE.md`。不让用户转发文件。

新建session的等价命令结构（所有路径已由主session解析）：

```bash
codex exec --sandbox read-only --ephemeral \
  -C "$review_worktree" \
  --output-last-message "$review_output" \
  - < "$rendered_review_prompt"
```

不得使用 `exec resume` 或转发聊天history。`--ephemeral`只控制会话记录持久化，不代表取消系统指令/项目指令，也不是独立性的唯一保证。prompt明确禁止检索此前Codex聊天和memory来推断实现是否正确。

本地CLI不支持某flag时先读 `codex exec --help`，使用同等的新会话/只读/文件输出功能；不得通过去掉只读限制或绕过权限使其工作。不要改全局CODEX_HOME/HOME来“清空记忆”。官方参考：
https://developers.openai.com/codex/noninteractive
https://developers.openai.com/codex/cli/reference

## 无嵌套 CLI 时的替代

若平台明确支持新agent且可设置 `fork_context=false`/`fork_turns=none`，主session可用该能力发同一份review prompt和事实包；等待它输出完整REVIEW.md内容，再原样落盘并记录agent/session identity、候选commit。必须确认真的未继承主会话，不能只写一句“假装独立”。

CLI和独立agent均不可用才人工打开新session：粘贴 `REVIEW_PROMPT.md` 和request路径，reviewer把最终文本保存/导出为REVIEW.md；主session恢复后继续。报告具体工具/权限障碍；不得主session自审代替。

## 审阅结果与自动推进

报告必须包含 `REVIEW_STAGE`、`REVIEWED_COMMIT`、`VERDICT`。允许：

- `PASS`：没有影响本阶段训练有效性的未解决问题，可以训练。非阻塞建议可记录后处理。
- `FIX_REQUIRED`：可定位的实现/科学问题，由主session修复。
- `BLOCKED_ENV`：缺少实际审阅所需文件/证据或读取环境，不等价于PASS。

主session必须阅读问题内容，不能只grep PASS。软件有效性审阅不要求新模型在128步pilot就超过成熟baseline；科学负结果不是审阅失败。

修复闭环：保留r01报告→按issue ID写FIX_RESPONSE→修复→定向CUDA/pilot复验→新commit→新reviewer定向复核r02。涉及梯度、源分布、数据、mask、loss、checkpoint语义的修改必须复核；仅文档排版/输出路径修正可由主session记录验证，注明r01依据如何适用于final commit。

最终 `review_clearance.json` 写 approved code commit、review文件hash、修复commit范围、已关闭issue、剩余非阻塞项、实际训练配置/manifest/预算hash。该文件是证据索引，不是reviewer写给人类的额外审批请求。

正式训练从获审阅的代码快照运行。实现session随后开发P3时，不能在正在运行P2的import源码树上改动；可给训练创建只读用途的detached run worktree，或等P2结束再继续改。配置在启动时复制并hash到run，不在run中途热修改。

同类故障两次修复仍无实质进展，定位根因并写状态；不无限“review→盲改→重训”。确有权限/缺文件/预算障碍时才交回用户。普通配置、可复现bug和科学比较的选择由主session自主处理。

## 只保留会改变结论的检查

实现/repair在前，CUDA smoke在完整科学pilot之前。定向数值检查覆盖本次改变的风险；没有新的证据需求不跑全量回归。reviewer可要求最小复现，但不得把格式、命名喜好升级为正式训练阻塞。

审阅期间主session可准备下阶段文档、读代码、做不依赖候选模型的数据汇总；不抢跑本轮正式训练。
