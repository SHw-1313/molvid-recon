# 合并到根 AGENTS.md 的长期规则

本文件是合并片段，不是根 AGENTS 的替代品。保留根文件的容器、仓库组织与操作权限要求；已存在同义规则则不重复。

```markdown
## Maintain readable model code

- Keep the shallow root `molvid/` package. Put geometry representation,
  history encoding, flow prediction, coordinate decoding, losses and
  training orchestration in their named modules. Do not recreate legacy
  source archives or a parallel model framework.
- Make tensor shapes, physical units, packed-system masks and gradient
  boundaries explicit at public interfaces. Physical time and flow time
  are different quantities; use different names.
- Keep observed conditions separate from training targets. Generation
  accepts observed context and query times, never hidden future coordinates.
- Parse configuration once. Keep loss weights in one resolved source;
  behavior must not depend on hidden defaults or scattered constants.
- Model forward methods compute model outputs; CLI files only construct
  and call components. Prefer composition and direct calls over registries,
  dynamic dispatch, monkeypatches or deep pass-through wrappers.
- Reuse compatible checkpoint weights with an explicit load report.
  Distinguish warm start, exact resume and a changed-objective continuation.
- Repair or implement the requested change before running targeted checks.
  Test meaningful numerical risks on CUDA; do not add redundant whole-suite
  gates or silently fall back to CPU model execution.
- Keep the actual architecture and one training-step dataflow readable in
  docs and the model inspector. Put transient experiment settings/status
  under the task directory, not into this instruction file.
```
