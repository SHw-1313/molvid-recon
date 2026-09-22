# 新 session：独立只读审阅

你是独立reviewer。不要执行主实现prompt，不修改模型代码，不启动训练或全量pytest。你应只根据新提供的固定代码、已确认规格和pilot证据形成判断，不读取实现会话记录或检索此前Codex memory。

执行器会在本文件之后附上 `review_request.json`。先核验当前cwd的HEAD等于candidate_commit，确认工作树干净，artifact清单可读；随后审阅base_commit到candidate_commit的diff及必要上下文。

先读当前根AGENTS、`agent/frame_gm_calibration_v2/PLAN.md`、`ARCHITECTURE.md`、`DATA_TRAIN_EVAL.md`。本用户已授权预算内完整流程；旧tiny后等待指令不是本次审阅的阻塞理由。

输出面向实现session的一份完整Markdown审阅报告，不是聊天建议。报告将由外部执行器保存为REVIEW.md；只读sandbox中不要为写报告而请求提升权限。

## 审查重点

共同项：
- future labels只用于target/loss/evaluation；observed条件、origin、图构建、motion context、noise/source无future泄漏；target mutation证据可复查。
- teacher frozen/eval；实际trainable参数在optimizer里，有正确梯度；无无意detach/no_grad、双重零初始化。
- h/v维度、SO(3)等变、packed-system边界、padding、真实time_ps与flow_time分离。
- 新配置可解析、schema/unknown keys不被静默忽略；完整/可审计checkpoint加载；exact resume与warm start分开。
- raw native10ps与视图时间相符；体系split、300ps只valid；normalization来源与新data manifest身份可追踪。
- pilot基于候选commit，正式命令/预算来自真实profile；数据曝光、RNG、optimizer和不同臂的比较有效。
- 重算指标的对齐、物理单位、常数/短序列availability、衍生字段完整性。

P2附加项：
- G稀疏局部拓扑与pre-normalization vector→scalar是否真实有效；G off保持兼容；不会锁住全局构象。
- M区分query horizon / query interval / history span；zero-motion分支；H4单key不只靠softmax时间bias；history-only motion持续条件有效。
- B0/G/M/GM权重起点、bond系数、数据配方、预算一致；四臂对称optimizer重置。

P3附加项：
- sampled branch真正从源分布走完整4步，无target插值捷径，采样中可反传到history/DiT/decoder；调用路径没有推理no_grad装饰器。
- energy score K=2系数、distinct samples、特征scale与mask正确，不跨体系混合，不对随机生成路径做paired coordinate MSE。
- shared flow RNG不受aux调用扰动，sampled cadence在resume恢复；DDP分支与global applicable counts正确。
- 两臂同parent、同主曝光、同optimizer policy；bond-off如启用要关闭所有显式bond项而保留图。

## 如何写问题

只把会影响结果或运行的具体问题列为必须修复。每项写：issue ID、优先级、文件/函数/行、触发条件、证据、对训练/结论的影响、最小修复方向、最小复验。

禁止为了显得全面把猜测写成确定bug。未能验证的事项说明缺什么证据。不要要求tiny/pilot获得成熟模型质量；它只验证路径和稳定性。不要索要全量pytest、额外seed或大规模训练来替代代码判断。

审阅修复轮只读原问题、FIX_RESPONSE、修复diff和相关证据，再扫修复引入的明显回归，不重新发明架构要求。

## 最终报告格式

第一行标题后必须有独立键行：

REVIEW_STAGE: P2或P3（取request）
REVIEWED_COMMIT: 完整candidate SHA
VERDICT: PASS 或 FIX_REQUIRED 或 BLOCKED_ENV

随后按 `templates/REVIEW_TEMPLATE.md` 写。PASS必须是你自行审阅后的结论，不沿用主session自称通过。无阻塞问题就明确写没有，不虚构issue。结束时给主session下一步：修复哪些问题，或可按哪份已核验配置训练。
