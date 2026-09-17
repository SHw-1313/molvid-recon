# B tasks

- [x] B001 固定commit建立独立worktree，记录用户变化/环境/输入，追加本树根阶段说明。
- [x] B002 冻结reference实现/数学contract；建立strict checkpoint兼容与非退化CUDA fixture。
- [x] B003 向量化layout、pool、sample/time打包attention、broadcast；保持既有scalar/vector语义。
- [x] B004 重复encode去重、派生metadata复用、backend/profile CLI与计时内存隔离。
- [x] B005 CUDA forward/gradient/update、mask/SE3/cache测试和真实R4/R2 smoke；受影响回归。
- [x] B006 同权重同输入reference/optimized短profile，分段trace、内存/吞吐报告。
- [x] B007 证据/DECISIONS/HANDOFF、diff检查、本地范围commit，停止。

Current status: B_BACKEND_READY_FOR_REVIEW

Evidence: evidence/dit_factorized_backend_v2/20260909_backend_v2/SUMMARY.json and SUMMARY.md.
CUDA parity: 4 passed; affected regression: 15 passed; real smoke and 200-step profile: PASS.
