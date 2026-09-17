# 决定记录

## 已准备决定 — 2026-09-10

1. 直接沿用B分支，不新建分支/worktree，不整体merge A；选择性引入诊断文件。
2. 修复、关键CUDA检查、source检查和本轮限定R4两臂训练均在执行范围；旧阶段等待状态不适用。
3. 首选repeat-last-coordinate-reencoded中心，block-state-zero-detail是声明的替代诊断；按实现正确性选择，不用真实未来RMSD排名。
4. 两臂source在标准化空间分别为eps和m+eps，sigma同为1；两臂history相同。
5. 新网络初始化完全一致；不从旧已训练DiT继续其中一臂。
6. R4主训练；R2只做backend等价/兼容检查；R1/pooling/no-temporal不训练。
7. 不新增几何loss、不改field权重、不重训codec、不扩模型或改变物理时间数据。
8. Gaussian默认兼容旧RF；新source合同必须进入checkpoint/resume。
9. 冻结codec后可以缓存，但有界、FP32存储、验证packed batch等价，不能重排数据。
10. 4500为历史检查点，共同终点优先20000、其次10000，预算不足4500标PARTIAL；不宣称固定步数收敛。
11. 主要比较共同固定步数checkpoint，独立的RF loss不跨source直接选赢家。
12. 无test payload、无DDP/额外coding agent/自动push。数值CUDA、元数据/I/O可CPU。

## 实施追加

以下待执行者填写，仅追加实际事实，不修改上述决定以迁就结果：

- 实际引入文件/来源与B新增commit差异：
- source中心选择及模板/边界证据：
- cache模式、容量与吞吐：
- 共同步数、资源预算及估算依据：
- 必要实现偏差及原因：
