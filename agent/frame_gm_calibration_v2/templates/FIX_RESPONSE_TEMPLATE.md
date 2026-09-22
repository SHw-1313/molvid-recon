# 对独立审阅的修复回应

- 原 REVIEW.md 路径/hash：
- 原 candidate commit：
- 修复后 candidate commit：
- 本轮配置/data/预算是否改变：

| Issue | 判断 | 修复文件/commit | 定向复验及结果 | 是否需独立复核 |
|---|---|---|---|---|
| R1 | confirmed / rebutted-with-evidence | | | |

不能只写“已修复”。反对审阅意见时给源码或数值证据，再让reviewer确认；不能删掉原报告。

## 未解决项

说明原因和对哪一步的影响。非阻塞项明确标记。

## 新增风险

记录修复改变的梯度、mask、loss、数据或checkpoint行为；影响pilot数值时更新相应pilot，保留原证据。

## 继续动作

需要复核则发起r02；通过后自动运行正式训练，不再索取重复授权。
