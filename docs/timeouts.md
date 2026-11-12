# 阶段超时配置

`config/default.yaml` 中的超时由服务端配置控制。修改后重启 live 服务；已有运行仍绑定原配置与 fingerprint，不能用新时限恢复原终态或重发结果未知的请求。

| 阶段 | 配置 | 默认时限 |
|---|---|---:|
| 查询规划 | `llm.request_timeouts_seconds.planning` | 120 秒／请求 |
| 外部检索、RAG 召回、去重、图分析 | `orchestrator.task_timeouts_ms` | 各 120 秒／任务 |
| 相关性筛选 | `orchestrator.task_timeouts_ms.relevance_gate` | 180 秒／任务 |
| 提取完整 Worker（全文准备、排队、批次、解析） | `extractor.task_timeout_ms` | 20 分钟 |
| 严格提取单批策略 | `extractor.automatic_timeout_seconds` | 210 秒 |
| 提取模型请求 | `llm.request_timeouts_seconds.extraction` | 180 秒 |
| Pro 正文生成 | `llm.request_timeouts_seconds.synthesis` | 300 秒／请求 |
| 摘要与结论生成 | `llm.request_timeouts_seconds.summary` | 180 秒／请求 |
| Reviewer | `llm.request_timeouts_seconds.reviewer` | 180 秒／请求 |
| 整体修订／定向修复 | `llm.request_timeouts_seconds.revision` / `repair` | 各 300 秒／请求 |
| 正文生成、审阅和修订链 | `adversarial.task_timeout_ms` | 30 分钟 |
| DAG 编排 | `orchestrator.timeout_ms` | 70 分钟 |
| Evaluation 请求 | `llm.request_timeouts_seconds.evaluation` | 180 秒 |
| 完整 Evaluation（分批及全局核验） | `eval.total_timeout_seconds` | 30 分钟 |

DAG 时限不包括尾部 Evaluation；这些值不是整个 Survey 的统一墙钟上限。PDF 下载、解析及独立 Judge/Jev 继续使用各自既有配置。`extractor.per_paper_timeout_ms` 为旧策略单篇回退时限，不代替严格提取整批和 Worker 时限。

## 发送前检查与取消

请求继承 DAG、Worker、策略或 Evaluation 中最早的 deadline。新策略要求剩余时间至少能容纳完整单次请求时限加 `llm.request_deadline_margin_seconds`（默认 5 秒）；否则返回 `request_deadline_insufficient`，记录 `llm.not_sent`，不发送 HTTP，也不增加请求计数或未知用量占用。

已经发送的请求被取消或超时时，仍保留未知用量的保守占用，不自动重发。延长时限不会增加 500,000 tokens／64 请求预算、重试次数或科学验收容忍度。

配置会校验请求、提取策略、Worker、Evaluation 和编排时限的关系。单次请求最多 600 秒；任务使用毫秒、请求与策略使用秒。未配置的新字段保持历史默认与旧 fingerprint；当前 YAML 显式启用新时限，故新运行身份会变化。

## 能保证什么

这组配置用于减少固定 300 秒提取任务和固定 180 秒正文请求造成的工程性中断。离线测试验证接线、真实 SDK timeout 参数、旧身份兼容、发送前拒绝及发送后取消计账；不代表新时限已通过真实模型运行，也不能保证任何主题都能生成合格正文。
