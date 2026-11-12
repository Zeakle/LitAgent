# 阶段超时配置

默认超时由 [config/default.yaml](../config/default.yaml) 设置。修改后重启 live 服务；已有运行仍绑定原配置与 fingerprint，不能用新时限恢复原终态或重发结果未知的请求。

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

已经发送的请求被取消或超时时，系统保留未知用量占用，避免自动重发。请求额度另见[模型用量与运行额度](output-capacity.md)。

## 独立双循环配置

[独立双循环配置](../config/separate-review-evaluation.yaml)将提取请求时限设为 216 秒，正文生成和修订设为 720 秒，摘要、Reviewer 和 Evaluation 请求设为 360 秒。完整 Evaluation 上限为 8,640 秒。
