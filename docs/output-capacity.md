# 模型用量与运行额度

使用默认配置启动时，以 [config/default.yaml](../config/default.yaml) 中的值为准。配置中的运行额度是上限，质量页的用量记录来自实际请求。

| 配置项 | 作用 |
| --- | --- |
| `llm.max_tokens` | 模型请求的默认输出上限。Reviewer 与 Evaluation 有各自的输出设置。 |
| `adversarial.review_max_tokens` | Reviewer 输出上限。 |
| `eval.max_tokens` | Evaluation 单次模型调用输出上限。 |
| `safety.planned_run_limits.max_tokens` | 计划型 Survey 的模型用量上限。 |
| `safety.planned_run_limits.max_requests` | 计划型 Survey 的模型请求数上限。 |
| `safety.max_cost_tokens` | 常规用量保护配置。 |

默认计划上限是 **10,000,000 tokens、200 次请求**。[独立双循环配置](../config/separate-review-evaluation.yaml)显式选择独立的写作修订和验收修复额度，其计划上限为 **47,420,000 tokens、226 次请求**。

## 查看一次运行的用量

在质量页可以查看预算准入结果、已知用量和未知占用。写作开始前会为后续评估和修订预留额度。请求已发出但缺少用量回执时，系统保留未知占用并停止自动重发。Jev 的提取核验用量单独记录。

模型调用和任务的时限见[超时配置](timeouts.md)。评估与交付规则见[Evaluation 说明](evaluation-results.md)。
