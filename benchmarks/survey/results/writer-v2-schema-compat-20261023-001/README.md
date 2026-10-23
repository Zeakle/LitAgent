# Writer 原生 schema 兼容验证：失败记录

这是一次固定公开 few-shot 材料的**协议观察**，不是完整 Survey 或科学质量 Benchmark。结果为 `blocked / writer_json_invalid`；没有生成完整报告，没有运行 Evaluation，科学支持率不可用，不能交付。

| 项目 | 实际结果 |
|---|---|
| 执行源码 | `9c95e52fe3c0b5680e0f9c6450d6957230106a5b` |
| 模型 | `doubao-seed-2-1-pro-260915` |
| 变化 | 仅供应商原生 schema 投影；完整本地 schema、消息、来源及模型不变 |
| 请求 / 重试 | 1 / 0 |
| 已知用量 / 新增未知占用 | 12,537 tokens / 0，用量完整 |
| 用时 | 69.221 秒 |
| HTTP / finish_reason | 200 / `stop` |
| 内容 | 5,759 字符，offset 5,759 / column 5,760 解析失败 |
| 原包与 SDK | content 逐字相同 |
| 后续阶段 | 摘要、Reviewer、Evaluation 未运行；Judge/Jev 关闭 |

原始 content SHA-256：`0c126776cfa94127ce595b8eeae931153124f27a59b9697aa96c9937da9d41c9`。

请求总上限 120,000 tokens、一次请求、输出上限 20,000；实际发送前预留 56,457。预留和上限不是实际费用。累计继承账本为 727 模型请求 / 3,853,754 已知 tokens，历史未知保守占用 1,030,414 保持不变；没有把协议观察计为新增完整 Survey 轮次，也没有重计历史提取。

本次 HTTP 原包已经包含不完整 JSON，可排除 SDK 对该响应正文的截断或改写。schema 兼容适配没有解决本次失败，远端具体原因尚未确定；不修补括号、不放宽验收、不追加重跑。该结果不能证明科学内容支持或不支持。

默认离线 4,211 passed / 25 deselected / 8 既有 warnings，相关格式检查通过。463 个冻结源码文件和 7 个本轮保护文件保持。公开数据为[同源 JSON](result.json)，不含论文原文、原始响应、凭证或私人运行记录。

本机 [LangFuse trace](http://localhost:3000/project/litagent-local-project/traces/6f20e8fe5be71d3af9d8d77cc98ffec5) 读回一个 generation / 12,537 tokens，科学输入输出为空；未声称验证完整 trace 树。完整请求和 HTTP 响应仅留本地。

当前三域质量目标仍未完成；旧三域结果及失败分母保持，Phase 14.5 为 `in_progress`。本条记录没有新增三域或正式十五题执行。
