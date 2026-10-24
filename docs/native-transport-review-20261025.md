# 同 schema 的供应商传输兼容审查

**本步仅离线审查，未实现生产适配，也未调用真实 Responses 服务。** [009](../benchmarks/survey/results/frozen-writer-three-domain-20261025-009/README.md)仍为严格交付 0/3，Evaluation 未开始。low 的实际传递和 reasoning 明细已经验证，不能继续把调参当作已证明的协议修复。

保留 `grounded-writer-v1` 的全部字段、层级、出处、完整原文和本地校验，只审查现有 LLM 层内的供应商字段映射。没有新 Agent、模型、来源、正文协议或更高预算。Responses 同样是 beta，官方支持该接口不等于真实输出一定可靠。[官方结构化输出说明](https://docs.volcengine.com/docs/ark/responses-api-structured-output?lang=zh)

| 原 Chat 请求 | 候选 Responses 请求 | 保持的语义 |
|---|---|---|
| `messages` | `input` | system/user 角色及 content 逐字相同 |
| `response_format.json_schema` | `text.format` | 同一 schema、名称与 `strict=true` |
| `max_completion_tokens` | `max_output_tokens` | 同数值的思考＋回答总上限 |
| `reasoning_effort=low` | `reasoning.effort=low` | 同一请求强度 |
| `thinking.type` | `thinking.type` | 显式开启思考 |

离线候选请求显式配置 `store=false`、`background=false`、`truncation=disabled`，避免把服务器存储、异步轮询或材料裁剪作为解决方案。它仍需真实接口验收，当前不能声称服务端已接受这些设置。

使用安装的 OpenAI SDK 及 `httpx.MockTransport` 构造一次模拟 HTTP 调用：输入取自 009 原请求，完整科学输入与 schema 相同。实际模拟请求 35,242 字节，加 20,000 输出预留仍在原保守预留 56,129 内；未裁剪原文或降低输出额度。这不保证未来章节、完整复评或修复都能装入预算。

SDK 的原响应接口可以取得 HTTP body；当前版本的 `parse()` 不需要 await。模拟响应故意设为 `status=completed`，内部文本保持 009 原非法 JSON。SDK 取得的文本逐字相同，SHA 仍为 `ba8a6c250fc89875cc4a42593956cbbfe833f937b18ab5e59feedac28c126855`，宿主依然返回 `writer_json_invalid`。没有补括号、选取部分内容或接受不完整正文。

私有证据位于 `artifacts/acceptance/native-transport-review-20261025/`；**1 次模拟 HTTP、0 次外部模型请求**，不是新的模型生成、科学判断或真实服务兼容性结果。完整原输入和模拟原包不公开。本步没有改生产代码，原 4412 项默认测试对应的源码保持；当前不以该探针替代后续 `tests/` 验收。

如果实施可选适配，必须先覆盖 completed/incomplete/refusal/failed/cancelled/未知 output 的解析、完整模型身份、usage 主计费及 reasoning 子集、私有原包与脱敏 trace、预留／取消／超时／零重试。已知 usage 的协议失败不能再加一次 unknown 占用；不得自动回退 Chat 或重新发出结果未知的请求。默认 Chat 和旧归档语义保持，API 方式必须进入冻结身份。

候选方向通过离线审查只说明值得实现受控映射，不证明它解决 009，也不保证三域严格通过。Phase 14.5 保持 `in_progress`。
