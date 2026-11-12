# 同 schema 的供应商传输兼容审查

**步骤178离线通过；步骤180真实请求被 HTTP 400 拒绝，供应商字段兼容尚未完成。** [010](../benchmarks/survey/results/frozen-writer-three-domain-20261025-010/README.md)严格交付 0/3，正文和 Evaluation 未开始。low 的实际传递和 reasoning 明细曾在009验证，不能继续把调参或 SDK 模拟测试当作协议已修复的证据。

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

私有证据位于 `artifacts/acceptance/native-transport-review-20261025/`；**1 次模拟 HTTP、0 次外部模型请求**，不是新的模型生成、科学判断或真实服务兼容性结果。完整原输入和模拟原包不公开。177 没有改生产代码；其原 4412 项测试与该探针均不能替代 178 新适配器的验收。

步骤177确定的实施门禁包括 completed/incomplete/refusal/failed/cancelled/未知 output 的解析、完整模型身份、usage 主计费及 reasoning 子集、私有原包与脱敏 trace、预留／取消／超时／零重试。已知 usage 的协议失败不能再加一次 unknown 占用；不得自动回退 Chat 或重新发出结果未知的请求。默认 Chat 和旧归档语义保持，API 方式必须进入冻结身份。

候选方向通过离线审查只说明值得实现受控映射，不证明它解决 009，也不保证三域严格通过。Phase 14.5 保持 `in_progress`。

步骤180的真实010请求暴露了适配错误：OpenAI SDK 接受的 `background=false` 不被方舟服务接受，返回 HTTP400 / InvalidParameter / unknown field background。请求没有进入正文生成，服务没有返回 usage；1次外部请求保留56,129未知占用，不能写成免费或完整用量。技术停机阻止了另两题，0重试、0修复、0份正文，科学状态未核验。累计753请求、4,238,381已知tokens、1,086,543未知保守占用。

这是供应商字段审查遗漏，不能归因于模型能力。SDK 能序列化及 HTTP 替身通过，不证明服务支持所有 OpenAI 参数。重新核对的[方舟创建响应接口](https://console.volcengine.com/ark/region:cn-beijing/docs/create-model-responses-api)明确支持 `store=false` 和 `stream=false`，没有列出 `background`、`truncation`。后续先冻结供应商支持字段表，说明同步返回、完整输入及上下文拒绝的实现边界，并检查响应字段和未知占用继承；不能删除报错字段后重发010，或继续宣称已有服务端禁裁剪控制。


步骤178的运行时实现保留默认 Chat，只有服务端固定 `writer_reviewer_low_responses_v1` 清单选择 Responses。首次正文、摘要、Reviewer 及共享修复的接口选择进入日志和恢复身份；Reviewer 可以独立选择 Chat。Evaluation 继续走原接口。没有新增普通 API 配置字段，没有改变旧默认 fingerprint 或增加重试、材料裁剪和 JSON 补全。

正常化直接检查原 HTTP JSON，主 usage 与输出状态分别处理：非法正文、拒答、截断、失败状态、错误模型和冲突包装仍不能通过；完整用量计费一次，缺少主用量保留完整预留并停止。reasoning 明细必须是原始非负整数且不超过输出 tokens，只作可选明细，不能再计一份费用。接口不暴露思考摘要，因此摘要字符数标为不可用。完整请求、响应及白名单请求标识只保存在私有 journal，不进公开导出或 LangFuse。

真实005的六份历史输出经实际 Synthesis、安装 SDK 和模拟 HTTP 重放，完整 user payload 与 schema 不变，正文 SHA 仍为 `34e8fca38d54b469c098281a4e320798ac9ae6ccf75801991659adaab6002e4f`。六次请求预留分别为 56,129／58,775／64,036／71,282／77,208／58,648；Reviewer 为 107,328。实际模拟请求字节加同额输出均未越过原预留。ViT／NeRF 只构造首章请求，不能据此保证未来正文和完整复评预算。

直接回归位于 `tests/test_native_responses.py` 和 `tests/test_responses_profile.py`，覆盖真实 SDK→Worker→唯一报告出口→Evaluation→自动判定、科学失败与修复拒绝、方法记录缺失，以及预算、未知占用、不重发、身份和脱敏。正例使用受控的模拟科学判断，只证明系统接线；不构成真实模型质量证据。完整离线结果与界面记录保存在本地 `artifacts/acceptance/native-responses-20261025/`。旧009及全部历史费用保留，新的外部模型请求为0，真实严格交付仍为0/3。

178最终离线验收：默认 `pytest -v` 为 **4476 passed、25 deselected、8 既有 warnings**；13 个改动 Python 文件 Black/isort 和 `git diff --check` 通过。首轮环境端口错误未跳过，在允许临时本机回环端口后完整重跑通过。36 个保护文件与 478 份历史源码副本逐 SHA 不变；桌面、390px、中英文、Enter 切换、Space 展开和 console 检查通过。预览明确标注为离线契约样例，不冒充真实验收结果。
