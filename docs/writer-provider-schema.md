# Writer 供应商 schema 兼容边界

2026-10-23 的三个同材料观察中，原完整 schema 对应一次合法 JSON、两次末尾不完整 JSON，均未进入科学 Evaluation。这不能证明唯一根因。另一个可静态核对的问题是：应用把通用 JSON Schema 的长度、数量、去重关键字直接交给原生解码器，而 [方舟官方支持清单](https://docs.volcengine.com/docs/ark/structured-output-beta?lang=zh) 没有列出这些格式约束。

`grounded-writer-v1` 输出字段与本地完整 schema 保持不变。`ark-structured-schema-v1` 只适配发送给原生解码器的 schema：

- 保留 `type`、`properties`、`required`、`additionalProperties`、`items`、`enum` 和 `description`。
- `minLength/maxLength`、`minItems/maxItems`、`uniqueItems` 转为描述提示；本地完整 schema 仍逐项强制校验，包含来源枚举、引用去重、非空及容量限制。
- 未知关键字在发送前拒绝，必须显式设计和测试映射；property 名称不被当作 schema 关键字过滤。
- 供应商仍使用 `json_schema`、`strict=true`；收到的原始 JSON 不修补。来源、字段类型、必填字段和闭合对象逐值保留。

正文、摘要、Reviewer、修复共用 `provider_response_format`。预算和请求身份按实际发送格式计算；journal 分别记录完整本地 schema 与供应商 schema 的 SHA。响应必须通过原始 JSON、完整本地 schema、正文契约及后续独立科学核验。原生解码器的成功不授予科学支持身份。

新执行版本为 `grounded-writer-execution-v3`，profile 进入 `body_summary_v2` fingerprint；旧策略默认值和 fingerprint 不变，历史产物只读，新执行不恢复旧 checkpoint。

`probe_writer_v2` 清单分别保存 `local_schema`、实际 `response_format`、profile 和唯一允许变化的请求字段。它继承已验证的完整账本，不清除历史费用或未知占用。本步只进行一次新独立协议验证，120,000 tokens / 1 请求 / 20,000 输出 / 零重试，不自动启动三域或正式十五题。

回归位于 `tests/test_writer_v2_schema.py`、`tests/test_writer_v2_protocol.py`，并复用实际 Synthesis→Reviewer→Evaluation→唯一交付出口测试。模型能力、真实 JSON 稳定性与科学质量是否改善仍须分别验证，不能仅从兼容适配或离线通过得出结论。

## 有界真实结果

源码 `9c95e52` 的[单次兼容验证](../benchmarks/survey/results/writer-v2-schema-compat-20261023-001/README.md)仍返回末尾不完整 JSON：HTTP 200、`finish_reason=stop`，5,759 字符在 EOF 处解析失败。供应商 HTTP 原包中的 content 与 SDK 保存内容逐字相同；本次可排除 SDK 对正文的截断或改写，但不能据此定位远端的具体组件或内部原因。

仅原生 schema 投影变化，完整消息、来源、本地 schema、模型及输出上限保持。新增一次请求、12,537 已知 tokens，用量完整；按上限停止，没有重试或启动三域。兼容适配没有解决本次 JSON 稳定性问题，也没有产生科学 Evaluation 或可交付正文。默认离线 4,211 项通过只证明工程回归，不能替代真实正文质量结果。

进一步的[层级审查](writer-layout-review.md)发现：EOF 之前，后续章节已经被错误嵌入前章容器。现从同一宿主 schema 派生可信路径提示；字段和验证规则不变，不把原失败移动字段或补闭合符后接受。真实效果仍须固定输入验证。
