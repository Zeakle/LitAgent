# 完整输入的单次写作协议观察

该诊断只验证 `grounded-writer-v1` 正文请求的 JSON/schema 和本地正文契约，不生成摘要，不调用 Reviewer、Evaluation、Judge 或 Jev，不产生可交付报告。协议完成时科学分数仍为不可用。

入口：`scripts/probe_writer_v2.py`。它从已终止的 `frozen-writer-three-domain-20261023-001` 读取既有 few-shot 输入，以当前 producer 重建完整请求，并逐值核对历史 SDK 消息、schema、模型、temperature 和输出上限。原始支持文本不截断，旧归档、原响应、评分及费用保持不变。

固定边界：

- Ark `https://ark.cn-beijing.volces.com/api/v3`，`doubao-seed-2-1-pro-260915`。
- 一次请求，120,000 tokens 总上限，20,000 输出上限，继承正文 300 秒超时，零重试。
- SDK 与本地闭合 schema、实际全文支持材料保持；不自动补括号、改字段或扩大来源。
- 新目录和独立身份，冻结源码及依赖；已开始或完成的请求均不重发。
- 已发出但结果未知时保留完整单请求预留。额度不是账单，已知用量与未知占用分开呈现。
- 不自动重启三题或正式十五题，也不把一次成功当作可靠性证明。

执行前先使用 `--prepare --root <新目录>` 冻结和预检；显式 `--live --root <同一目录>` 仅执行既有授权范围内的单次请求。普通 API 不提供任意本机材料路径入口。模型密钥使用现有服务端环境配置，不写入 manifest 或结果包。

完整 SDK 请求与供应商 HTTP 响应只保存在本地。捕获发生在 SDK 解析之前，记录原包 SHA、状态和供应商请求标识；不保存 Authorization 请求头。HTTP 跳转与 SDK 重试关闭。LangFuse 使用既有 metadata_only 脱敏记录。对外 JSON/Markdown 只公开状态、用量、SHA 和稳定错误码，服务端错误中可能回显的输入内容不公开。

`tests/test_writer_v2_protocol.py` 用真实 SDK、预算、ModelBoundClient、WriterSession 和归档消费者，只替代 HTTP 传输，验证合法与非法 JSON、未知用量、取消、模型身份、原包一致性、身份漂移、进程中断、不重发及公开错误脱敏。

当前依据及限制见 [计划审查](writer-plan-review-20261023.md)。只有新的真实观察完成后才能判断这次协议是否通过；科学质量仍须完整 Survey Evaluation 证明。
