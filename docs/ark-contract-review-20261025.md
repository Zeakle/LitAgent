# 方舟字段与保守记账修复

本次为步骤181的**离线修复**，没有新模型调用。真实010仍为0/3：首HTTP400拒绝 `background`，未生成正文或进入Evaluation。原状态、原包、评分和费用不改。

## 请求映射

新映射 `ark-responses-mapping-v2` 仅发送当前功能需要且在方舟自身接口中定义的字段：`model`、`input`、`text`、`max_output_tokens`、`temperature`、`store`、`stream`、`thinking`、`reasoning`。同一份 strict schema、完整科学材料、原输出额度和科学门禁保持。`store=false`、`stream=false`；没有后台轮询或自动回退。

移除了未支持的 `background`、`truncation`。宿主继续检查完整输入和上下文预算；不再宣称有服务端显式禁裁剪参数。依据为[方舟接口](https://console.volcengine.com/ark/region:cn-beijing/docs/create-model-responses-api)及[官方SDK签名](https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkarkruntime/resources/responses/responses.py)。独立目录位于 `tests/fixtures/ark-responses-fields-20261025.json`，HTTP替身会拒绝目录以外字段。

目录来源是2026-10-25读取的官方网页快照；直接获取SDK固定Git提交时DNS失败，**没有取得提交SHA**，不能把该目录说成已固定的官方源码版本。该限制不妨碍将已观察字段范围固定为本地回归契约，但真实服务兼容性仍需单独验证。

新服务端profile为 `writer_reviewer_low_responses_v2`。旧v1描述仍可读，新prepare/live不能用它继续调用。默认Chat和普通配置对象不变；新旧映射的checkpoint身份不同。业务正文协议仍为 `grounded-writer-v1`。

## 未知费用不能退款

旧 `accounting_history` 默认继续拒绝未知用量。仅新profile选择 `carry_terminal_rejection_v1`：必须收到可验证的HTTP 4xx拒绝包装，原请求、响应、操作配对、模型路由、原文/schema、输入SHA、完整预留、预算公式、终态和连续历史均一致。超时、未完成任务、缺原包、篡改、重复或占用不一致仍拒绝；不会重发原请求。

010没有保存完整预算快照。对它的继承明确标记为 `reconstructed_from_frozen_records`，使用冻结预算源码、请求/响应及原终态交叉重建保守记账，**不是补写原始快照**。新driver另外持久化真实 `budget_snapshot`，已有快照必须与费用、请求和占用一致。

只读核对完整12个父记录后，累计保持 **753请求／4,238,381已知tokens／1,086,543未知保守占用**，其中010的56,129完整保留。未知不是实际零费用，reasoning也不重复计费。明确非终态或未知状态中的计数视为暂时量，保留完整预留；合法终态的坏正文仍只计一次已知usage。

旧HTTP记录未保存origin时，只能以冻结配置说明目标来源，receipt标为 `frozen_config_origin_only`；不能声称已捕获当时的实际Host。新私有receipt记录不含凭证的origin。完整原包不进入公开结果或LangFuse。

## 展示与验收

导出现在从精确归档及匹配receipt自动投影HTTP状态、受限错误码和参数名，同时保留 `usage_unknown`。JSON、CSV、Markdown和HTML同源；不公开自由错误文案、原包或论文材料。历史010用新版只读渲染验收，原公开包未覆盖。

- 相关回归118通过；默认离线 `pytest -v` 为 **4509 passed、25 deselected、8既有warnings**。
- 5个改动Python文件Black/isort及 `git diff --check` 通过；40个保护文件和481份历史源码副本SHA保持。
- 005六份历史输出经实际SDK和Worker重放，完整输入、schema及正文SHA不变；Reviewer仍预留107,328。ViT/NeRF只检查首请求，不能承诺后续全文和完整复评预算。
- 桌面、390px、中英文、Enter切换、Space展开、页面无水平溢出及console检查通过。界面工具两次审批等待超时，均核对状态后按允许次数重试成功，没有绕过拒绝或保留失败页标记。

证据保存在本地 `artifacts/acceptance/ark-contract-20261025/`。这些检查证明适配、记账和展示契约，**不证明真实正文质量改善**。新的三域生成尚未执行，Phase14.5保持 `in_progress`。
