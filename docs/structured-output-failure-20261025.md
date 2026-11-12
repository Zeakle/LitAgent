# 011结构化输出故障审计

011使用提交`0b90025`，未重搜或提取，完整输入、模型、输出额度和科学门槛与冻结清单一致。真实结果见[011记录](../benchmarks/survey/results/frozen-writer-three-domain-20261025-011/README.md)。严格交付0/3，Evaluation未执行。

## 可以确认的边界

- 010的未知字段拒绝已消失；011两次请求均HTTP200，原包均`status=completed`，模型与冻结Pro一致。
- 引言合法；taxonomy请求schema要求数组，响应是包含`sentences`的对象。原包、adapter输出、journal原文逐字相同，故该类型偏差已经存在于服务原始响应，不是宿主遗漏括号。
- 两次请求均发送`text.format.type=json_schema`、`strict=true`及正确数组schema，服务还逐字段回显了相同的`text.format`，`service_tier=default`。当前证据不支持“本地没有传schema”或“回显的是另一个章节schema”的解释。
- 输出1513和1474均小于20000；总99.5秒、没有超时或输出截断。不能用增加timeout或输出上限作为本次已验证修复。
- 本地拒绝而没有自动包一层数组是预期行为。类型合法也不等于事实获原文支持。

## 官方契约与不确定性

[方舟Responses结构化输出文档](https://console.volcengine.com/ark/region:cn-beijing/docs/responses-api-structured-output)（2026-10-25读取，页面更新2026-10-22）说明`text.format`的schema、strict和类型约束，同时标注beta阶段及可用性限制。文档示例使用disabled，不构成禁止enabled的声明。页面还列出TPM保障包限制；当前原包只证明实际请求/响应，不能证明账户购买类型。

不能从单次类型偏差判定供应商内部根因，也不能把beta提示解释成所有失败必然合理。schema名称复用、思考强度、平台负载等均未建立因果证据。当前不更改模型、业务schema、输入材料或科学门槛，不追加付费试到通过。

## 离线回归

`tests/test_responses_shape_regression.py`只替换HTTP，保留安装SDK、Client、WriterSession、真实schema、完整合成fixture材料与预算。复现对象代数组及相关类型错误，确认拒绝、原包保存、费用只计一次、无后续章节/摘要/重发。合法数组正例完成五章，但不宣称科学正确。结构fixture不是011完整原始正文，也不是人工科学金标准。

011完整原包仅在本地重放，原请求/响应SHA绑定、原文无损和错误路径另存私有验收证明。公开包只含类型位置、SHA、状态和统计。历史40保护及新增011四文件保持，不改旧费用或评分。

验收：新增6例及相关测试78通过；默认离线4515通过、25排除、8既有warnings，Black/isort及diff检查通过。011私有完整重放2个模拟HTTP、0外部请求，不形成新增费用或科学结果。桌面/390px、中英、Enter/Space和console核对通过。

## 仍未解决的恢复边界

当前`initial_body`仅保留枚举引用错误供既有完整正文修复；类型错误立即停止。`patch`需要合法正文及明确句子位置，不能把错误类型章节偷偷转换后送入该入口。这是现有有界修复的覆盖边界，不能把“两次上限存在”写成“所有协议错误均能修复”。后续须先审查现有控制器的目标绑定、原稿保护和完整后续预算，再决定能否在原协议内覆盖该类错误；本次没有实现新的恢复路径，生产行为未改变。
