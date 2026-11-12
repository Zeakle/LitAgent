# Phase 14.5 修复计划审查（2026-10-23）

## 结论

计划的来源边界、宿主引用、严格 Evaluation、有限修复和真实失败保留方向成立，但实现存在发送前检查与消费者接线缺口，不能直接判为“没有漏洞”。本次补齐已确认的工程问题。Pro 返回不完整 JSON 的根因仍未证实；不能将这些修复描述成真实正文质量已经改善。

## 已确认问题与处理

| 问题 | 原行为与影响 | 本次处理与证据 |
| --- | --- | --- |
| 正文检查太晚 | schema 合法但正文为空白、带模型引用或超出容量时，先请求摘要，编译时才拒绝 | `validate_body` 在摘要前及编译时共用；至少为摘要、结论各留一个单元。实际 Synthesis 和唯一报告出口回归确认零摘要调用、错误原因可见。 |
| 修复目标身份太晚 | 目标 original/path 在付费响应后比较；Python 负索引可能选中错误单元 | 发送前拒绝负数、bool、越界、重复路径、旧原文和错误章节；零请求、零修复计次。响应后的保护仍保留。 |
| 无进展重复消耗 | runner 将无进展与新增重复候选一并 continue | 无进展立即终止；新增同章重复仍按既有规则最多使用剩余机会。真实生成→Evaluation→修复→出口测试验证只计一次且不交付。 |
| 修复预算 ownership 缺口 | SDK 在 `evidence_rewrite` 约定由调用方预留，但 WriterSession 仅 check，没有 reserve | 调用方在发送期间持有完整请求预留，finally 释放；未知响应已计入的费用占用不退款。实际 SDK MockTransport 检查发送期间预算及取消、未知、成功分支。 |
| 响应完成与用量检查不足 | 只特殊处理 length，其他终止原因及不完整非空 usage 可被误当正常 | 只接受正常 stop 与明确完整的非负整数用量；原始响应先落盘。补 request_id、raw SHA、原始解析位置，拒绝后不重发。 |
| 超量预估被钳制 | `min(8, ...)` 可把非法的超过 128 条断言输入当作八批处理 | 非法数量在预算预检处直接拒绝，不隐藏超容量。实际 Evaluation 容量继续独立检查。 |
| 提前失败原因丢失 | 把检查前移后，未建立摘要调用记录，唯一报告出口可能退化成“报告不可用” | 本地检查失败写入同一 journal 根因，端到端检查 `writer_text_invalid` 保留，Evaluation 为 skipped。 |

实现位于 [writer](../src/litagent/writing_v2.py)、[runner](../src/litagent/runner.py)、[运行身份](../src/litagent/workbench.py)。回归位于 [发送前检查](../tests/test_writer_preflight.py) 和 [真实消费者链](../tests/test_writer_v2.py)。

## 需要明确的计划边界

1. **供应商结构化输出不是成功证明。** 已有原请求确实传递 `json_schema/strict=true`、禁用 thinking 和 20,000 输出上限；原响应 `stop`、输出 3,073 tokens，却缺少完整 JSON 结束结构。这些事实排除了“本机仅因 20k 上限主动截断”的说法，仍不能唯一定位供应商根因。官方也保留结构化输出错误的说明，详见 [方舟结构化输出文档](https://docs.volcengine.com/docs/ark/structured-output-beta?lang=en)。不因此删除本地约束、补括号或将原失败改成成功。
2. **后续预算预估不等于保证任意修复都可完成。** 新正文、独立抽取的断言数和后续请求尚未生成，预估不能证明它们一定可装入剩余额度。每次真实请求仍按完整输入、最大输出和上下文余量检查；后续预算不足的候选不得提交，也不能降低输出预留或裁剪原文。可保证预算与提交边界，不能保证两次修复都能发出。
3. **三题严格通过仍只是固定样本证据。** 缺少人工金标准，不能把模型判断当作科学准确率，也不能从离线测试数量推导真实生成通过率。
4. **旧三题套件保持技术终态。** 本次不复活已终止任务、不追加题目；下一次真实验证必须使用新的冻结身份和明确有界清单。当前阶段保持 `in_progress`。

`grounded-writer-v1` 的输出字段和历史读取不变。仅新 `body_summary_v2` 执行增加 `grounded-writer-execution-v2` 身份，防止修改后的执行规则恢复旧 checkpoint。`legacy`、`single_v1`、`body_summary_v1` 的指纹语义不变。没有新增 Agent、模型、科学宽松门槛或自动重试。

## 原失败重放

使用本机原 SDK 请求和原始响应，通过真实 SDK、预算和 WriterSession 消费链重放；只替代 HTTP transport。原请求逐值一致、原响应 SHA 为 `4a101a71503f146da2bcec6b157414787d4e3096ae5036d481ce9428670c9b47`，再次在 offset 6089 / column 6090 拒绝，第二次提交不重发。外网请求为零，新增付费为零；旧 12,477 tokens、归档和套件状态未变。

私有验收目录为 `artifacts/acceptance/writer-audit-20261023/`，原材料和完整响应不纳入公开文档。最终定向回归 61 项通过，默认离线 `pytest -v` 为 4,181 passed / 25 deselected / 8 既有 warnings；5 个改动 Python 文件 Black/isort 及 `git diff --check` 通过。没有 UI 改动，不把旧视觉验收冒充本轮新增验收。
