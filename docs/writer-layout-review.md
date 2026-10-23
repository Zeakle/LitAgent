# 正文输出层级修复审查（2026-10-23）

## 证据与结论边界

对三份原始非法响应和一份合法响应进行只读 token 级位置检查，结果如下。没有补括号、移动章节或重新解释原输出。

| 观察 | taxonomy 等后续章节的位置 | 原本地校验 |
|---|---|---|
| `frozen-writer-three-domain-20261023-001` | 前章数组中的对象；后续还继续向内嵌套 | 非法 JSON，拒绝 |
| `writer-v2-protocol-20261023-001` | 五章都在根对象下 | JSON/schema/本地正文检查完成，科学未核验 |
| `frozen-writer-three-domain-20261023-002` | 前章数组中的对象；后续还继续向内嵌套 | 非法 JSON，拒绝 |
| `writer-v2-schema-compat-20261023-001` | introduction 数组里的另一个对象 | 非法 JSON，拒绝 |

EOF 是解析器最终无法完成解析的位置，不能据此断言响应只在末尾截断。已捕获的 HTTP 原包与 SDK 内容一致，排除该响应本地 SDK 内容改写；仍不能唯一定位供应商内部原因。

原系统提示说明了句子字段和来源规则，却没有直接展示完整容器层级，依赖供应商原生 schema 处理。[方舟结构化输出文档](https://docs.volcengine.com/docs/ark/structured-output-beta?lang=zh)将该能力标为 beta；本地必须持续严格验证。增加格式提示是针对已观察问题的修复假设，不能预先保证真实输出稳定。

## 实现范围

保持 `body_summary_v2 / grounded-writer-v1`：字段、数组、事实卡、引用权限、完整支持原文及本地 schema 均不改变。`request_messages` 从同一宿主 schema 的 `properties/items/type` 派生路径指南，明确同一对象中的字段是同级字段、子容器须闭合。它不复制来源或 schema description 到系统指令，不给出示例科学事实或偏置引用。

正文、摘要、Reviewer、正文与摘要修复均共用这一个构造器。预算预检、后续修复预估、请求 SHA 与实际 SDK 使用相同的最终 messages；完整原文不截断。journal 标记 `grounded-writer-layout-v1`，执行身份更新为 `grounded-writer-execution-v4`，旧 checkpoint 不跨版本恢复。旧 API 默认、旧策略和历史归档语义保留。

失败费用继承还需接受已知用量完整的协议观察 `blocked` 终态。仍核对 manifest、版本、operation、请求数和归档用量；运行态、未知用量或任意伪造 blocked 不接受。最新失败的 12,537 tokens 不从账本消失，也不计作新的完整 Survey 轮次。

## 验收与真实执行边界

测试直接位于 `tests/test_writer_v2_layout.py`，复用完整消费者 `tests/test_writer_v2.py`，并覆盖实际 SDK 请求及修复预算。格式指导不能让非法 JSON、平衡但错误嵌套或未知引用变成可用输出；科学支持判断仍由独立 Evaluation 完整执行。

固定三域：few-shot、ViT、NeRF 的既有公开快照，分别 20/20/18 卡、9/12/11 篇。离线通过后才冻结同一源码，每题一次，500,000 tokens / 64 请求，总计 1,500,000 / 192；两次共享修复、零传输重试。Lite 负责 Reviewer/Evaluation，Pro 负责正文、摘要和修订，Judge/Jev 关闭。

不重新搜索、提取或加入 Benchmark 标注。技术/身份/未知用量失败停止后续；科学失败与证据不足如实记录后继续。失败及未运行保留在三题分母，不追加到通过、不自动启动十五题。未完整 Evaluation 不可交付；三题未全过时 Phase 14.5 保持 `in_progress`。当前离线和历史重放不能证明真实质量提升。

离线验收：相关 106 项通过，默认 `pytest -v` 4,226 passed / 25 deselected / 8 既有 warnings；7 个改动 Python 文件 Black/isort 与 `git diff --check` 通过。首轮全量唯一错误为沙箱禁止测试夹具绑定临时 loopback 端口，正常权限原样复跑通过。三域完整首请求预留分别 58,604 / 64,861 / 64,931，含完整原文和 20,000 输出；这不保证尚未生成的摘要、修复与 Evaluation 一定能装入余额，仍逐请求检查。

## 固定三域实际结果

源码 `53a821a` 的 [003 结果](../benchmarks/survey/results/frozen-writer-three-domain-20261023-003/README.md)中，few-shot 的正文、摘要和 Reviewer 均完成，首次进入独立断言抽取。132 条模型断言超过 128 容量，后续支持批次和全局核验未运行；116 条暂标科学断言保持未核验。4 请求 / 89,534 已知 tokens、无新增未知和重试，修复请求 0；技术停止后两题未运行，严格交付 0/3。

这说明该次输出跨过了 JSON 层级阻塞，不能从一个样本推导一般协议可靠性或正文质量提升。没有删除超量项、扩大容量或临时改代码继续三域。

后续只读诊断修复保留 `details.skipped_reason` 的真实原因，并将 `attempted=false` 的修复预检占位与真正请求分开。按归档 run_id 和报告 SHA 读取，不回写原状态、评分、正文或费用；旧记录缺失统计显示不可用。渲染器为 view-v3，执行源码仍为 `53a821a`。新增七项回归，默认 4,233 项通过；该修复没有新增模型请求。
