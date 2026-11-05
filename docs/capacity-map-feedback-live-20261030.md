# 映射反馈修复后的三域真实验收 031（2026-10-30）

本轮确实调用了方舟服务，但在few-shot摘要的本地结构校验失败后停止。ViT、NeRF未发送，Evaluation未开始，严格交付0/3，Phase14.5仍为`in_progress`。此次不能证明科学质量提升，也没有验证251新增映射反馈或清单恢复的真实效果。

## 冻结与费用

套件`frozen-writer-three-domain-20261030-031`，身份`9867decf949314f55a273de5cded3c9ffc811370697acdc25580c0b7090ad0ec`。538源码文件与251默认5529项测试实际退出0的版本相同，90最大请求、六旧计划及两类最大资源路径证明均绑定；31费用父链连续，4992历史路径保持。原三域bundle逐值等于030，不重新检索或提取。

Pro `doubao-seed-2-1-pro-260915`写作／摘要／修订，Lite `doubao-seed-2-1-lite-260915`审阅／Evaluation；端点`https://ark.cn-beijing.volces.com/api/v3`，思考关闭，Judge/Jev关闭，零传输重试、无审批节点、两次共享修复及科学支持率1.0保持。

计划few-shot443万、ViT447万、NeRF447万tokens，合1337万／192请求；每题硬上限450万／64，预留不是费用。实际新增6请求／74887 tokens／0未知占用／0思考tokens，耗时167.942秒。累计882请求／6173874已知tokens，历史1142672未知占用单列保留；历史Jev67请求／756769估算输入、usage不完整保持，新增Judge/Jev均为0。

## 实际运行到哪里

| 环节 | 结果 |
|---|---|
| 五章正文 | 五次响应收到，摘要请求包含48个宿主正文单元；未完成整篇编译与科学核验 |
| 摘要及结论 | 第六次响应为完整JSON，abstract=2、conclusion=3；结论超过本地2单元上限 |
| Reviewer／Evaluation／共享修复 | 均未运行；不是事实支持率为0 |
| ViT、NeRF | 按技术停止规则未运行，保留三题分母 |
| 最终交付 | 不可交付，无完整最终报告SHA |

最终原因`writer_flat_schema_invalid`，精确诊断`path=["conclusion"], reason=maxItems`。原始响应`finish_reason=stop`、`provider_status=completed`、无provider error，不是非法JSON、输出截断、认证失败或预算耗尽。原响应SHA为`fb4f666cafdd9de91699d4b2c1af8b64d535708f9f709812861a4c35274ac804`。

## 已证实的工程契约冲突

读取的是此次实际发送的消息与schema，不是只看当前配置：

| 位置 | 实际约束 |
|---|---|
| `payload.summary_limits_per_section.target_sentence_units` | `[3,5]` |
| 同字段`max_sentence_units` | `6` |
| 共享CONTENT CONTRACT文字 | normally two or three findings |
| `writing_envelope.summary_max_units_per_section` | `2` |
| `writing_plan.summary_section_max_units` | `2` |
| 本地flat schema | abstract、conclusion各`maxItems=2` |
| 实际provider schema描述 | Host validation requires maxItems=2、minItems=1；服务端结构投影未原生强制数组数量 |

Pro选择3个结论单元，遵循了请求中的一部分描述，却违反另一部分硬边界。独立使用实际构造器与flat parser只读重放，仍准确拒绝相同maxItems错误。不能把整个失败归因于模型能力，不能删第三项、拼接句子、补JSON、扩大上限或恢复031来接受结果。

5529项离线测试和最大请求证明没有发现此次请求内部的摘要规模矛盾。它们证明已覆盖的契约和资源反例，不证明所有提示词一致或真实科学通过。

## 对账与展示

`verify_terminal.py`核对538源码副本、4992保护路径、六次发送／结算／原始事件一致，已知usage74887、未知0，旧030及全部历史账本保持。LangFuse读回六个Responses generation，usage相同，科学输入输出均为空；仅核验generation层，未宣称完整trace树验收。

JSON与HTML内嵌数据逐值相同，CSV计数和终态字段、Markdown条目与原因一致。桌面1280、390px中英文与键盘语言切换已检查，未核验、未运行及不可交付明确呈现。无完整正文，不把原文定位、附录或PDF／Word导出算作已验收。旧8000只读服务起初没有031，现仅复制逐字相同的原归档到既有回放缓存，不重启服务、不初始化模型或任务恢复。

[031 Benchmark](http://127.0.0.1:8020/frozen-writer-three-domain-20261030-031/benchmark.html) · [031运行诊断](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261030-031-few-shot-metric&view=quality)

本步没有修改业务源码，沿用538同源的5529默认验收；`git diff --check`退出0。没有再次运行全量pytest，也没有以旧通过冒充新增科学验证。未提交或推送代码。

## 下一步253：先离线统一摘要规模，不再追加付费试跑

1. 从同一有效envelope派生摘要目标、数量上限、writing plan、schema及正文编译限制，明确旧策略兼容。移除新路径中相互矛盾的3–5／6及two-or-three建议，不增加输出格式、预算、修复次数或Agent。
2. 同时审查首次摘要、正文修订后的摘要更新、摘要局部修复和全部提示词消费点。每单元含多个断言仍完整独立核验，不能为2单元上限强行堆复合事实或删除核心发现。
3. 在tests直接新增此次真实请求形状和raw反例；联合核对实际system／payload／provider schema描述／host schema及真实消费者，而不是仅测schema。合法2单元正例、3单元原响应仍拒绝、配置变化及历史fingerprint反例都需保留。
4. 改动后重新封存源码，执行相关及默认pytest实际退出0、改动Python格式、全部最大请求及旧预算兼容，再决定新的独立有界验证。不恢复031、不追加到通过、不自动15题／默认推广。

本机证据位于`artifacts/acceptance/frozen-writer-three-domain-20261030-031/`：manifest、offline-gates、preflight-review、independent-preflight-review、preservation-before-live、state、acceptance-proof、summary-contract-review、trace-proof、public-export-proof及readonly-cache-proof。原始响应和完整科学材料仅在本地归档与journal中保留。
