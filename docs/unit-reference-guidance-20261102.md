# Phase 14.5：修复指导的有序目录传输

本步只解决完整修复指导无法放进请求的问题，不重评历史正文，不新增付费调用。045 三题仍为严格交付 **0/3**，Phase 14.5 保持 `in_progress`。

## 已实现的契约

可信服务端冻结路径可显式选择 `repair_guidance_wire_revision=host-unit-ref-guidance-v1`。新选择同时保存独立身份，要求既有 `host-structured-repair-v1` 和 source-map-v3 父契约完整有效。旧 `None` 不增加消息字段，不改变旧身份或 canonical 12,288 字节限制。

完整 Evaluation 解释、原文位置、canonical 指导及失败候选留本地。模型请求中的每个 origin 包含有序原文目录和完整 typed 问题：

- `b` 指向当前请求中已认证的完整正文单元。
- `u` 指向摘要修复请求中已认证的只读正文单元。
- `l` 保留该 origin 的完整原文与引用卡；仅复用可由已认证卡精确生成的程序引用后缀，重渲染必须逐字相同。

`fN`、`iN` 和支持片段的读取方式在实际 system legend 中说明。prior 使用自己的原文与谱系，不能因文本相同借用 current。所有原问题、位置、引用权限、来源顺序和报告身份均须逆向验证；未知字段、越权目标、错来源、顺序篡改或不充分见证被拒绝。

新的 12,288 字节检查针对完整 current＋prior 编码对象。最终请求构造器还检查提示词、完整来源、native schema、20,000 输出预留和上下文余量；不能只凭目录尺寸宣布可发送。450 万 tokens／64 请求、两次共享修复、零重试及科学支持率 1.0 均保持。

这里的字节按 `nested_json_size` 的二层 JSON 嵌入表示计量，包含实际消息中的转义开销，不与原始文件大小或 tokens 混用。

本地见证在保存认证目录、增加修复次数和发送之前检查；发送边界再次复核。坏见证不会伪装成一次已付费尝试。纯历史费用读取只认证 pinned 数据，不执行当前 codec 或恢复旧预算 owner。

## 同源离线证据

最终源码清单为 605 个文件，digest `75e9d1559fa0fe75025b9997c5bfa68cb565d39dd42b75d56c7c80fe6e318f59`；新 wire 身份为 `52e322d92f903aa02e2d730c9c1cf3db66c52d0217cd89a2293a5efa570b2b85`。改动仅 6 个生产文件及 4 个直接写入 `tests/` 的新测试文件。

| 检查 | 当前结果 |
| --- | --- |
| 最终相关回归 | 323 通过，实际 pytest 5935／wrapper 5932 退出 0，源前后相同 |
| 默认离线回归 | 7,351 通过／25 排除／8 warnings，实际 pytest 6087／wrapper 6084 退出 0，435.16 秒，源前后相同 |
| Black／isort／diff | 10 个本步 Python 文件，三个检查实际退出 0 |
| 真实 045 ViT 指导 | 原 54 问题／88 关系／37 目标，49 正文目录项＋4 摘要原文，完整保留 |
| 最终 codec 和 SDK | 11,742 字节；含 native schema／输出预留的检查值 108,107，小于 117,964；实际 SDK HTTP Mock 一次，无真实外部请求 |
| 两修复预算 | 三域 128 容量的三轮完整核验／两修复，以及最大前置消耗后选择 176 容量的两轮完整候选核验均实际模拟通过；更大容量的零前置消耗例单列，不冒充三轮最坏路径 |
| 联合发送矩阵 | 72／72 组 current＋prior 最终 builder／wire 检查通过；最大 113,860／117,964，最大指导 10,568／12,288；复杂 typed 和精确边界另核，不从本表推出任意反馈均可放下 |
| 精确边界及转义 | 66 单元／完整 32k＋4k／全来源／转义输入的 current＋prior 指导填满 12,288 时，实际 SDK HTTP Mock 检查值 116,118；12,289 发送前拒绝。144 份实际 messages 的重序列化差分与组件计量逐一相等 |
| 最大摘要专项 | 三域 66 单元／32k 只读正文＋4k 摘要／全部来源／12,288 指导，真实摘要 prior 谱系及 fresh replay 通过；每域一次 SDK HTTP Mock，检查值 108,386／113,464／112,436，均小于 117,964 |
| 历史保护 | actual 6361 退出 0，35,008 文件、原 Plan SHA、首轮红和源码快照保持 |

真实回放仍保留旧 None 对 68,718 字节 canonical 的拒绝。首次新 codec 的 14,985 字节资源红也原样保留；后续只对重复的程序引用后缀和目录标签进行可逆表示调整。另一次私有回放混用了旧 bindings 与新 document envelope，被严格见证校验拒绝；修正测试输入后，保持原报告 document／bindings 逐值不变，独立传入新请求 envelope，完整 fresh check 和 SDK 构造才通过。

SDK 消费者测试只 Mock 外部传输，实际执行生成、Reviewer、摘要、Evaluation、共享修复和自动判定。无证据科学断言、错误论文归属、核心覆盖回退、失败候选和第三次修改仍不能交付；未知响应保留预算占用，不重发。

## 证据与边界

所有证据位于 `artifacts/acceptance/unit-reference-guidance-20261102/`：

- `source-final-001.json`、`source-final-delta/`：最终源码身份及与只读完整快照的差异。
- `related-final-001/process.json`：最终相关测试实际进程、退出和日志身份。
- `default-final-001/process.json`：最终默认测试实际退出 0；`budget-validation-001/` 单列合成最大消耗、动态容量及完整核验路径，不当作实际费用。
- `replay045-001/`、`replay045-002/`、`replay045-003/`：保留资源红、测试输入红及实际最终成功构造。
- `integration-review-001/review.json`：独立只读接线审查，不冒充模型理解或资源证明。
- `resource-001/`、`boundary-002/`、`boundary-003/`：72 组构造、复杂 typed／摘要／reportless local，以及实际 SDK 和转义边界。首个 boundary 的非法私有推荐配置失败保留，不改生产放行。
- `summary-resource-001/`、`summary-resource-002/`：先 current-only，再补齐真正摘要失败候选的 raw／schema／receipt／谱系和最大只读正文，三域完整构造实际通过。
- `preservation-002-proof.json`：35,008 历史文件及原 Plan 的完整哈希复核。
- `final-closure-001.json`：最终源码、测试、资源、预算、独立审查、历史保护及当前计划文档的收束；不代表新的科学评估。

本步未新建或恢复真实套件，未重发 045，未改变报告、评分、SHA、费用、模型或阈值，也未重启服务或提交／推送。未生成新正文，因此不宣称完成新报告视觉或导出验收。原有 Evaluation 清单／判定一致性和 ViT PDF 第 3 页裁剪问题仍需各自修复；本步的请求可用性不能代替正文质量通过。
