# 原生写作的思考模式配置接线

本批只修复已有 `thinking_mode` 参数被原生写作路径忽略的问题，不改变默认 `disabled`，不执行新的付费请求。执行身份更新为 `grounded-writer-execution-v9`，生成 schema、提示词、来源、模型分工和科学门禁不变。

## 问题与边界

旧 `WriterSession` 将请求固定为 `thinking.type=disabled`，绕过了 Synthesis 和 Reviewer 已有配置。冻结执行器构造写作 Worker 时也漏传该配置。使用 007 冻结的旧源码运行新增回归，确认配置为 `enabled` 时实际请求仍为 `disabled`，然后在当前源码验证修复。

**007 原本就配置为 disabled，因此该缺陷不是已确认的非法 JSON 原因。** [真实结果](../benchmarks/survey/results/frozen-writer-three-domain-20261024-007/README.md)仍是严格交付 0/3：前四章本地契约通过，第五章非法 JSON，Evaluation 未运行。不能把参数接线或离线 Mock 当作真实协议稳定性、科学支持率或通过率提升。

## producer / consumer 契约

- 初稿、摘要及共享修复继承冻结的生成模式；模式进入 journal 身份和宿主 `writer_document.generation_thinking_mode`。
- Reviewer 使用自身配置，重新打开 journal 时仍绑定原生成模式。旧文档缺少模式字段按旧实际行为 `disabled` 读取；这不允许旧 checkpoint 跨新执行版本恢复。
- `enabled`、`disabled` 发送对应 `thinking.type`；`provider_default` 完全省略参数。非法值在发送前拒绝。
- 记录请求模式及收到的 reasoning 字符数，不保存或引用 reasoning 正文。字符数不是 tokens、科学证据，也不证明服务实际执行了请求模式。
- 真实 SDK 加 HTTP 替身验证：当 completion usage 已包含 reasoning 时只计一次；仅有 reasoning、没有合法 JSON 正文仍失败且计费，不自动重发。这是离线 fixture 结论，不代替真实供应商计费核对。
- 仍为每题 500,000 tokens／64 请求、原阶段预留、10% 上下文余量、两次共享修复、零重试；不裁剪材料、不降低输出预留、科学支持率仍要求 1.0。

现有受控套件的 disabled 配置门禁保持，不能直接删掉门禁开始思考模式试跑。后续若验证其他模式，应先固定该配置、输出及 reasoning 计费语义、请求身份和有界预算；不能重复当前默认请求来宣称已验证新模式。

随后官方契约审查发现新的前置问题：当前 Doubao 的 `max_tokens` 只限制回答，不包含思考；`max_completion_tokens` 才限制两者合计，且两字段不能同时发送。因此本批 enabled 模式的 SDK 接线测试并不证明真实请求具有完整输出上界。默认 disabled 和原套件门禁继续保持，先完成该预算消费修复，再进行任何真实思考模式验收。[官方长度控制说明](https://docs.volcengine.com/docs/ark/context-management?lang=zh)

## 离线验收

[新增模式测试](../tests/test_writer_thinking.py)覆盖三种模式、实际 Synthesis→Reviewer、摘要及修复继承、历史缺字段、模式身份冲突、SDK 序列化、独立模型路由、usage 及 reasoning 隔离。真实冻结消费者测试同时覆盖 enabled/disabled，执行 Synthesis→Reviewer→唯一报告出口→Evaluation，外部传输才使用替身。

最终默认 `pytest -v`：**4337 passed、25 deselected、8 warnings**。5 个改动 Python 文件 Black/isort 与 `git diff --check` 通过。007 脱敏结构反例仍非法，不补括号接受。24 个历史保护路径、472 份冻结源码副本、007 原归档及 journal 均保持；公开 JSON/CSV/HTML 三题分母和费用一致。

本离线批新增付费调用为零。历史累计 749 次模型请求／4,176,901 已知 tokens，以及旧 1,030,414 未知保守占用分别保留。私有验收记录在 `artifacts/acceptance/writer-thinking-20261024/`。没有前端或新报告变更，不宣称新增 PDF/Word 视觉验收。

Phase 14.5 保持 `in_progress`。三域真实严格通过的目标尚未完成。

## 完整输出预算修复（execution-v10，仅离线）

在上述配置接线之后，继续修复总输出参数：原生写作在 `disabled` 时保持原 `max_tokens`；在 `enabled` 或可能开启思考的 `provider_default` 时，用同额 `max_completion_tokens` 约束思考与回答合计。两字段不会同传，SDK 也不会额外补上默认 `max_tokens`。20k 仍是完整输出预留，思考会占用回答空间；不保证额外获得 20k 回答，触顶仍拒绝交付。

共享客户端按实际总上限预留；参数冲突或非法值在请求计数和传输前拒绝。Journal 输入身份包含实际字段，执行版本更新为 v10；旧记录和 checkpoint 不被改写。事件保留历史数值字段 `max_tokens` 作为宿主预留口径，同时用 `output_limit_parameter` 明示实际 API 参数，脱敏记录和 LangFuse 保留该安全元数据。

新增 [总输出回归](../tests/test_completion_output_limit.py) 经真实 SDK、HTTP 替身及 shared CostBudget 检查字段互斥、小于默认值的显式上限、额度不足、超时及缺 usage。响应未知时保留完整预留占用，不自动重发；completion 内的 reasoning 仍不重复计费。原生写作还覆盖 `length`、仅 reasoning、非法 JSON 均不能放行。替身不能证明真实供应商一定遵守参数，也不能证明科学支持。

本批相关 **172 项通过**；最终默认 **4352 passed、25 deselected、8 warnings**，6 个改动 Python 文件格式及差异检查通过。24 个保护路径、472 份旧源码副本、007 原归档及 journal 不变。本批新增付费零，默认仍 disabled，原套件模式门禁未放宽，真实严格结果仍 0/3。后续使用独立冻结配置验收，不续跑旧终态。

## 独立冻结配置入口（仅离线与准备）

三域 CLI 现在提供固定的 `writer_reviewer_thinking_v1` 配置，仅在 `--prepare` 时显式选择。它同时开启 Pro 初稿／摘要／修复和 Lite Reviewer 的思考；Lite Evaluation、提取及规划仍为 disabled。生成、Reviewer 采用原 20,000／4,096 的总输出上限，Evaluation 保持 16,384。默认 `disabled_v1`、缺少该字段的旧 manifest 及旧模型检查门禁保留原语义；没有修改默认 YAML 或模型。

有效配置使用深拷贝，模式与完整配置进入 manifest 身份；`--live` 从已冻结配置恢复，不允许重新传入模式。原 3 题／150 万 tokens／192 请求上限及所有科学门禁保持，不自动进入正式 15 题。历史 receipt 必须继续单列继承，准备新配置不能重置已花费用。

JSON、CSV、Markdown 和 HTML 展示同源模式与输出额度；页面首部明确写作、Reviewer、Evaluation 的模式。缺失的历史额度显示不可用，不推算默认值。页面检查使用标有“离线配置预览（无模型请求）”的临时数据，桌面／390px、中英文、键盘切换与展开均通过，页面宽度 390、console 错误和警告为零；它不是一份新 Survey 或真实质量结果。

[配置入口回归](../tests/test_writer_thinking_profile.py)覆盖显式模式、旧门禁、默认配置不变、输出／模型／预算漂移、manifest 身份、CLI 不得覆盖及四格式同源。相关 132 项通过，页面相关最终 38 项通过；最终默认 **4369 passed、25 deselected、8 warnings**，2 个改动 Python 文件格式及差异检查通过。首轮测试发现 consumer 误读配置层级，已按真实 `build_config_summary.effective` 修复，没有伪造扁平 fixture。

此入口只建立可验证的实验配置，不证明开启思考能解决 007 的非法 JSON。下一次真实验收仍须单独冻结源码、三域完整材料、配置和累计账本，各题一次、技术或未知用量即停，失败保留，不追加至通过。

## 随后真实验收：008 停于总输出上限

[008](../benchmarks/survey/results/frozen-writer-three-domain-20261024-008/README.md)使用提交 `edfbc13`、476 份源码和原三域材料。引言通过本地契约，耗时 253.5 秒、9,206 输入＋11,652 输出 tokens；分类章耗时 260.6 秒、10,518 输入＋20,000 输出，`finish_reason=length`，正文字符数为零。它未触发 300 秒超时，不能把本次失败归因于超时。

2 次 Pro 请求共 51,376 已知 tokens，完整 usage、零重试、新增未知为零。没有摘要、Reviewer、Evaluation、修复或最终报告，另两题未运行，严格交付仍为 0/3。保持总上限后拒绝了不完整输出，没有补 JSON、扩大额度或追加试跑。

返回的 reasoning 字符数并非计费 tokens；当前 normalizer 没有持久化可选 reasoning-token 明细，不能事后推算精确的思考占比。此结果证明总输出门禁生效，不证明启用思考提高了正文质量，也不证明此前非法 JSON 的原因已经解决。

28 个保护路径、476 份执行源码、原结果保持。LangFuse 读回 2 个 generation／51,376 tokens、`max_completion_tokens` 元数据及科学 I/O 为空，不声称完整 trace 树。四格式同源与三题分母、桌面／390px、中英文、键盘及 console 零错误／警告已核对。历史累计 751 次模型请求／4,228,277 已知 tokens，旧 1,030,414 未知保守占用继续单列。Phase 14.5 保持 `in_progress`。

## 固定 low 强度与明细消费（2026-10-25，仅离线）

新增固定配置 `writer_reviewer_low_thinking_v1`，显式发送 `reasoning_effort=low`。它由服务端冻结参数进入真实 Synthesis、摘要、Reviewer 和共享修复，不增加普通 AppConfig 字段，不改变旧默认配置指纹。旧两种配置保留原含义；新参数进入 manifest、journal、输入身份和宿主文档，执行身份升级为 v11。Reviewer 可以独立省略 effort，不会误继承生成侧设置。

总输出仍为生成 20,000／Reviewer 4,096；Evaluation 16,384、每题 500k／64、原超时、两次共享修复及科学支持率 1.0 保持。`low` 只是间接调节思考长度，不保证回答空间、耗时或科学质量，不能据此扩大额度或接受不完整正文。[官方思考控制说明](https://docs.volcengine.com/docs/ark/deep-thinking?lang=en)

SDK 现在转发显式 effort，并保留可选 `completion_tokens_details.reasoning_tokens`。明细只作为 completion 的子集展示，不再次计费；缺失保持不可用，畸形解析值标为明细无效，不改已知主计费。缺主 usage 仍按原门禁停止。SDK 会先转换部分原始类型，例如 HTTP 布尔值可能被转换为整数；本层检查 SDK 暴露值，不宣称验证了原始 HTTP 类型。它不依据思考摘要字符数补算 tokens，也不将明细用于科学门禁。

真实 SDK／HTTP 替身、独立 Reviewer、两次共享修复及实际 Synthesis→Reviewer→唯一报告出口→Evaluation 的正反例均通过。原 008 空正文及 `length` 响应仍被拒绝，007 坏 JSON 仍非法。实际 driver／recorder／四格式导出验证了完整与部分明细：部分缺失时不提供推算总数，主计费保持。LangFuse 只增加安全标量元数据，不把 reasoning 另列为计费 usage，不传思考正文。

相关回归 **217 项通过**；最终默认 **4412 passed、25 deselected、8 warnings**，12 个改动 Python 文件 Black/isort 及差异检查通过。32 个保护路径、476 份旧源码、008 原运行归档及 journal 保持，累计费用不变，探针继承链补至 008。本离线批新增付费请求为零。

大报告预检在真实 Synthesis 重放 005 六份旧响应，完整 user payload／schema 与旧请求相同，草稿 SHA 仍为 `34e8fca38d54b469c098281a4e320798ac9ae6ccf75801991659adaab6002e4f`；Reviewer 使用离线替身。五章预留 56,129／58,775／64,036／71,282／77,208，摘要 58,648、Reviewer 107,328；ViT／NeRF 首章 62,386／62,512。以上只证明完整请求构造与参数接线，不是模型对 low 的真实生成、语义判断或未来全程预算保证。

带“离线配置预览（无模型请求）”标识的页面完成桌面／390px、中英文、键盘及 console 零错误／警告验收，明确展示 `effort=low`；没有新最终正文，不宣称 PDF／Word 正文验收。下一步先提交后独立冻结三域配置，不续跑 008。真实严格结果仍为 0/3，Phase 14.5 继续 `in_progress`。

## 随后真实验收：009 仍有非法 JSON

[009](../benchmarks/survey/results/frozen-writer-three-domain-20261025-009/README.md)固定提交 `783a8ae`、478 份源码及原三域材料。首个引言请求约 29.6 秒，9,208 输入＋896 输出＝10,104 已知 tokens；其中 reasoning 343 已包含在输出中。`finish_reason=stop`，没有触及 20k 总输出或 300 秒超时，却在原 1,082 字符的 EOF 解析失败：章节数组和根对象仍未闭合。原字节及 SHA 保持，没有补结构接受。

仅 1 个 Pro 请求，usage 完整、零重试／新增未知；摘要、Reviewer、Evaluation 和修复未运行，另两题未请求，最终报告 SHA 不可用，严格交付 0/3。LangFuse 已读回 low／343 reasoning 元数据、1 个 generation／10,104 tokens 及科学 I/O 为空，不声称完整 trace 树。明细及参数接线真实生效不等于协议或科学质量已改善。

32 个保护路径、478 份执行源码、旧费用及评分保持，累计 752 请求／4,238,381 已知 tokens，旧 1,030,414 未知保守占用单列。四格式同源、三题分母及桌面／390px、中英文、键盘、console 零错误／警告已核对。009 未捕获 HTTP 原包，不能拿旧独立 HTTP 观测或本地 SDK 替身排除本次所有边界问题。

后续停止付费调参，先离线核对同一 `grounded-writer-v1` schema 的供应商接口映射与响应解析。Responses API 同样支持结构化输出但仍为 beta；审查该映射是兼容性验证方向，不是它一定能解决失败的证据。[官方接口说明](https://docs.volcengine.com/docs/ark/responses-api-structured-output?lang=zh) 不修改科学门槛，不减少核心问题或替换冻结材料，Phase 14.5 仍 `in_progress`。
