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
