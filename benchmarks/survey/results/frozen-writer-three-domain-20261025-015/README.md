# 三域真实验收015：正文完成，Reviewer思考耗尽输出上限

**严格交付0/3，质量目标未完成。** few-shot五个核心章节及摘要/结论完成本地编译，共93个正文句子单元、8个派生摘要/结论单元；这不是独立核验后的科学断言数量。随后Lite Reviewer返回HTTP200、`incomplete / length`，4096输出tokens全部为思考，没有审阅答案。系统拒绝不完整响应，Evaluation尚未运行；ViT/NeRF按技术停止规则未请求，保留在三题分母。

- 冻结源码`52121927b0b14f19edf8aee47f3cfeb8ebe6318b`，486源码，identity`fd5e7721bf49bfe8815102f5fd13cf1f34277d56127d67c36beace81ca9247b8`；提示`grounded-writer-native-intent-v3`，业务协议和恢复v2不变。
- 原三域20/20/18卡、9/12/11篇与完整问题/原文/配置保持，16完整父账本、56历史保护，不重搜提取。Lite审阅评估/Pro写作，Judge/Jev关闭，无人工审批。
- 每题500000 tokens/64请求，总1500000/192，2共享修改、零重试、原科学1.0/覆盖/全局门禁不变。
- 代表性方法与综合比较各发生一次JSON末尾错误，2个独立恢复响应通过原schema与全部内容/引用/段落顺序保真检查。原失败响应保留；共享额度2/2已经使用，不另开修改池。
- 9次真实请求、149128完整已知tokens、0新增未知占用、约363.3秒。Pro 8次/112179tokens，Lite Reviewer 1次/36949tokens。6484思考tokens已包含在输出内，不重复累计。
- Reviewer输入32853tokens，输出上限4096（含思考）；实际输出4096、reasoning4096，只有incomplete reasoning item，无message答案。单次输入/输出预留104365合法，本次不是上下文或500k总预算不足，也不是JSON解析器丢失答案。
- 中间正文SHA`651a8942fdd9a6632a65d7b5ef64b40e6d25536490a0763d834f94e28b0dcea0`。正文与引用位置已编译，但没有完整Reviewer或Evaluation结果；最终报告SHA为null，未交付，科学分数不可用，不能把格式通过记为科学通过。
- 全部9份真实请求/响应经安装SDK→Synthesis→Reviewer逐字离线重放，恢复相同中间正文SHA，仍拒绝同一不完整Reviewer，0新增外部请求。本地草稿另存并明确标为未核验，不覆盖唯一最终报告出口，也不放入公开包。
- 累计771请求、4483702已知tokens、1142672旧未知占用，Jev67、Judge0保持。旧失败及013欠费占用不退款、不清零。
- LangFuse读回9个generation、149128tokens、6484思考明细、空科学IO；不声称完整业务链或科学正确性已验证。

[HTML](benchmark.html) · [JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](benchmark.md) · [脱敏诊断与分阶段用量](protocol-diagnostic.json) · 同schema提示离线证据

本轮没有在中途修改源码、追加重试或恢复已终止任务。正文编译完成不证明提示造成统计意义上的质量提升。下一步先离线审查Pro写作与Lite Reviewer思考配置的耦合，在原4096输出与总预算内调整角色分工；不自动提高预算或付费重跑015。Phase14.5保持`in_progress`。
