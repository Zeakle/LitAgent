# 冻结正文后的单次摘要验证 / One summary on the frozen body

**摘要协议通过，完整组装失败；科学未核验，仍不可交付。 / Summary protocol passed, report assembly failed; science is unverified, not deliverable.**

摘要 JSON 和原生 schema 已通过；组装因摘要 heading 必须为空而拒绝。 / Summary JSON/schema passed; assembly rejected a nonempty heading. 复用已修正文，仅验证摘要。没有重发初稿、检索、提取或修复；完整来源保留。 / Reused the repaired frozen body, with full sources. Only one summary request was allowed.

状态 `failed_no_replay`，错误类型 `ValueError`，源码 `22c578d04386a7e1e95b2dce1ea74f767e723bbc`，合同 `native-body-summary-v7`。

| Check | Status | Location |
|---|---|---|
| frozen_body_integrity | passed | 2f144af560e5d2e29dc3c7e4f50b0d68501b33a0f5ebfc67bb46d5d6493ebd49 |
| full_original_sources | passed | 50 body units; 48 scientific units; 22 cards; 15 complete sources retained |
| new_body_requests | none |  |
| new_repair_requests | none |  |
| summaries_json | passed |  |
| summaries_schema | passed |  |
| local_report_compilation | failed |  |
| reviewer | not_run |  |
| evaluation | not_run |  |
| judge | not_run |  |
| jev | not_run |  |
| scientific_acceptance | not_evaluated |  |
| delivery | blocked |  |
| assembly_constraint | failed | schema_const; sections.abstract.blocks.0.heading |

新增 1 请求 / 11,736 已知 tokens / 0 未知占额；最多1请求/120000tokens/20000输出/零重试。
发送前完整预留67555，阶段余额67576、余21，仅对本份固定输入成立；原500000/64与阶段限制保持。
父族 33 请求 / 244,160 tokens；历史 615 请求 / 3,342,081 已知 / 1,003,467 旧未知占额保留。
继承1次已完成修复，本次修复0；不增加初稿Survey轮数。

未运行Reviewer/Evaluation/Judge/Jev，无人工金标准，不提供准确率或误放行率。协议完成不代表科学支持；Phase14.5 in_progress，正式15题未运行。
完整正文、原始响应与来源只留本地；公开仅脱敏状态、身份和用量。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](README.md) · [HTML](benchmark.html)
