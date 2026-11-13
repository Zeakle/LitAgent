# 正文类型约束与有限修复 / Typed body and bounded repair

**科学未核验 / Scientific content not evaluated · 不可交付 / Not deliverable**

仅复用冻结 few-shot 材料；完整来源和章节权限保留。未运行 Reviewer、Evaluation、Judge 或 Jev。协议完成不代表科学支持，不计为新增 Survey 轮。 / Frozen inputs only. Scientific content remains unverified. 本次7处正文诊断由同一冻结源码只读重放原始响应得到，原终态仍是修复预算拒绝；不改旧记录。 / Seven body diagnostics are a read-only replay of the preserved response; the terminal error remains the repair budget gate. 停止原因：7处复合句引用范围不明确，修复需预留106,977 tokens，当前阶段仅余100,849；未发送修复请求。 / Stopped before repair: full reservation exceeds remaining writing-stage budget.

状态 `failed_no_replay`；源码 `5bc7e7ff9c4d8702da88cb1c5d05c6f3d131fa78`，合同 `native-body-summary-v4`。

| Check | Status | Location |
|---|---|---|
| body_json | passed |  |
| body_schema | passed |  |
| summaries_json | not_run |  |
| summaries_schema | not_run |  |
| full_local_compile | not_run |  |
| reviewer | not_run |  |
| evaluation | not_run |  |
| scientific_acceptance | not_evaluated |  |
| delivery | blocked |  |
| body_host_validation | failed |  |
| body_repair_budget | blocked | evaluation_run_budget_exceeded; reservation:106977, stage_remaining:100849 |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.experiments.first_block.first_sentence |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.experiments.additional_blocks.0.additional_sentences.1 |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.open_problems.first_block.first_sentence |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.open_problems.additional_blocks.0.additional_sentences.0 |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.open_problems.additional_blocks.1.first_sentence |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.open_problems.additional_blocks.1.additional_sentences.0 |
| host_diagnostic | failed | strict_writing_unit_citation_scope_ambiguous; sections.open_problems.additional_blocks.2.additional_sentences.0 |

新增 1 请求／22,428 已知 tokens／0 未知占额；最多4请求／120000 tokens，每次20000输出，零重试。
父族累计 30 请求／199,151 tokens；全历史 612 请求／3,297,072 已知 tokens，旧未知占额 1,003,467 保留。

失败不重放；预算不足不裁证据、不借Evaluation额度。无人工金标准，准确率、误放行率和误拒绝率不可用。Phase14.5保持in_progress，正式Benchmark仍0/15。

完整正文、请求、来源和响应只在本地；公开文件仅含脱敏状态、诊断位置、用量与身份。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](README.md) · [HTML](benchmark.html)
