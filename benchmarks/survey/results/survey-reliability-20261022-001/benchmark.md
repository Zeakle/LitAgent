# 单次 Survey 验证：未交付

固定提交 `5d4a7f941ad2e67277db1d1646384c76eb656d7e`，few-shot metric-learning，Lite 提取/Evaluation、Pro 正文；中文 academic_v2、IEEE。只授权并运行 **1 题**，Judge/Jev 关闭，零重试；其他试点及正式套件未执行。

实际 **27 请求 / 111,166 tokens / 410.837 秒**，usage 完整；上限 64 请求/500,000 tokens。Lite 26 请求/88,664 tokens，Pro 1 请求/22,502 tokens。本次原始 JSON 与原生 schema 通过，正文组装因 `flat_body_block_metadata_conflict` 在 `methods[1].heading` 拒绝：前两行共用 block_index=0，却给出不同 heading。原响应保留，不改写后冒充通过。

| 阶段 | 状态 |
|---|---|
| preparation | completed |
| body_response | completed |
| raw_json | passed |
| native_schema | passed |
| body_compile | failed |
| summary | not_run |
| reviewer | not_run |
| evaluation | not_run |
| judge | not_requested |
| jev | not_requested |
| delivery | blocked |

摘要、Reviewer 和 Evaluation 未运行；科学支持率、支持/不支持数量与准确率均不可用。24 张输入事实卡和模型标记的 34 科学行不等于已核验事实。本次属于工程契约失败，不是科学不支持判定。无最终报告 SHA；原响应 SHA `4cb2122c4e6951dcf27524601e821d0228a6095c6d029007670062884af43aba`。科学门禁和预算没有放宽。

本次纳入历史累计：32 次首次生成、642 模型请求、3,453,247 已知 tokens；旧未知保守占用 1,003,467 保留，各版本不能直接混算通过率。正式15题本次0执行。

[工作台运行记录](http://127.0.0.1:8000/flow-demo?run=archive%3Asurvey-reliability-20261022-001-pilot-few-shot-metric&view=quality) · [本地 LangFuse](http://localhost:3000/project/litagent-local-project/traces/64aa1d7f4e740100b87957988fffd54b)

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

自动评测，人工校准未验证；公开包不含原始响应、完整正文、论文支持原文或凭证。输入绑定 `890bf6cce2848a2beb9341ae9b3858618037ca7d7b76ef18de4e942548a49682`，归档 SHA `ac1704caf46c7bbbfbe5ae142e601f526115f4e2f37cc242c77a954c998d0486`。
