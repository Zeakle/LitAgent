# 单次真实 Survey 验证：blocked

仅运行 few-shot metric-learning 1 题，源码 `eb675a424af69a5eafa68b80c136d9ed66eb3b82`，合同 `native-body-summary-v11`。Lite 提取/Evaluation，Pro 正文；Judge/Jev关闭、零重试、最多两次共享修复。其他17个引擎槽未授权执行，正式15题本次0运行。

本次 **28 次已发送请求 / 359.298 秒**。已知完成用量 **132,596 tokens**，另有0次响应未知，保守占用 **0 tokens**；运行账本合计132,596，不是已知实际账单。套件计入用量132,596，本次额度不退回或重置。上限64请求/500000tokens。usage完整：True。交付：blocked；publishable=False。Evaluation：not_run；科学统计：`{"supported_count": null, "unsupported_count": null, "unverified_count": null}`。

提取完成论文数：50。原schema共有5处章节引用越权，涉及4个正文单元。模型明确拒绝修复：True；引用修复完成：False；共享修复尝试：`[{"index": 0, "stage": "native_initial", "status": "failed", "error_code": "ValueError"}]`。原始引用错误路径：`[["methods", 4, "fact_card_refs", 0], ["experiments", 1, "fact_card_refs", 1], ["experiments", 4, "fact_card_refs", 0], ["open_problems", 5, "fact_card_refs", 0], ["open_problems", 5, "fact_card_refs", 1]]`。逐次 JSON/schema 状态见 JSON 与页面诊断，局部合法不代表科学通过。

科学支持率：None。None表示不可用，0不等于未执行。无人工校准，不提供科学准确率、误放行率或误拒绝率。

| 阶段 | 结果 |
|---|---|
| preparation | completed |
| body_response | completed |
| reference_repair | failed |
| body_compile | failed |
| summary | not_run |
| reviewer | not_run |
| evaluation | not_run |
| judge | not_requested |
| jev | not_requested |
| delivery | blocked |

诊断：`body_reference_repair_unavailable`，阶段 `body_repair_refs_0`，原路径 `null`。其他失败码：`["final_report_unavailable", "evaluation_not_passed", "automatic_inventory_missing", "automatic_report_partial"]`。原响应和历史失败保留；不在同一冻结运行中改代码，不追加到通过。

最终报告SHA：`None`；None表示未生成可冻结完整报告。归档SHA：`fd8557b63ecc3dc5dfc2570f0409dcb0cba04f1e18874a145ffc27ed2495e6d3`。输入绑定：`0e34842044ce1d61264c30f20f907bbe69a05f7c0cd5d0d231f9c5818836ffb8`。

历史累计：35首次生成/723模型请求/3,801,740已知tokens；累计未知保守占用1,030,414保留。不同源码、检索材料和评估版本不能直接混算通过率；这是一个自动评测试点。

[工作台](http://127.0.0.1:8000/flow-demo?run=archive%3Asurvey-reliability-20261022-004-pilot-few-shot-metric&view=quality) · [本地Trace](http://localhost:3000/project/litagent-local-project/traces/0801e15fb9ea7c8334b3c10416927793) · [旧失败结果](http://127.0.0.1:8009/benchmark.html)

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

公开包仅包含脱敏指标、版本和诊断；完整模型响应、正文、论文原文、运行记录及凭证留在本地。Phase14.5仍in_progress，本文件不代表科学质量保证。
