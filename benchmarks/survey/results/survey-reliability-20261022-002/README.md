# 单次真实 Survey 验证：blocked

仅运行 few-shot metric-learning 1 题，源码 `5ad66f6f4e23b7c25cb0ce5d474ffe6fe7d023a3`，合同 `native-body-summary-v10`。Lite 提取/Evaluation，Pro 正文；Judge/Jev关闭、零重试、最多两次共享修复。其他17个引擎槽未授权执行，正式15题本次0运行。

本次 **26 次已发送请求 / 315.523 秒**。已知完成用量 **96,704 tokens**，另有1次响应未知，保守占用 **26,947 tokens**；运行账本合计123,651，不是已知实际账单。套件保守占用500,000，本次额度不退回或重置。上限64请求/500000tokens。usage完整：False。交付：blocked；publishable=False。Evaluation：not_run；科学统计：`{"supported_count": null, "unsupported_count": null, "unverified_count": null}`。

科学支持率：None。None表示不可用，0不等于未执行。无人工校准，不提供科学准确率、误放行率或误拒绝率。

| 阶段 | 结果 |
|---|---|
| preparation | not_complete |
| body_response | not_run |
| body_compile | not_run |
| summary | not_run |
| reviewer | not_run |
| evaluation | not_run |
| judge | not_requested |
| jev | not_requested |
| delivery | blocked |

诊断：`worker_timeout`，阶段 `extract`，原路径 `null`。其他失败码：`["final_report_unavailable", "evaluation_not_passed", "automatic_inventory_missing", "automatic_report_partial"]`。原响应和历史失败保留；不在同一冻结运行中改代码，不追加到通过。

最终报告SHA：`None`；None表示未生成可冻结完整报告。归档SHA：`847a0599a38a37a18f8a95910b9fd47fc95dc0cf025820d91dc4ba486c3c98a3`。输入绑定：`e1e497062b52b41b0a6e1127f40774526bbf55ef8fe4779aea9369918ce77659`。

历史累计：33首次生成/668模型请求/3,549,951已知tokens；累计未知保守占用1,030,414保留。不同源码、检索材料和评估版本不能直接混算通过率；这是一个自动评测试点。

[工作台](http://127.0.0.1:8000/flow-demo?run=archive%3Asurvey-reliability-20261022-002-pilot-few-shot-metric&view=quality) · [本地Trace](http://localhost:3000/project/litagent-local-project/traces/304a3539207eef1ea8b144f7f63938d6) · [旧失败结果](http://127.0.0.1:8007/benchmark.html)

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

公开包仅包含脱敏指标、版本和诊断；完整模型响应、正文、论文原文、运行记录及凭证留在本地。Phase14.5仍in_progress，本文件不代表科学质量保证。
