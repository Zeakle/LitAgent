# 单次真实 Survey 验证：blocked

仅运行 few-shot metric-learning 1 题，源码 `59fca65209caaf3232e70e007ee43bf29e912365`，合同 `native-body-summary-v10`。Lite 提取/Evaluation，Pro 正文；Judge/Jev关闭、零重试、最多两次共享修复。其他17个引擎槽未授权执行，正式15题本次0运行。

本次 **27 次已发送请求 / 441.923 秒**。已知完成用量 **119,193 tokens**，另有0次响应未知，保守占用 **0 tokens**；运行账本合计119,193，不是已知实际账单。套件计入用量119,193，本次额度不退回或重置。上限64请求/500000tokens。usage完整：True。交付：blocked；publishable=False。Evaluation：not_run；科学统计：`{"supported_count": null, "unsupported_count": null, "unverified_count": null}`。

本次50篇提取全部完成，Pro正文用时85.415秒，无超时。原JSON合法；原schema检出2处章节来源越权：methods[0].fact_card_refs[1]=m12、methods[2].fact_card_refs[1]=m8。阅读卡c12/c8虽然存在，但宿主没有生成m12/m8授权映射；模型不能根据卡序号自行拼出methods引用。未扩权、未删引用或改正文冒充通过；本次修复请求0，摘要、Reviewer及Evaluation未运行。

科学支持率：None。None表示不可用，0不等于未执行。无人工校准，不提供科学准确率、误放行率或误拒绝率。

| 阶段 | 结果 |
|---|---|
| preparation | completed |
| body_response | completed |
| body_compile | failed |
| summary | not_run |
| reviewer | not_run |
| evaluation | not_run |
| judge | not_requested |
| jev | not_requested |
| delivery | blocked |

诊断：`stage_contract_invalid`，阶段 `body`，原路径 `["methods", 0, "fact_card_refs", 1]`。其他失败码：`["final_report_unavailable", "evaluation_not_passed", "automatic_inventory_missing", "automatic_report_partial"]`。原响应和历史失败保留；不在同一冻结运行中改代码，不追加到通过。

最终报告SHA：`None`；None表示未生成可冻结完整报告。归档SHA：`3d535cfcb14814e885154c0f19d22ab41aa1e362f13ebaccad9a951cba95a0af`。输入绑定：`d712643aae37206d057e3c6569fd56ab1782c2a27ec3efca04d10096b17a091c`。

历史累计：34首次生成/695模型请求/3,669,144已知tokens；累计未知保守占用1,030,414保留。不同源码、检索材料和评估版本不能直接混算通过率；这是一个自动评测试点。

[工作台](http://127.0.0.1:8000/flow-demo?run=archive%3Asurvey-reliability-20261022-003-pilot-few-shot-metric&view=quality) · [本地Trace](http://localhost:3000/project/litagent-local-project/traces/ae00b775623500c0a6c1edf0d57f3a43) · [旧失败结果](http://127.0.0.1:8008/benchmark.html)

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

公开包仅包含脱敏指标、版本和诊断；完整模型响应、正文、论文原文、运行记录及凭证留在本地。Phase14.5仍in_progress，本文件不代表科学质量保证。
