# 三域冻结正文验收

计划3题，运行1题，生成1题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `53a821a4f83ce0ad4e04001ebf49affeb9566f76`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|Yes|not_complete|No|0/0/116|evaluation_assertion_capacity_exceeded|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

新增已知用量89534 tokens，未知保守占用0，4请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)
