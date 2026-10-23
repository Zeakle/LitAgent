# 三域冻结正文验收

计划3题，运行1题，生成0题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `d19420c0a3802a0f0cf1da553d15b5fccdcbc305`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|No|not_complete|No|—/—/—|writer_json_invalid|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

新增已知用量12422 tokens，未知保守占用0，1请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

本次非法 JSON 定位于第5,974列（5973字符末尾），`finish_reason=stop`；尚未执行科学 Evaluation，因此没有科学支持分数。模型、消息、schema、temperature、输出上限和输入 SHA 与[成功协议观察](../writer-v2-protocol-20261023-001/README.md)相同，这两次结果不证明服务已稳定。原归档、原响应和失败记录保留，Phase14.5仍为in_progress。
