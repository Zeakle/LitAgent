# 三域冻结正文验收

计划3题，运行1题，生成1题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `52121927b0b14f19edf8aee47f3cfeb8ebe6318b`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|Yes|not_complete|No|—/—/—|responses_incomplete / HTTP 200 / unavailable|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

冻结模式 / Frozen profile: `writer_reviewer_low_responses_reemit_v2`。

```json
{
  "id": "writer_reviewer_low_responses_reemit_v2",
  "generation_thinking_mode": "enabled",
  "reviewer_thinking_mode": "enabled",
  "evaluation_thinking_mode": "disabled",
  "output_limit_parameter": "max_output_tokens",
  "generation_output_tokens": 20000,
  "reviewer_output_tokens": 4096,
  "evaluation_output_tokens": 16384,
  "configured_models": {
    "extraction": "doubao-seed-2-1-lite-260915",
    "generation": "doubao-seed-2-1-pro-260915",
    "evaluation": "doubao-seed-2-1-lite-260915"
  },
  "reasoning_effort": "low",
  "provider_api": "responses",
  "mapping_contract": "ark-responses-mapping-v2",
  "storage": false,
  "stream": false,
  "host_input": "complete",
  "server_truncation_control": "unavailable",
  "reemission_contract": "intact-unit-reemission-v2"
}
```

Writer prompt: grounded-writer-native-intent-v3

新增已知用量149128 tokens，未知保守占用0，9请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

共享修复次数 / Shared repair attempts (limit 2; missing history unavailable)

|Case|Attempts|
|---|---|
|few-shot-metric|2|
|vit-data-efficient|—|
|nerf-foundations|—|

思考明细是 completion 的子集，不再次累计；缺失时不推算。

|Case|Reasoning tokens|Details complete|Available requests / all requests|
|---|---|---|---|
|few-shot-metric|6484|Yes|9/9|
|vit-data-efficient|—|—|—/0|
|nerf-foundations|—|—|—/0|
