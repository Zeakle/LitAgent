# 三域冻结正文验收

计划3题，运行1题，生成0题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `690778d88e1f6b830deb0224b814c36252022f9b`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|No|not_complete|No|—/—/—|usage_unknown|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

冻结模式 / Frozen profile: `writer_reviewer_low_responses_v1`。

```json
{
  "id": "writer_reviewer_low_responses_v1",
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
  "provider_api": "responses"
}
```

新增已知用量0 tokens，未知保守占用56129，1请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

思考明细是 completion 的子集，不再次累计；缺失时不推算。

|Case|Reasoning tokens|Details complete|Available requests / all requests|
|---|---|---|---|
|few-shot-metric|—|No|0/1|
|vit-data-efficient|—|—|—/0|
|nerf-foundations|—|—|—/0|


HTTP 400：服务拒绝 background 字段；未生成正文，Evaluation 未开始。1次请求无usage，保留56,129 tokens未知占用，不能视为实际费用为零。另两题未请求，严格交付0/3。 / HTTP 400: provider rejected background. No body or Evaluation. One request without usage; 56,129 tokens held as unknown, not zero actual cost. Strict delivery 0/3.
