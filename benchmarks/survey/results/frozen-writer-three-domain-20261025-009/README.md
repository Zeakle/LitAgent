# 三域冻结正文验收

计划3题，运行1题，生成0题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `783a8aede5296ed51ce47015a90ef1a6fc08e0a5`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|No|not_complete|No|—/—/—|writer_json_invalid|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

冻结模式 / Frozen profile: `writer_reviewer_low_thinking_v1`。

```json
{
  "id": "writer_reviewer_low_thinking_v1",
  "generation_thinking_mode": "enabled",
  "reviewer_thinking_mode": "enabled",
  "evaluation_thinking_mode": "disabled",
  "output_limit_parameter": "max_completion_tokens",
  "generation_output_tokens": 20000,
  "reviewer_output_tokens": 4096,
  "evaluation_output_tokens": 16384,
  "configured_models": {
    "extraction": "doubao-seed-2-1-lite-260915",
    "generation": "doubao-seed-2-1-pro-260915",
    "evaluation": "doubao-seed-2-1-lite-260915"
  },
  "reasoning_effort": "low"
}
```

新增已知用量10104 tokens，未知保守占用0，1请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

思考明细是 completion 的子集，不再次累计；缺失时不推算。

|Case|Reasoning tokens|Details complete|Available requests / all requests|
|---|---|---|---|
|few-shot-metric|343|Yes|1/1|
|vit-data-efficient|—|—|—/0|
|nerf-foundations|—|—|—/0|

## 本次真实终态 / Real terminal outcome

固定源码 `783a8ae`、478份执行源码，原三域完整材料，`writer_reviewer_low_thinking_v1`。实际仅few-shot首个引言请求发出：9,208输入＋896输出＝10,104已知tokens；其中reasoning 343已经包含在输出中，不再累加。请求约29.6秒，finish_reason=stop，未触及20,000总输出上限或300秒超时。

原1082字符响应在EOF解析失败，结构扫描仍有root对象和introduction数组未闭合。没有补括号或重写原响应；原raw SHA保留。这里的结构脱敏摘要不是论文证据或科学判分。

The first introduction response was invalid JSON (`finish_reason=stop`), with its root object and chapter array still open at EOF. One Pro request consumed 10,104 known tokens, including 343 reasoning tokens already counted within completion. It hit neither the output cap nor timeout. No summaries, Reviewer, Evaluation, repair or final report were produced. Strict delivery remains 0/3; ViT and NeRF remain unrun.

[绑定的脱敏诊断 / Bound diagnostic](protocol-diagnostic.json) · [本机trace / Local trace](http://localhost:3000/project/litagent-local-project/traces/6bc653f2584fad425244dd947cd6fc64)。LangFuse核对1generation/10,104及科学I/O空，不声称完整trace树。

本次未捕获HTTP原包，不能仅凭journal排除SDK或服务端边界；旧独立HTTP观测曾验证过另一次坏JSON与原包一致，但不能替代本次证明。低强度未解决本次协议失败，科学支持率不可用；不把两个非配对失败试点当作统计质量提升证据。

新增usage完整、未知0、重试0，历史累计752请求／4,238,381已知tokens，旧1,030,414未知保守占用继续单列。原报告、评分、SHA、失败及32保护路径保持，不追加运行至通过。
