# 三域冻结正文验收

计划3题，运行1题，生成0题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `edfbc13088d84d021268c27877f10f857132dc9a`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|No|not_complete|No|—/—/—|writer_output_truncated|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

冻结模式 / Frozen profile: `writer_reviewer_thinking_v1`。

```json
{
  "id": "writer_reviewer_thinking_v1",
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
  }
}
```

新增已知用量51376 tokens，未知保守占用0，2请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

## 本次真实终态 / Real terminal outcome

固定源码 `edfbc13`、476份执行源码；`writer_reviewer_thinking_v1` 同时开启Pro生成及Lite Reviewer思考，Evaluation保持disabled。实际仅few-shot发出2次Pro请求，第一章本地契约通过，方法分类章达到20,000总输出tokens后 `finish_reason=length`，返回正文为空，按既定门禁停止。

新增51,376已知tokens、usage完整、零重试与新增未知占用。没有摘要、Reviewer、Evaluation、修复或最终报告SHA；ViT与NeRF未运行。严格交付0/3，科学支持率不可用。没有补JSON、放宽上限或追加试跑。

The introduction passed its local contract. The taxonomy request exhausted the 20,000-token reasoning-plus-answer cap and returned no answer (`finish_reason=length`). Two Pro calls consumed 51,376 known tokens with complete usage and no retries. No summaries, Reviewer, Evaluation or final report were produced; the other two cases remain unrun. Strict delivery is 0/3; scientific quality is unverified.

返回的思考字符数不是计费tokens，本版normalizer未持久化可选reasoning-token明细，不能据此推算全部思考用量或正确率。

[原始绑定的脱敏诊断 / Bound sanitized diagnostic](protocol-diagnostic.json) · [本机trace / Local trace](http://localhost:3000/project/litagent-local-project/traces/89c1baa60ec2953de0eaf30bb1f1ef0f)。LangFuse只读核对2个generation/51,376tokens及科学I/O空，不宣称完整trace树。

历史累计751请求/4,228,277已知tokens；旧1,030,414未知保守占用单列不重置。此前报告、评分、SHA、失败与28个保护路径保持。结果不能证明启用思考修复了协议或提高了科学通过率。
