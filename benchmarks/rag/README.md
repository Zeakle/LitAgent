# LitAgent RAG Benchmark

本目录只保存可审计输入：dirty-data case 定义、retrieval judgments 和显式 profiles。
完整运行结果保存在 `artifacts/benchmarks/rag/`，不提交论文正文、PDF 或模型缓存。

## 运行

```powershell
litagent benchmark ingestion --dataset benchmarks/rag/ingestion_cases.yaml
litagent benchmark retrieval --live `
  --dataset benchmarks/rag/retrieval_dataset.yaml `
  --profiles benchmarks/rag/profiles.yaml
```

## 解释

- 主指标：Recall@10、nDCG@10；MRR@10 反映首个相关结果位置。
- 约束：empty/duplicate ratio、p50/p95 latency、index time 与 footprint。
- 不计算单一总分；默认配置变更必须同时满足质量收益与工程约束。
- chunk 对比只在 selected-fulltext profile 中有效。

## 限制

- Judgment 已依据官方 arXiv metadata 做来源核对，当前状态是 `source_reviewed`，尚未
  冒充 `human_reviewed`；正式发布模型结论前仍需人工逐条签核相关性。
- Live latency 依赖硬件、模型缓存和 Qdrant 状态，必须结合 artifact 环境字段解释。
- 本阶段不包含 OCR、公式、表格、图片理解或 Survey 成品质量 benchmark。
