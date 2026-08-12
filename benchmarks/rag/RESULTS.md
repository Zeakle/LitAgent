# Phase 14.3 RAG Benchmark Results

## 结论

- 运行基线：提交 `8976ac6`，25 篇论文、15 条 `source_reviewed` query、每条重复 3 次。
- 生产默认保持 `abstract + all-MiniLM-L6-v2 + RRF`，本阶段不修改 `default.yaml`。
- 质量优先的可选检索是 `RRF + CrossEncoder rerank`，但 p50 从 14 ms 增至 647 ms。
- SPECTER2 改善 Recall@5、MRR 与 nDCG，但依赖 benchmark extra、首次模型下载和明显更高内存；当前 judgment 规模不足以支持默认切换。
- selected-fulltext 场景推荐 `section_aware`：Recall@5 最高且索引点数显著少于 page-block/recursive；其 p95 更高，因此不改变默认 `content_mode=abstract`。

## Retrieval

| Profile | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Points | Footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM + RRF baseline | 0.8867 | 0.9556 | 1.0000 | 0.8056 | 0.8238 | 14 | 17.0 | 25 | 143,019 B |
| BM25 | 0.8778 | 0.9389 | 1.0000 | 0.9167 | 0.8817 | 4 | 6.0 | 25 | 143,019 B |
| MiniLM dense | 0.7711 | 0.9289 | 0.9867 | 0.7302 | 0.7440 | 10 | 12.3 | 25 | 143,019 B |
| RRF + rerank | 0.9089 | 0.9700 | 0.9867 | 1.0000 | 0.9406 | 647 | 653.3 | 25 | 143,019 B |
| Fulltext page-block | 0.8444 | 0.9833 | 1.0000 | 0.8556 | 0.8300 | 44 | 47.9 | 7,541 | 35,791,342 B |
| Fulltext recursive | 0.8311 | 0.9833 | 1.0000 | 0.8500 | 0.8237 | 45 | 49.3 | 7,670 | 36,653,358 B |
| Fulltext section-aware | 0.9222 | 0.9778 | 1.0000 | 0.8167 | 0.8426 | 61 | 125.9 | 1,635 | 12,002,042 B |
| SPECTER2 + RRF | 0.9089 | 0.9444 | 1.0000 | 0.9333 | 0.8942 | 32 | 36.6 | 25 | 181,594 B |

所有 profile 的 empty result rate 与 duplicate paper ratio 均为 0。完整 query 结果、环境、fingerprint 与 run ID 保存在本地 `artifacts/benchmarks/phase14-3-final/`；提交版摘要见 `phase14_3_results.json`。

## Ingestion Robustness

17 个 case 覆盖单/双栏、页眉页码、扫描件、损坏/加密/伪 PDF、重复块、版本更新、prompt injection、乱码和多页 locator。outcome、reason code、metadata、locator、duplicate suppression 与 incremental update 的适用样本准确率均为 1.0，p50/p95 为 16/37.8 ms。

## 解释边界

- 相关性 judgment 已按官方 arXiv metadata 核对，但尚未完成独立人工双审，不能把结果外推为通用学术检索排名。
- 25 篇语料使 Recall@20 接近饱和；模型选择主要参考 Recall@5、nDCG@10、MRR@10、延迟、索引规模和运行依赖。
- 冷启动下载时间不计入 query latency。SPECTER2 首次下载曾因官方端点连接中断，使用 `HF_ENDPOINT=https://hf-mirror.com` 完成；这属于部署成本证据。
- MuPDF 对少数 PDF 输出 color-space syntax warning，但解析、locator、索引和检索均成功；未宣称支持 OCR、公式、表格或图片理解。
