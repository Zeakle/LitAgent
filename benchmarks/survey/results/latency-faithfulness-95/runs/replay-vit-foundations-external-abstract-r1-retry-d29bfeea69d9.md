# vit-foundations / external-abstract / r1

[← 返回 Survey Benchmark](../README.md)

摘要检查点续跑 · `replay-vit-foundations-external-abstract-r1-retry-d29bfeea69d9` · 完成

## 运行结果

| 指标 | 结果 |
| --- | --- |
| 严格交付 | 通过 |
| Faithfulness（门槛 ≥ 0.95） | 100.00%（587/587） |
| 引用 ID 有效性 | 100.00% |
| 内部一致性 | 100.00% |
| 终稿引用精确率 | 0.00% |
| 未获支持 / 未核验断言 | 0 / 0 |
| 矛盾对 | 0 |
| 独立 Judge | 未运行 |

## 任务流程

```mermaid
flowchart LR
  n0["arXiv 搜索 × 4<br/>复用"]:::reuse
  n1["Hugging Face 搜索 × 4<br/>复用"]:::reuse
  n2["本地召回 × 4<br/>复用"]:::reuse
  n3["去重<br/>复用"]:::reuse
  n4["相关性筛选<br/>复用"]:::reuse
  n5["文献抽取<br/>复用"]:::reuse
  n6["关系分析<br/>复用"]:::reuse
  n7["写作与审阅<br/>完成"]:::done
  n2 --> n3
  n0 --> n3
  n1 --> n3
  n3 --> n4
  n4 --> n5
  n4 --> n6
  n5 --> n7
  n6 --> n7
  classDef reuse fill:#e1edfb,stroke:#5077aa,color:#173155;
  classDef done fill:#dff5e7,stroke:#43865c,color:#173725;
```

同类任务在终态和依赖关系一致时合并显示。箭头来自已归档的任务依赖。

复用了 `replay-vit-foundations-external-abstract-r1` 的 16 个已完成任务。复用节点没有本次耗时或模型用量。

## 阶段耗时与用量

| 阶段或边界 | 记录 |
| --- | --- |
| 摘要检查点到 Survey 完成 | 414.0 秒 |
| 归档总耗时 | 414.6 秒 |
| 写作与审阅 | 39.8 秒 |
| 初稿写作 1 | 25.4 秒 |
| 审阅 1 | 14.3 秒 |
| 首次 Evaluation 1 | 367.3 秒 |
| 已记录模型请求 | 9 |
| 已结算 Tokens | 241,229 |

— 表示归档未提供有效测量。阶段可能并行，耗时不能相加。

## 版本与归档

| 记录 | 结果 |
| --- | --- |
| 初稿 | 1 条 |
| 首次 Evaluation | 1 条 |
| 单独修订版本 | 未记录 |
| 复评 | 未记录 |
| 写作与修订记录 | 1 条 |
| 开始 / 结束 | `2026-11-26T02:34:52+00:00` / `2026-11-26T02:41:47+00:00` |
| 原始运行 | [`replay-vit-foundations-external-abstract-r1`](replay-vit-foundations-external-abstract-r1.md) |
| 源运行归档 SHA-256 | `844e3bbc06bc22b36b797ae88e2d7c588c62b22a2fe3df92bc6d2c0e7936f2ab` |
| 报告内容 SHA-256 | `5001b3437301c098ae9bf77bdd5871af65b6528fccfb4d8624df9d13eb43421d` |
