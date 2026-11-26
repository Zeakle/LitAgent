# vit-foundations / external-fulltext / r2

[← 返回 Survey Benchmark](../README.md)

原始受控回放 · `replay-vit-foundations-external-fulltext-r2` · 完成

## 运行结果

| 指标 | 结果 |
| --- | --- |
| 严格交付 | 未通过 |
| Faithfulness（门槛 ≥ 0.95） | — |
| 引用 ID 有效性 | — |
| 内部一致性 | — |
| 未获支持 / 未核验断言 | — / — |
| 矛盾对 | — |
| 独立 Judge | 未运行 |

归档阻塞项包括 摘要或结论超过长度上限、Evaluation 未通过、报告未完成。

## 任务流程

```mermaid
flowchart LR
  n0["arXiv 搜索 × 4<br/>完成"]:::done
  n1["Hugging Face 搜索 × 4<br/>完成"]:::done
  n2["本地召回 × 4<br/>完成"]:::done
  n3["去重<br/>完成"]:::done
  n4["相关性筛选<br/>完成"]:::done
  n5["文献抽取<br/>完成"]:::done
  n6["关系分析<br/>完成"]:::done
  n7["写作与审阅<br/>失败"]:::failed
  n2 --> n3
  n0 --> n3
  n1 --> n3
  n3 --> n4
  n4 --> n5
  n4 --> n6
  n5 --> n7
  n6 --> n7
  classDef done fill:#dff5e7,stroke:#43865c,color:#173725;
  classDef failed fill:#fde7e6,stroke:#ba524a,color:#611c18;
```

同类任务在终态和依赖关系一致时合并显示。箭头来自已归档的任务依赖。

## 阶段耗时与用量

| 阶段或边界 | 记录 |
| --- | --- |
| 外部输入就绪到 Survey 完成 | 594.6 秒 |
| 归档总耗时 | 595.9 秒 |
| 规划 | 0.0 秒 |
| 搜索与召回 | 0.3 秒 |
| 去重与筛选 | 3.2 秒 |
| 全文获取 | 6.0 秒 |
| 抽取（含全文获取） | 251.9 秒 |
| 写作与审阅 | 338.7 秒 |
| 初稿写作 1 | 338.6 秒 |
| 已记录模型请求 | 36 |
| 已结算 Tokens | 259,002 |

— 表示归档未提供有效测量。阶段可能并行，耗时不能相加。

## 版本与归档

| 记录 | 结果 |
| --- | --- |
| 初稿 | 未记录 |
| 首次 Evaluation | 未记录 |
| 单独修订版本 | 未记录 |
| 复评 | 未记录 |
| 写作与修订记录 | 1 条 |
| 开始 / 结束 | `2026-11-25T08:07:08+00:00` / `2026-11-25T08:17:04+00:00` |
| 源运行归档 SHA-256 | `a8a66ddbf79b4d585dd5e44c6a2f511abf334fb12243ba76b7db57dde7eb39b3` |
| 报告内容 SHA-256 | `8c8205bbbf6594919fc2d415c367e049c0691252a9c32d20bcca2e087e3fbaa2` |
