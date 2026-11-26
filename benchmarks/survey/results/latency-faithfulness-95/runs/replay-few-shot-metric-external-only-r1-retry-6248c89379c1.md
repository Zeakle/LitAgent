# few-shot-metric / external-only / r1

[← 返回 Survey Benchmark](../README.md)

外部输入回放重跑 · `replay-few-shot-metric-external-only-r1-retry-6248c89379c1` · 完成

## 运行结果

| 指标 | 结果 |
| --- | --- |
| 严格交付 | 通过 |
| Faithfulness（门槛 ≥ 0.95） | 99.12%（904/912） |
| 引用 ID 有效性 | 100.00% |
| 内部一致性 | 100.00% |
| 未获支持 / 未核验断言 | 8 / 0 |
| 矛盾对 | 0 |
| 独立 Judge | 未运行 |

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
  n7["写作与审阅<br/>完成"]:::done
  n2 --> n3
  n0 --> n3
  n1 --> n3
  n3 --> n4
  n4 --> n5
  n4 --> n6
  n5 --> n7
  n6 --> n7
  classDef done fill:#dff5e7,stroke:#43865c,color:#173725;
```

同类任务在终态和依赖关系一致时合并显示。箭头来自已归档的任务依赖。

## 阶段耗时与用量

| 阶段或边界 | 记录 |
| --- | --- |
| 外部输入就绪到 Survey 完成 | 1,555.6 秒 |
| 归档总耗时 | 1,560.4 秒 |
| 规划 | 0.0 秒 |
| 搜索与召回 | 0.0 秒 |
| 去重与筛选 | 4.8 秒 |
| 全文获取 | 72.5 秒 |
| 抽取（含全文获取） | 269.3 秒 |
| 写作与审阅 | 319.1 秒 |
| 初稿写作 1 | 305.5 秒 |
| 审阅 1 | 13.4 秒 |
| 首次 Evaluation 1 | 419.4 秒 |
| 修订（含准备） 1 | 86.6 秒 |
| 复评 1 | 455.2 秒 |
| 已记录模型请求 | 53 |
| 已结算 Tokens | 713,346 |

— 表示归档未提供有效测量。阶段可能并行，耗时不能相加。

## 版本与归档

| 记录 | 结果 |
| --- | --- |
| 初稿 | 1 条 |
| 首次 Evaluation | 1 条 |
| 单独修订版本 | 未记录 |
| 复评 | 1 条 |
| 写作与修订记录 | 1 条 |
| 开始 / 结束 | `2026-11-25T13:37:53+00:00` / `2026-11-25T14:03:53+00:00` |
| 原始运行 | [`replay-few-shot-metric-external-only-r1`](replay-few-shot-metric-external-only-r1.md) |
| 源运行归档 SHA-256 | `316c62d9ea8fd4be3dcb82db4b7179076e48125cadecef3029648896c8ab9a31` |
| 报告内容 SHA-256 | `442e541a885894b6f0ddf33bb54f5a8d5d1e91d5feb282a4e075b66773bebd4f` |
