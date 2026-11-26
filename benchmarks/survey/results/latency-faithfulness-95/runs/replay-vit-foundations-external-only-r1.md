# vit-foundations / external-only / r1

[← 返回 Survey Benchmark](../README.md)

原始受控回放 · `replay-vit-foundations-external-only-r1` · 完成

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

归档阻塞项包括 模型供应商拒绝请求、Evaluation 未通过、报告未完成。

## 任务流程

```mermaid
flowchart LR
  n0["arXiv 搜索 × 4<br/>完成"]:::done
  n1["Hugging Face 搜索 × 4<br/>完成"]:::done
  n2["本地召回 × 4<br/>完成"]:::done
  n3["去重<br/>完成"]:::done
  n4["相关性筛选<br/>完成"]:::done
  n5["文献抽取<br/>失败"]:::failed
  n6["关系分析<br/>完成"]:::done
  n7["写作与审阅<br/>跳过"]:::inactive
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
  classDef inactive fill:#eeeef3,stroke:#777786,color:#3d3d4e;
```

同类任务在终态和依赖关系一致时合并显示。箭头来自已归档的任务依赖。

## 阶段耗时与用量

| 阶段或边界 | 记录 |
| --- | --- |
| 外部输入就绪到 Survey 完成 | 78.2 秒 |
| 归档总耗时 | 78.3 秒 |
| 规划 | 0.0 秒 |
| 搜索与召回 | 0.0 秒 |
| 去重与筛选 | 2.1 秒 |
| 全文获取 | 10.1 秒 |
| 抽取（含全文获取） | 76.0 秒 |
| 已记录模型请求 | 5 |
| Tokens（部分用量未确认） | 42,270 |

— 表示归档未提供有效测量。阶段可能并行，耗时不能相加。

## 版本与归档

| 记录 | 结果 |
| --- | --- |
| 初稿 | 未记录 |
| 首次 Evaluation | 未记录 |
| 单独修订版本 | 未记录 |
| 复评 | 未记录 |
| 写作与修订记录 | 未记录 |
| 开始 / 结束 | `2026-11-24T15:30:34+00:00` / `2026-11-24T15:31:52+00:00` |
| 源运行归档 SHA-256 | `2c617779849709b8f935f55ec03ab4e0f3a7b654a2a6c13debc0836bd03a31ce` |
| 报告内容 SHA-256 | `f3ff400c2ef085f90039f2f3a25043236e7ee350b1b55f755a722558920e7276` |
