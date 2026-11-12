# 严格自动验收 Benchmark

自动评测 · Judge disabled / 未启用 · 人工校准未验证。

正式套件：0/15 自动交付；实际运行 0/15。
试点：0/3 自动交付；实际运行 1/3。
停止原因：strict_worker_failed。
用量不完整时，tokens 是包含未知请求预留的保守计账，不是精确实际消耗或费用。

## 模型与历史账本 / Models and prior accounting

| 项目 / Field | 值 / Value |
|---|---|
| 提取 / Extraction | doubao-seed-2-1-lite-260915 |
| 写作与修订 / Writing | doubao-seed-2-1-pro-260915 |
| 核验 / Evaluation | doubao-seed-2-1-lite-260915 |
| 历史首次生成轮数 / Prior generation rounds | 28 |
| 历史模型请求 / Prior model requests | 527 |
| 已知 tokens / Known tokens | 2817185 |
| 未知用量保守预留 / Unknown-usage holds | 1003467 |
| 历史 Jev 请求 / Prior Jev requests | 67 |
| Jev 保守输入占用 / Jev input accounting | 756769 |
| 历史 Judge 请求 / Prior Judge requests | 0 |
| Jev 用量完整 / Complete Jev usage | False |

| 阶段 | 题目 | Evaluation | Judge | Jev | 交付 |
|---|---|---|---|---|---|
| pilot | few-shot-metric | 执行 / execution: not_run · 门禁 / gate: failed | disabled | disabled | blocked |
| pilot | vit-data-efficient | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| pilot | nerf-foundations | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | few-shot-metric | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | few-shot-optimization | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | few-shot-simple-baselines | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | few-shot-cross-domain | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | few-shot-vlm | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | vit-foundations | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | vit-data-efficient | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | vit-hierarchical | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | vit-self-supervised | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | vit-efficiency | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | nerf-foundations | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | nerf-acceleration | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | nerf-sparse-generalization | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | nerf-antialias-unbounded | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |
| formal | nerf-dynamic | 执行 / execution: not_run · 门禁 / gate: unverified | unverified | unverified | not_run |

Jev stage=extraction 时只核验提取事实；正文由 Evaluation / Judge 核验。历史 stage=report 表示终稿复核。

## 服务用量（保守计账）

| 服务 | tokens | 已知请求数 | 用量完整 |
|---|---:|---:|---|
| survey | 124448 | 27 | True |
| judge | 0 | 0 | True |
| jev | 0 | 0 | True |
