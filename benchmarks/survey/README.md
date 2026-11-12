# Survey Benchmark

The benchmark measures a complete literature survey, from finding papers to
writing and evaluating a report. It records which papers and topics the report
covers, whether claims are supported by their citations, whether the report is
ready for delivery, and how much time and model usage the run takes.

The [dataset](dataset.yaml) contains 15 questions. Its source labels were
prepared with AI assistance and approved by the project owner. The
[independent review checklist](HUMAN_REVIEW_CHECKLIST.md) records the work needed
for a separate paper-by-paper audit. The formal controlled baseline is pending.

## Published results

| Record | Scope | Outcome |
| --- | --- | --- |
| [August 2026 pilot](results/pilot-2026-08-25.md) | Nine observations across three topics and three profiles | Checkpoint recovery and usage tracking were exercised. Retrieval inputs differed across profiles, so the comparison did not establish a winner. |
| [September 2026 NeRF run](results/post-repair-pilot-2026-09-08.md) | One full-text run with a same-model Judge | Runtime Faithfulness was 0.8958 and delivery was marked ready under that run's policy. |
| [Revision case 377 → 380](../../src/litagent/static/presentation/frozen-repair-377-380.json) | One report revised using frozen evidence | In the final reassessment, all 398 extracted scientific assertions were supported. |
| [November 2026 run 055](../../docs/closeout.md) | One of three planned topics ran | The Reviewer reached its output limit. Evaluation did not run. |

## Inspect a benchmark plan

The following command previews one question and profile in the historical pilot
workflow. It lists the planned Survey and Judge actions with zero model calls.
Profile definitions are in [profiles.yaml](profiles.yaml).

```sh
litagent benchmark survey --stage pilot \
  --dataset benchmarks/survey/dataset.yaml \
  --profiles benchmarks/survey/profiles.yaml \
  --judge-config benchmarks/survey/judge.example.yaml \
  --case-id few-shot-metric --profile-id external-fulltext --dry-run
```

To browse the packaged records, start the [offline workbench](../../docs/presentation-guide.md)
and open Benchmark.

## Historical Judge runs

These commands describe the historical benchmark workflow. Adding `--live`
sends requests to the configured model services and can incur charges. Set up
the [Judge configuration](judge.example.yaml) and all three budget flags
(`--max-observations`, `--max-product-tokens`, and `--max-judge-tokens`)
before starting a pilot or a bounded ablation.

- `--stage pilot` accepts explicit case and profile selections.
- `--stage ablation` runs the comparison matrix. `--stage baseline --reuse-ablation <path>`
  uses a completed, comparable ablation.
- `--judge-only --reuse-surveys <path>` audits an existing Survey benchmark
  artifact. Add `--dry-run` to see which observations can be reused. A shared
  generation and Judge model is recorded as `self_judged`.

Private reports and traces are stored under `artifacts/`. An interrupted
request with unknown usage stops further calls in that run.
