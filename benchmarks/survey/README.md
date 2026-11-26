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

To browse the packaged records, start the [offline workbench](../../README.en.md#offline-demonstration-without-api-keys)
and open Benchmark.

## Survey latency and the 0.95 delivery experiment

From the repository root, create a dry-run manifest and the single-table report:

```sh
python scripts/run_survey_latency.py --dry-run \
  --output artifacts/benchmarks/survey/latency-faithfulness-95
```

The manifest lists three workbench captures and eighteen controlled replays. It
records per-run and campaign limits, model configuration, input hashes, and the
approval hash. To use tighter campaign limits, pass `--max-tokens` and
`--max-requests` to both invocations. Use `--max-runs 1` in both invocations
to authorize only the first workbench capture while keeping all 21 table rows.
Each new run requires enough unspent
budget for its complete configured maximum. Once the totals are approved,
replace `--dry-run` with `--approve-manifest <approval_sha256>` to run the same
manifest. The live campaign needs the configured model credentials, local
corpus services, and Chromium. Use `--chrome` if Chromium is installed at a
custom path.

To continue from a terminal segment with settled usage, use a new output
directory and pass `--resume-from <previous-output>` to both invocations.
The new manifest validates and preserves prior snapshot bytes and schedules only
the remaining rows. Campaign limits apply to new requests.

If one request in a prior row has unknown usage, pass `--usage-receipt <json>`
to both invocations with the authenticated Ark `GetInferenceRequestUsage`
response for the recorded provider request ID. The response must contain
explicit token counts. An empty result leaves usage unresolved. The new
manifest binds the receipt bytes, and the table links the receipt and displays
the reconciled total alongside the original recorded usage.
[Ark request usage API](https://docs.volcengine.com/docs/ark/get-inference-request-usage-api?lang=zh)

After a stop for unknown usage, an explicitly approved continuation can pass
`--carry-unsettled` to both invocations. It keeps the previous usage unknown,
reserves that run's full token and request limits, and caps the new segment at
the balance of the preceding approval. A new unresolved request stops the
new segment in the same way.

For a separate repair comparison, use a fully observed baseline with
`--resume-from` and pass `--retry-replay-slot <slot-key>` for each selected
blocked replay with settled usage. Pass both new budget limits to each
invocation. The original external-input snapshots are reused. The comparison
directory records the new outcomes with distinct run IDs. The manifest binds
each retry to its original run ID and archive hash.

The compact report is written to `benchmark.md` in the output directory. It
shows delivery, Faithfulness, citation ID validity, consistency, duration and
tokens for each run. Launch the local viewer to open the
workbench links or browse the same table at `/benchmark`:

```sh
python scripts/view_survey_latency.py --campaign artifacts/benchmarks/survey/latency-faithfulness-95-final-run
```

The viewer uses port 8765 by default. Add `--diagnostics <campaign>` to show
completed diagnostic runs and `--revalidation <json>` to show verified offline
delivery checks. Add `--retries <campaign>` to show completed repair retries
beside their original executions and compare strict delivery across the same
matrix. The campaign retains its original artifacts and detailed measurements.
Browser and replay samples have distinct latency boundaries.

The five archived summary-length failures have a separate continuation entry
point. It authenticates the original writer responses and settled ledger, keeps
the accepted body and evidence, and preflights a summary repair before any
model request. It continues through review and Evaluation under the original
strict delivery policy. Its latency starts at the summary checkpoint.

```sh
python -m scripts.run_survey_summary_recovery \
  --baseline artifacts/benchmarks/survey/latency-faithfulness-95-final-run \
  --output artifacts/benchmarks/survey/latency-faithfulness-95-summary-checkpoint \
  --dry-run
```

Once the aggregate limits are approved, replace `--dry-run` with
`--approve-manifest <approval_sha256>`. Pass `--retries` once per comparison
campaign to view the full rerun and the summary continuations together.

If the terminal five-case campaign contains one settled summary-length failure
after all three local attempts, use `--fallback-from <checkpoint-campaign>` with
the original baseline and a fresh output directory. The preflight carries the
three receipts and repair history forward, checks the remaining approved budget,
and validates a fresh summary request against the frozen body. The continuation
retains the failed attempt and has its own Workbench link.

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
