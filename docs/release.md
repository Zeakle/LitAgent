# Engineering release acceptance

This release packages reproducible engineering capabilities. It does not declare
Phase 14.5 or the 15/15 scientific quality objective complete.

## Fixed release boundary

- Python 3.11; direct dependency versions in `requirements/release-constraints.txt`.
  This is not a complete transitive lock. Record platform and Node version with tests.
- Presentation is packaged-only, read-only and does not require API keys, Docker,
  model weights, private history, or runtime network access. Installation uses pip/network.
- Existing offline/private replay and live API contracts retain their meanings.
- Latest public real diagnostic is explicitly unverified; built-in ready is synthetic.
- No new Agents, retrieval architecture, LLM comparison or automatic full 15-case run.

## Required checks

1. Relevant regression tests and default `python -m pytest -v`.
2. Black 88 and isort black for changed Python files; `git diff --check`.
3. Fresh Python environment and checkout/wheel install; no `.env` or private artifacts.
4. Presentation import/lifespan does not construct private repositories, recovery,
   model clients or audit executors. Mutations fail before any task is scheduled.
5. Desktop/390 CSS pixels, Chinese/English, keyboard, source navigation and six exports.
6. JSON/CSV/Markdown/HTML counts agree, including planned and unrun rows.
7. Tracked files contain no credentials, raw private responses, local artifacts or PDFs.
8. Push the reviewed commits. Create the engineering tag only after mandatory checks pass.
   Remote Linux CI is pending until its actual result can be read.

Current acceptance logs remain under ignored `artifacts/acceptance/closeout-20261104/`.
A short public status summary is written after checks, without private paths or prompts.

## Live validation boundary

The current engineering closeout uses one frozen-source run each for few-shot,
ViT and NeRF, in that order, reusing existing public evidence without retrieval or
extraction. `engineering-closeout-v1` fixes the existing writer, source-only view,
separated evaluation dimensions and at most one inventory recovery within two
shared modifications. It adds no public API switches or model response protocol.

Pro writes and repairs; Lite reviews and evaluates. Thinking is disabled;
Judge/Jev are off and transport retries are zero. Each case is admitted before
execution with a fixed plan, at most 4.5M tokens / 64 requests; the suite ceiling
is 13.5M / 192. Historical charges and unknown holds remain separate. Technical,
identity or unknown-usage failures stop further paid work; scientific and valid
quality failures remain visible and do not remove cases from the denominator.
There is no extra protocol probe or second suite to obtain a successful outcome.

Strict scientific support remains 1.0 with full coverage and global/core checks.
Engineering release does not require all three scientific outcomes to pass.
When they do not, live Survey stays experimental and Phase 14.5 stays in progress.
No score, original report or previous charge is rewritten for release.

[Current completed trial and closeout status](closeout.md). Earlier bounded
protocol grants and results below are historical; their smaller limits do not
describe the current suite.

## Known limitations

- Live structured-output reliability and end-to-end strict quality are unproven.
- No independent human calibration; accuracy and false acceptance/rejection unavailable.
- Retrieval dataset is small, with source-reviewed rather than independent double-reviewed labels.
- Legacy successful scores are not comparable with current strict policy scores.
- No OCR, arbitrary-URL PDF crawler, or complete PRISMA/submission-format claim.
- macOS verification does not by itself prove Linux/Windows operation.

[Current verified status and bounded protocol outcome](release-status.md).

The later explicit reliability request opened a separate capped three-observation
attempt at commit `aa0ade7`. It stopped after its first protocol failure; it did not
replay the earlier protocol grant or expand into the formal matrix. See the
[bounded trial evidence](../benchmarks/survey/results/survey-reliability-20261020-001/README.md).

The subsequent body/summary split at `0163e56` also stopped in its first body
response, before summary or Evaluation: 27 requests / 126,023 known tokens.
[2026-10-21 trial](../benchmarks/survey/results/survey-reliability-20261021-001/README.md).
No current strict scientific baseline is established by these technical trials.
