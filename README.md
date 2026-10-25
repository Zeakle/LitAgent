# LitAgent

[![English](https://img.shields.io/badge/lang-English-1769aa.svg)](./README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-c5221f.svg)](./README.zh-CN.md)

An evidence-grounded literature-review workbench with an asynchronous multi-agent
DAG, hybrid retrieval, adversarial review, and version-bound evaluation.

**Engineering demonstration; live Survey remains experimental.** The retrieval
benchmark is measured. A formal 15-case Survey baseline has not been established.
An execution finishing does not mean its report passed quality checks.

## Start the public demonstration

Python 3.11 is required. Install dependencies with internet access, then run the
presentation without API keys, Docker, model downloads, or runtime network calls:

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -c requirements/release-constraints.txt -e ".[api]"
litagent demo --presentation
```

Open [the workbench](http://127.0.0.1:8000/flow-demo) or
[the evidence overview](http://127.0.0.1:8000/presentation).
Use `--no-browser --port 8001` when needed. `--presentation` and `--live` are
mutually exclusive. Presentation mode loads only packaged public records, denies
writes, and does not scan private archives or recover paid tasks.

Three built-in `ready`, `blocked`, and `partial` samples are **synthetic contract
fixtures**, not model-quality measurements. A separate real failure replay shows
why delivery was blocked. [Demo and technical walkthrough](./docs/presentation-guide.md).

For local private archives, the existing `litagent demo` command remains available.

## What the system does

```mermaid
flowchart LR
    Q[Research question] --> P[Planner / bounded DAG]
    P --> S[External search + local RAG]
    S --> E[Dedup / relevance / extraction]
    E --> F[Fact cards / optional extraction Jev]
    F --> W[Synthesis / Reviewer]
    W --> V[Evaluation / bounded repair]
    V --> D[Quality / delivery policy]
    D --> UI[Versioned report / evidence / replay]
```

- Sources, complete selected support text, and host-generated locators remain
  distinct from model-extracted findings; a valid card is not a verified fact.
- Evaluation verifies atomic claims against their actual citations. Candidate
  rewrites stay isolated until complete re-evaluation succeeds.
- Report versions, evidence, SHA, quality and delivery are bound together.
  Incomplete, unsupported or interrupted outputs cannot silently become publishable.
- Fixed request/token budgets, zero automatic retries for controlled trials,
  durable checkpoints, and conservative unknown-usage accounting limit paid work.
- Local artifacts retain detailed diagnostics; current controlled runs send only
  redacted metadata and usage to optional LangFuse.
- `academic_v2` has seven academic sections and a host-managed retrieval appendix.
  IEEE, APA 7 and GB/T citation styles are deterministic local projections.
  Markdown/PDF/Word and BibTeX/RIS/CSL JSON exports do not call an LLM.

[Architecture and ownership](./docs/system-overview.md) ·
[Capability/evidence ledger](./docs/design-truth.md).

## Measured results

Retrieval: **25 papers, 15 queries, 3 repetitions per query**, source-reviewed
judgments without independent human double review; not a general retrieval ranking.

| Profile | Recall@5 | nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: |
| MiniLM + RRF | 0.8867 | 0.8238 | 14 |
| BM25 | 0.8778 | 0.8817 | 4 |
| MiniLM dense | 0.7711 | 0.7440 | 10 |
| RRF + reranker | 0.9089 | 0.9406 | 647 |
| Fulltext section-aware | 0.9222 | 0.8426 | 61 |

Reranking improves ranking metrics but adds substantial latency; it is not the
default. Full-text profiles change indexed material and are separate from pure
ranking comparisons. [Full results and limitations](./benchmarks/rag/RESULTS.md).

Latest frozen-evidence writing validation after account recovery: **1/3 cases
started**, **9 HTTP requests / 149,128 known tokens**, with complete usage.
Five body chapters and the abstract/conclusion compiled after **2/2** shared
protocol recoveries. Lite Reviewer then exhausted its 4,096-token output cap
entirely on reasoning and returned no review answer. Evaluation did not run;
strict delivery remains **0/3**, with two cases not started. The intermediate
draft is unverified and is not a delivered report or a scientific success.
The earlier account-rejection run and its **56,129-token conservative unknown
hold** remain preserved; that hold is not a measured token charge.
No retrieval or extraction was repeated. The earlier end-to-end diagnostic used
**27 requests / 126,023 tokens** and also stopped before Evaluation. Formal cases:
**0/15 run**. These are technical failures, not measured scientific scores.
An older single NeRF run passed an older, less strict policy; it is not success
under the current policy or evidence of a quality improvement.
[Latest writing validation](./benchmarks/survey/results/frozen-writer-three-domain-20261025-015/README.md) ·
[Offline Reviewer configuration repair](./docs/reviewer-role-review-20261026.md) ·
[Provider mapping review](./docs/native-transport-review-20261025.md) ·
[Thinking configuration and budget review](./docs/native-thinking-review-20261024.md) ·
[Offline chapter-generation repair](./docs/writer-sections-review-20261024.md) ·
[Complete-input budget repair](./docs/native-input-review-20261024.md) ·
[Summary protocol review](./docs/native-intent-review-20261024.md) ·
[Earlier end-to-end trial](./benchmarks/survey/results/survey-reliability-20261021-001/README.md) ·
[Survey methodology](./benchmarks/survey/README.md) ·
[Packaged public records](./src/litagent/static/presentation/manifest.json).

## Optional live use

```sh
python -m pip install -c requirements/release-constraints.txt -e ".[dev]"
# Copy .env.example to .env and set an OpenAI-compatible provider key.
docker compose up -d
litagent demo --live
```

The current configuration routes extraction/planning/review/Evaluation to
`doubao-seed-2-1-lite-260915`, and generation/repairs to
`doubao-seed-2-1-pro-260915`. Model access and paid usage are your responsibility.
The new UI uses `strict_auto_v2`, Chinese, `academic_v2`, and IEEE; legacy API
requests retain legacy defaults.

Mandatory Evaluation v3 checks full unit coverage, scientific support, structure,
core coverage and global consistency. `citation_accuracy` means **citation ID
validity**, not semantic correctness. Strict delivery requires scientific support
1.0; the historical 0.8 base threshold remains a separate metric.
Judge and pre-writing extraction Jev are independent optional switches, off in the
new UI. Enabled checks must pass. Judge using a generation-family model is
`self_judged`; Jev confidence is not accuracy. No independent human calibration
has been completed. [Configuration example](./.env.example).

Stage-specific request and Worker deadlines are configured in `config/default.yaml`;
see [timeout settings and cancellation accounting](./docs/timeouts.md).

Academic styles require Node.js (validated locally with 24.14.0). PDF/Word Chinese
rendering requires a CJK font; see the demo guide. There is no OCR, arbitrary-URL
PDF crawling, or submission-format certification.

## Verify

```sh
python -m pytest -v
python -m black --check src tests
python -m isort --check-only --profile black src tests
```

Default tests exclude `integration` and `live`; they do not prove scientific
accuracy. [Release checklist and known limitations](./docs/release.md).

[2026-10-20 release validation / 收尾验收](./docs/release-status.md).

[Screenshot / short screen-capture walkthrough](./docs/presentation-guide.md#保存的演示材料).
