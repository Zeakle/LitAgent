# Engineering release status · 2026-10-21

## Verified locally

- Default offline regression: **3758 passed, 25 deselected** (8 deprecation warnings).
- Python 3.11.15 / Node 24.14.0 on macOS; direct dependency constraints install in
  a fresh virtual environment with no broken requirements (`pip check`).
- Clean source snapshot has no `.env` or private `artifacts`; presentation health
  ready in **1.837 seconds**, excluding dependency installation.
- Fresh-process import and lifespan deny inference-runtime imports, private archive,
  SQLite, LLM client and audit executor construction; six local exports work.
- Built-in synthetic source ID corrected; bibliography contains the cited fixture
  rather than returning HTTP 200 with an empty reference list.

## Bounded real protocol: failed; no further paid runs

Full frozen input from the latest failed Survey, 21 selected cards with complete
source text. One Pro request with lossless inline schema, 120k total cap / 20k
maximum output / zero retries. Actual use **23,744 input + 4,152 output = 27,896 tokens**,
finish `stop`, complete usage. Invalid JSON at **line 1, column 8288, offset 8287**.
The original response is preserved and rejected; it was not repaired by adding delimiters.

No new Survey, Reviewer or Evaluation was started by this protocol test. The
conditional three-domain validation did not run. Formal Survey cases remain **0/15**.
This does not prove the cause is model capability, and does not establish scientific
support or accuracy. Live Survey remains experimental; Phase 14.5 is in progress.

[Public protocol binding and accounting](../src/litagent/static/presentation/protocol.json).
The original failure replay and retrieval measurements remain separately identified.

## Export and UI verification

- PDF and Word fixture exports rendered and inspected: one page each, no clipping
  or overlapping content; Markdown, PDF, Word and bibliography exports share the
  same source SHA. Word rendering used the bundled LibreOffice runtime.
- Desktop and 390 CSS-pixel Chinese/English views, document language, keyboard
  evidence/diagnostic expansion and unavailable scientific scores verified.
- [Short screen-capture walkthrough](images/demo-walkthrough.mp4) is a 25-second
  sequence of captured views, with subtitles, not a real-time generation recording.
- The actual protocol LangFuse readback has one generation, empty input/output,
  and 23,744 input / 4,152 output tokens; no scientific result is inferred.
- GitHub API and origin access currently fail (including `github.com` DNS
  resolution), so remote CI and tag publication are not claimed complete.

## Pending external validation

Remote Linux CI and release/tag publication are only complete after actual GitHub
results are available. Local macOS tests do not substitute for that result.

## Subsequent bounded reliability trial

Commit `aa0ade7` removes duplicated analytical instructions and contradictory
writing/display order in new array requests; source text, schema and gates remain.
A fresh three-domain attempt ran **1/3** before its technical stop: **27 requests /
128,392 tokens**, complete usage. Pro returned invalid JSON at offset **7879**
(line 1, column 7880), finish `stop`, 3,185 output tokens below the 20,000 cap.
Evaluation did not run; ViT/NeRF and all formal cases remain unrun. No retries or
extra attempts were issued. These fixes do not establish scientific reliability.

[Reproducible result and public four-format exports](../benchmarks/survey/results/survey-reliability-20261020-001/README.md).
The packaged presentation remains an explicitly frozen earlier diagnostic replay;
the new local workbench Benchmark is available under `survey-reliability-20261020-001`.

## Two-stage generation trial · 2026-10-21

Commit `0163e56` implements body-first / summary-second requests inside the same
writer and shared budget. The bounded trial again stopped at **1/3 pilots**:
26 Lite + 1 Pro requests, **126,023 tokens**, complete usage. The body response
has an unclosed root object at offset 7393 (line 1, column 7394), finish `stop`,
24,380 input + 3,376 output tokens. No summary request, Reviewer or Evaluation
ran; no scientific score or delivery success is inferred. ViT/NeRF and formal
15 cases were not dispatched. No retry or delimiter repair was accepted.

[Immutable trial bindings and four public formats](../benchmarks/survey/results/survey-reliability-20261021-001/README.md).
LangFuse readback contains 27 generations / 126,023 tokens with empty generation
I/O. API four-format exports match the public files byte for byte. The packaged
8001 presentation remains the earlier frozen diagnostic; it is not overwritten.
The complete browser visual acceptance was interrupted by approval review usage
limits; an archive-list timeout also remains unresolved. No new visual pass is claimed.

The subsequent flat-body wire revision passed 95 related tests and the default
3758-test offline suite (25 deselected). It remains an offline candidate. It preserves
full sources and exact sentence/block mappings; it is not evidence of improved
live protocol or scientific reliability. Phase 14.5 remains in progress.
