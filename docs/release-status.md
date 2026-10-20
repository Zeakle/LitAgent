# Engineering release status · 2026-10-20

## Verified locally

- Default offline regression: **3719 passed, 25 deselected** (8 deprecation warnings).
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
