# Engineering release status · 2026-10-21

## Verified locally

- Default offline regression: **3897 passed, 25 deselected** (8 deprecation warnings).
- Python 3.11.15 / Node 24.14.0 on macOS; direct dependency constraints install in
  a fresh virtual environment with no broken requirements (`pip check`).
- Clean source snapshot has no `.env` or private `artifacts`; presentation health
  ready in **1.837 seconds**, excluding dependency installation.
- Fresh-process import and lifespan deny inference-runtime imports, private archive,
  SQLite, LLM client and audit executor construction; six local exports work.
- Built-in synthetic source ID corrected; bibliography contains the cited fixture
  rather than returning HTTP 200 with an empty reference list.

## Initial bounded real protocol: historical failure and stop

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

## Frozen-input protocol follow-up · 2026-10-21

At `3f84492`, the same frozen few-shot evidence produced valid raw JSON, but
schema validation rejected six outside-section card references and one unknown
concatenated card ID. Only one Pro request ran: 24,382 input + 3,848 output =
28,230 tokens, complete usage, zero retries. No summary or Evaluation request,
report acceptance, or new Survey observation was made. This does not establish
scientific reliability or a general protocol success rate.

[Public protocol result](../benchmarks/survey/results/flat-body-protocol-20261021-001/README.md).
Desktop/390px bilingual text, keyboard expansion and no overflow were checked on
this new standalone result page. Prior workbench visual limitations remain
separate. The live trace has one redacted generation but lacks its root terminal
span; the executor now emits an explicitly unverified completion for future runs.
The live trace was not rewritten. Final offline regression remains 3773 passed.

## Chapter-scoped short references · historical offline candidate

`native-body-summary-v3` asks the model for short chapter-prefixed references;
the host restores the exact stable fact-card IDs before ordinary validation.
Unknown, foreign-section, duplicate and mixed references remain rejected. Complete
source text and semantic fields are unchanged. The seven actual prior reference
errors are preserved as ID/scope regression fixtures, not scientific gold labels.

3798 default tests and 107 related tests passed. A local reconstruction preserves
24 cards, 17 full originals and all 60 permission edges. The complete body request
reservation is 90,505 versus the previous 95,603; this is not actual token savings
or a measured quality improvement. No new paid request was issued. The older paid
protocol CLI refuses the new contract until the latest accounting lineage is wired.
The previous real trial and its failures remain unchanged.

## Short-reference live observation · 2026-10-21

At `519e433`, `native-body-summary-v3` produced valid JSON and passed its reference
schema on the same frozen public few-shot input. One Pro call used 18,682 input +
3,788 output = **22,470 tokens**, complete usage, zero retries. Host validation
then rejected 12 writing-contract issues: five invalid scope notices, three
scientific units without cards, and four ambiguous compound citation scopes.
Summary, Reviewer, Evaluation, Judge and Jev did not run; the report was not
delivered. These are local contract diagnostics, not 12 confirmed scientific errors.

[Public observation and original response SHA](../benchmarks/survey/results/short-card-protocol-20261021-001/README.md).
Its four result formats agree. Desktop/390px bilingual text, keyboard error/identity
expansion, zero console errors and no overflow were checked. LangFuse readback has
one generation / 22,470 tokens with empty generation I/O and an explicitly
unverified root terminal span. Earlier trace and visual limitations remain recorded.
The parent family is now 29 requests / 176,723 tokens; cumulative history is 611
model requests / 3,274,644 known tokens, retaining 1,003,467 older unknown holds.
Protocol probes do not add initial Survey observations.

## Typed body and bounded repairs · current offline validation

At `840c699`, `native-body-summary-v4` limits scientific rows to nonempty
chapter-authorized references. Optional notices select catalog IDs at explicit
positions; the host supplies their fixed text. Body validation can now consume the
existing shared maximum of two located repairs before freezing summaries. All
non-target units and sources remain protected; no-progress candidates, newly
duplicated findings, invalid responses, unknown usage, cancellation and restarts
fail closed. Failure locations bind the actual body or repair response SHA.

**3861 default tests and 173 related tests passed**, including 36 new cases and a
real SDK consumer chain through Synthesis, Reviewer, Runner context and Evaluation
v3, with external responses controlled by fixtures. Scientific and global negative
cases still block completion. All 12 original diagnostics remain reproducible.
The complete 12-target repair reservation is 113,867 with 20,000 output tokens
reserved, below the unchanged 117,964 context ceiling; full sources, original
candidate and permissions are preserved. This is a conservative reservation,
not measured token savings or scientific quality improvement.

This revision made **zero paid calls**. The v4 paid probe remains disabled until a
new bounded executor inherits the latest receipt above; an older budget cannot be
reused. The development service was reloaded and checked healthy with no active
Survey. The frozen presentation and previous result pages remain unchanged.
Live Survey is still experimental, formal cases remain **0/15**, and Phase 14.5
remains in progress pending real semantic validation.

## Typed-body bounded validation · 2026-10-21

Executor `5bc7e7f` inherits both previous protocol charges and the same complete
few-shot evidence. Default offline regression: **3872 passed / 25 deselected**;
final related tests: **56 passed**, including 13 added cases. Two of these cases
were added after default collection and are covered by the final related run.

The real v4 attempt sent **one Pro request / 22,428 tokens** (18,913 input +
3,515 output), complete usage, finish `stop`, no retries. Raw JSON and schema
passed; a read-only replay of the preserved response identifies **seven compound
citation-scope errors**. The complete repair reservation **106,977** exceeds the
inherited writing-stage remainder **100,849**, so no repair request or summary
was sent. Reviewer and Evaluation did not run. No scientific acceptance or
improvement in reliability is inferred.

The parent family now totals 30 requests / 199,151 tokens; all historical charges,
failures and unknown usage holds remain. LangFuse readback has one generation
with empty I/O and 22,428 tokens, plus the completed survey observation marked
unverified / not delivered. Its outer parent is absent from the returned list;
this is not represented as a verified complete trace tree.

[Immutable identity, budget stop, read-only diagnostics and four public formats](../benchmarks/survey/results/typed-body-validation-20261021-001/README.md).
No retrieval, extraction, Judge, Jev or formal 15-case run was added. Live Survey
remains experimental; Phase 14.5 remains in progress.

## Offline body-repair correction · 2026-10-21

The new staged contract `native-body-summary-v5` preserves the initial writing
diagnostics before repair budget checks. A refusal records the original response
and candidate SHA, body problems and independent budget failure, while spending
zero repair attempts. Trace metadata includes only problem codes and counts.

The versioned repair wire replaces repeated ledger IDs with reversible `cN`
references and uses a body-specific prompt. Full sources, prose, card identities
and section permissions remain exactly recoverable. Known sibling schema references
are explicitly expanded; every native object field is required, and each target
allows at most three additional scientific units. Original v6 repair consumers
and single-stage identities retain their existing behavior.

For the immutable failed few-shot input, full-source reservation is **97,555**
versus **106,977** previously, including the unchanged **20,000** output reserve.
The writing stage has 100,849 remaining and the old grant had 97,572: this fixed
input fits by only **17** against the tighter bound. It does not establish a
margin for different inputs; every future call must be checked again. No failed
attempt was resumed or overwritten, and no paid call was added.

Offline validation: **3897 passed / 25 deselected** (8 warnings), **145 related
cases**, including 23 additions; the explicit v4/v5 identity check was strengthened
and passed separately. The real SDK transport chain through repair, Reviewer and
Evaluation is exercised with external responses controlled. A validly formatted
but unsupported scientific claim still blocks strict acceptance. These are
engineering regressions, not human calibration or evidence of live reliability.
The v5 paid probe remains closed pending a separately frozen execution/accounting
migration. Phase 14.5 remains in progress; the 8004 result retains its real failure.
