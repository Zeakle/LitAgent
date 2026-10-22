# Engineering release status · 2026-10-22

## Verified locally

- Default offline regression: **4024 passed, 25 deselected** (8 deprecation warnings).
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


## Received-body repair observation · 2026-10-22

At `1d58383`, a separate `frozen-received-body-protocol-v1` executor reconciles
all three earlier protocol receipts and reuses the exact already-charged body
response. It does not resend initial generation or reset historical charges,
shared repair counts, terminal journals or the 500k/64 and phase budgets. Default
writing identities and the retired v5 paid-probe guard remain unchanged.

Offline validation: **3916 passed / 25 deselected** (8 warnings), including
19 new cases for the real SDK boundary, three-hop accounting, immutable inputs,
duplicate grants, interruption and request ceilings. The earlier related run
passed 125 cases; the final full run includes the two subsequently added cases.
Four changed Python files passed Black/isort; history and Plan SHA checks passed.

One new bounded real attempt used exactly **2 Pro requests / 33,273 tokens**,
complete usage, zero retries. The first request used 17,858 input + 1,018 output
and repaired all seven local citation-scope issues without changing protected
units or source permissions. This is a local contract pass, not a scientific
support verdict. The second used 13,424 input + 973 output; its summary JSON ends
incomplete at offset **2854** (line 1, column 2855), despite finish `stop` and the
20,000 output cap. The raw response is preserved, with no delimiter patch or
extra paid request. Reviewer/Evaluation did not run and the result is not deliverable.

The family now totals **32 requests / 232,424 tokens**; historical totals are
614 model requests / 3,330,345 known tokens, retaining the 1,003,467 older unknown
holds. Protocol continuations do not add initial Survey rounds. LangFuse readback
contains two generations / 33,273 tokens with empty generation I/O and an
unverified survey terminal span; its outer parent remains absent from the returned
list. A complete trace tree is not claimed.

[Immutable received-body result and four public formats](../benchmarks/survey/results/received-body-protocol-20261022-001/README.md).
Desktop/390px bilingual text, keyboard expansion, no overflow and no console errors
were checked. The new result is served separately on 8005; 8004 remains unchanged.
Live Survey remains experimental. Next work is offline summary-wire diagnosis;
no new initial body generation, paid replay or formal suite was triggered.
Phase 14.5 remains in progress.


## Offline summary reference wire · 2026-10-22

`native-body-summary-v6` uses `grounded_summaries_v2_refs` for new staged runs.
The model returns flat abstract/conclusion sentence rows and short `bN` selectors;
the host restores exact original scientific body coordinates. Only scientific
units with actual cards enter the menu. Complete source records, prose, conditions,
position order and permissions remain; repeated input card identities use reversible
`cN` keys. The local binding retains original identities and body SHA. Provider
bytes and restored-summary SHA are stored independently. Legacy v1 summary
transport remains explicit; single-stage fingerprints retain their semantics.

The real malformed summary is replayed read-only at its original EOF. Public tests
contain a same-character-length projection with text values redacted, plus distinct
projection and original-response SHA fields. They do not publish the complete raw
model response or infer scientific correctness from the projected fixture.

Offline regression: **3939 passed / 25 deselected** (8 warnings), including
23 new cases. A related run passed 210 cases before the final additional premise
limit check; the full run includes that check and stronger journal binding assertions.
Real SDK test transports exercise Synthesis, repair, Reviewer and Evaluation without
paid calls; valid formatting still does not waive unsupported scientific claims.
Nine changed Python files pass Black/isort. Seven historical protection maps and
Plan SHA remain unchanged. No new UI or visual pass is claimed; 8005 retains the
real failed observation.

The exact repaired body has 50 units, including 48 scientific units, 22 used cards
and 15 complete sources. Full restoration equals the original host input. The
initial flat directory reserved 75,833 tokens; reversible IDs/JSON whitespace
reduction lowers this to **72,317**, retaining the **20,000 output reserve**. The
latest writing-stage balance is only **67,576**: preflight remains blocked by
**4,741**. Even the old v1 request (71,669) no longer fits. This is an input
reservation measurement, not actual usage savings or evidence of model reliability.

No paid replay, new generation or formal suite was started. Parent totals remain
32 requests / 232,424 tokens; history remains 614 model requests / 3,330,345 known
tokens and 1,003,467 older unknown holds. Retired paid executors reject v6 until
latest accounting and an independently bounded execution identity are wired.
Next work is further lossless input encoding and then bounded live validation;
Phase 14.5 remains in progress, and live Survey remains experimental.


## Compact frozen-body summary observation · 2026-10-22

At `22c578d`, staged contract `native-body-summary-v7` encodes every frozen body
unit once in an explicit position table, including scientific `bN` selectors.
Complete sources, text, conditions, identity mappings and all 50 body units remain
exactly recoverable. Full reservation decreases **72,317 → 67,555**, retaining the
20,000 output reserve, versus a writing-stage balance of 67,576. The **21** headroom
applies only to this exact frozen input. No source or budget gate was weakened.

A new summary-only executor reconciles the complete receipt chain through the
previous received-body attempt, reapplies its original repair proof, and retains
the one completed repair attempt. A versioned host-body envelope records the
canonical body and provenance; it is never represented as a new provider response.
The executor permits one summary call, zero initial-body/repair calls and no retries.

Offline validation: **3964 passed / 25 deselected** (8 warnings), including 25 new
cases; 186 related cases passed before four additions included in the full run.
Seven Python files passed Black/isort and staged diff checks. Eight protection
maps remain valid. The prior map's 11 active source paths are checked against their
recorded `3c90c4e` Git bytes; private immutable files remain checked at their original
paths. No historical manifest or protection map was rewritten.

The bounded real call used **10,954 input + 782 output = 11,736 tokens**,
finish `stop`, complete usage, 17.458 seconds, zero retries. Its JSON and native
summary schema passed. Full array-report assembly then failed `schema_const` at
`sections.abstract.blocks.0.heading`: the new summary schema allowed a nonempty
heading while the original complete schema requires an empty string. Read-only
replay finds seven such heading errors. The model also emitted four/three blocks
against the existing two-block rule. These additional replay findings do not
rewrite the original first-error event or terminal receipt.

This exposes a remaining producer/consumer constraint gap. No field was erased,
no response was patched, and no extra paid attempt was sent. Reviewer/Evaluation
did not run; science remains unverified and the report is not deliverable.

[Immutable summary result and four public formats](../benchmarks/survey/results/frozen-summary-protocol-20261022-001/README.md).
Desktop/390px bilingual text, keyboard inspection, zero overflow and console errors
were verified. The new static result uses 8006; 8005 and earlier artifacts remain
unchanged. LangFuse readback has one generation / 11,736 tokens with empty generation
I/O and an unverified terminal survey span; its outer parent is not in the response.

Parent totals are now **33 requests / 244,160 tokens**. History is **615 model
requests / 3,342,081 known tokens**, preserving 1,003,467 older unknown holds and
31 initial Survey rounds. Next work is offline alignment of summary fields and
host constraints, plus the latest remaining-stage budget. Live Survey remains
experimental; Phase 14.5 remains in progress.


## Host-managed summary layout · offline repair, 2026-10-22

Staged contract `native-body-summary-v8` uses `grounded_summaries_v3_layout`.
The provider returns only paragraph indices, scientific text and actual body-unit
dependencies. The host derives empty headings and paragraph type from the complete
report schema and shared summary rules. Each section allows at most two paragraphs
and six sentence units; the existing total capacity, prose-length and scientific
gates remain. Text, dependency order, sources and frozen body are preserved.
The layout and its SHA are recorded with the summary projection. Old v1/v2 codecs
remain explicit, single-stage identity is unchanged, and retired paid executors
reject v8 pending a separate accounting migration.

The actual previous four/three-block layout is retained as a redacted structural
regression: the old native contract accepts it while the complete compiler rejects
it; it is never sanitized into a successful historical result. New tests cover
Chinese/English, layout limits, metadata injection, ordering, reference drift,
complete canonical compilation and SDK response persistence/no replay. Existing
Synthesis/Reviewer/Evaluation tests retain the unsupported-science counterexample.

Default offline regression: **3990 passed / 25 deselected**, eight warnings,
including 26 new cases. The earlier run recorded three stale request-count assertions
in added SDK fixtures and a sandbox loopback bind error; both are preserved in
local acceptance logs. After fixing the count table, the complete suite passed in
an environment allowing the controlled localhost test. Eleven Python files passed
Black/isort and diff checks. Nine historical protection maps and Plan SHA remain
unchanged. No UI/export code changed; no new browser visual acceptance is claimed.

No paid request was issued. With all 50 body units, 22 cards, 15 complete sources
and the unchanged 20,000 output reserve, the v8 request reserves **67,622 tokens**.
The latest writing balance is **55,840**, leaving a deficit of **11,782**. The actual
budget consumer rejects it without changing the ledger. Parent totals stay
**244,160 tokens / 33 requests**; history stays **615 requests / 3,342,081 known
tokens**, with the older 1,003,467 unknown hold preserved. The immutable 8006 result
continues to show the real failure, not this offline repair as a live success.
Further lossless encoding and reconciliation of the latest paid tip are required
before another bounded continuation. Reviewer/Evaluation have not run on a new
real report; live Survey remains experimental and Phase 14.5 stays in progress.


## Complete metadata tables and receipt reconciliation · offline, 2026-10-22

Staged contract `native-body-summary-v9` shares repeated body metadata, card-field
names and section permissions through explicit directories. Source rows select
their own field-set shape, preserving optional fields, null values and extensions.
All original scientific prose, positions, card values and complete source records
round-trip exactly; no text is compressed into model-generated claims. Native
output remains `grounded_summaries_v3_layout`, with the existing canonical and
scientific gates. Historical input codecs and single-stage identity remain intact.

The new read-only `scripts.reconcile_frozen_summary` adapter reproduces the exact
historical v7 summary request and verifies its input SHA, frozen body, shared repair
counter, grant, raw response, per-call usage and cumulative ledger. It creates no
grant or HTTP client. Repeated reconciliation does not duplicate fees. Retired
paid executors continue to reject the new generation contract.

Full private-input replay preserves **50 body units / 48 scientific units / 22
cards / 15 complete sources** and the **20,000** output reserve. Reservation falls
**67,622 → 60,979**, a reduction of **6,643** in the conservative request upper bound,
not measured paid usage. Against the latest writing balance of **55,840**, a
**5,139** deficit remains and the actual budget consumer rejects the request.
No sources were removed and evaluation-stage capacity was not borrowed.

Final offline regression: **4024 passed / 25 deselected**, eight warnings,
including 34 new cases. The 292 related tests and earlier 4020-test full pass preceded
four additional source-completeness counterexamples. Final decoding also rejects
missing source records, unknown card-source links and absent or invalid original
support text; a structurally valid table alone cannot establish complete materials.
Six Python files pass Black/isort and diff checks. Ten immutable protection maps
and the architecture Plan SHA remain valid. No UI code changed or new visual
acceptance was claimed; the 8006 result retains the actual prior failure.

No paid request was issued. Parent totals remain **244,160 tokens / 33 requests**;
history remains **615 model requests / 3,342,081 known tokens**, preserving the older
1,003,467 unknown hold. New v9 inputs have not received real model or scientific
validation. The remaining reservation gap and an independently bounded execution
identity must be resolved before any continuation; the exhausted prior grant is
not reopened. Live Survey remains experimental and Phase 14.5 stays in progress.


## One complete Survey attempt · blocked, 2026-10-22

Frozen source `5d4a7f941ad2e67277db1d1646384c76eb656d7e` ran one few-shot
metric-learning case, with Lite extraction/Evaluation and Pro writing. The
persisted limit was one observation, 500,000 tokens / 64 requests, zero retries;
Judge/Jev were disabled. Other pilot/formal slots were not authorized or run.

Actual usage: **27 requests / 111,166 tokens / 410.837 seconds**, complete usage.
Lite used 26 requests / 88,664 tokens; Pro used one / 22,502. Raw JSON and the
native body schema passed. Canonical body expansion rejected
`flat_body_block_metadata_conflict` at `methods[1].heading`: two rows in the
same block supplied different headings. The original response and failure are
preserved, without normalization or retry. This attempt had sufficient budget;
it did not exercise the v9 summary input on real data.

Summary, Reviewer and Evaluation did not run. Scientific support counts and
faithfulness are unavailable; 24 fact cards and 34 model-labelled scientific rows
are not verified scientific results. There is no valid final-report SHA. The
report is **blocked**, not a successful Survey or a scientific rejection.

[Public observation](../benchmarks/survey/results/survey-reliability-20261022-001/README.md)
provides one projection in JSON/CSV/Markdown/HTML, including exact source/input
bindings, costs, stage statuses and failure location. Full responses and paper
text stay local. History totals become 32 initial rounds / 642 model requests /
3,453,247 known tokens; the older 1,003,467 unknown hold remains unchanged.
Different historical contracts are not pooled into a current success rate.

Preflight regression: **102 passed**. Production Python was unchanged; the
frozen source's **4024 passed / 25 deselected** default result remains the source
validation, not evidence of live scientific quality. Eleven historical protection
maps and the architecture Plan SHA verify unchanged. All four static formats
match their files; archive API reads succeed. Desktop/390px, English/Chinese,
keyboard failure-location expansion and console checks passed. LangFuse readback
confirms 27 generations / 111,166 tokens with empty scientific I/O; an absent outer
parent means the complete trace tree was not verified.

The next offline task is to remove or bind repeated block metadata while keeping
all scientific prose, citations, ordering and permissions. No extra paid request
was made after this failure. Live Survey remains experimental and Phase 14.5
stays in progress.


## Single-declaration body block metadata · offline, 2026-10-22

Staged contract `native-body-summary-v10` requests `grounded_body_v5_blocks`.
Each block declares its heading/type once in an ordered per-section directory.
Scientific rows retain block indices, exact prose, section-scoped card references
and existing unit fields. The host restores metadata before the original canonical,
source, language and scientific checks. Distinct blocks are never merged, even
when headings match. Invalid or unused directory entries, mixed legacy/new fields,
noncontiguous indices and unauthorized references fail closed.

This removes the repeated-header equality requirement that blocked the latest
real attempt. It does not validate the scientific meaning of a heading or sentence.
The actual prior response still fails at `methods[1].heading` under its original
v4 codec. Read-only replay preserves all **34 scientific rows / 2 host notices**,
raw response SHA and charges. The public regression fixture is explicitly redacted
structural data, not the complete model response or scientific gold.

The new staged fingerprint prevents reuse across the change; single-stage identity
and explicit historical v4 decoding remain intact. Received-v1 and frozen-summary
accounting readers explicitly consume the original codec. An internal HTTP observer
recognizes the new schema name, while old paid-executor version/grant guards remain
closed to v10. The v9 lossless summary metadata tables and v3 host layout continue
unchanged; no additional model call or budget was introduced.

Final default regression: **4058 passed / 25 deselected**, eight existing warnings,
including **34 new cases**. The latest focused compatibility suite passed **168**
cases. Earlier failures are retained locally: the structural fixture initially
counted host notices as scientific rows, and expanded tests exposed historical
readers and the HTTP observer selecting the wrong wire. Eleven changed Python
files pass Black/isort; diff checks and twelve historical protection maps, including
the architecture Plan SHA, pass. Real SDK-boundary tests cover writing, Reviewer,
Evaluation, bounded repair, failure persistence and no replay; only HTTP is mocked.

No paid request was issued. The previous blocked report, fees and 8007 result are
unchanged. No UI/export code changed and no new browser visual acceptance is claimed.
A separately bound real trial is still needed to establish live behavior; offline
contract success does not prove improved scientific pass rate. Live Survey remains
experimental and Phase 14.5 stays in progress.


## v10 single-case real validation · extraction timeout, 2026-10-22

Frozen source `5ad66f6f4e23b7c25cb0ce5d474ffe6fe7d023a3` ran exactly one new
few-shot case, capped at 500,000 tokens / 64 requests, zero retries, Judge/Jev off.
Offline preparation exactly reproduced the previous 24 cards / 12 complete sources
and legacy schema, including Synthesis's existing selected-union summary scopes.
The new full body request reserves **89,897**, below **117,964**, retaining the
20,000 output reserve. This fixed-material check did not guarantee wall-clock
capacity for a fresh retrieval/extraction run.

The real run stopped after **315.523 seconds**. Planner's extraction Worker has a
hardcoded **300-second** deadline. It cancelled the final extraction call after
4.679 seconds, leaving its response unknown. **26 requests were sent: 25 completed,
one cancelled without a response.** Known provider usage is **96,704 tokens**;
the unknown request retains its exact conservative reservation of **26,947**.
The run ledger is **123,651**, while the suite retains its full **500,000** cap.
Neither ledger figure is a known actual bill, and they are not added together.
The unknown-usage gate stopped further requests; there was no retry or extra case.

Pro writing, summaries, Reviewer and Evaluation never ran. No complete report was
frozen; a digest of partial placeholder text is not presented as a final-report
SHA. The v10 block-directory contract therefore remains **offline validated only**.
The timeout is an engineering failure, not evidence that scientific claims failed.
[Public result](../benchmarks/survey/results/survey-reliability-20261022-002/README.md)
separates known usage, unknown holds, stage status and unavailable scientific counts.
The earlier 8007 failure remains unchanged.

Cumulative records: **33 initial rounds / 668 sent model requests / 3,549,951 known
tokens / 1,030,414 conservative unknown tokens**, including the older 1,003,467 hold.
These trials span different contracts and materials and do not establish a paired
improvement or calibrated scientific accuracy.

All four public formats match their served bytes and shared projection; exact
archive API reads succeed. Desktop/390px, English/Chinese, document language,
keyboard diagnostic expansion and console checks pass. LangFuse confirms 26
generations, 96,704 known tokens, one unknown generation and empty scientific I/O.
Its survey output separately records the 123,651 run ledger; an absent outer parent
means the whole trace tree was not verified. Thirteen historical protection maps,
the architecture Plan SHA and all 242 frozen input files remain unchanged. No
production Python changed; the source's prior 4058-test offline pass is retained
as source validation, not a new scientific success.

The next offline task is to align extraction Worker, batch and orchestration
deadlines and check remaining time before sending queued requests. The stopped
trial and its unknown request will not be revived or replayed. Live Survey remains
experimental and Phase 14.5 stays in progress.

## Configurable stage timeouts · offline, 2026-10-22

The extraction Worker is no longer fixed at 300 seconds. The current YAML uses
20 minutes for extraction, 30 minutes for the writing/review chain and 70 minutes
for the DAG, with a separate 30-minute Evaluation cap. Pro body calls use 300
seconds; summaries and Reviewer calls use 180 seconds. All values are server
configuration; see [settings and units](./timeouts.md).

Nested deadlines propagate to the actual provider transport. If the remaining
window cannot fit the complete request timeout plus a five-second margin,
`llm.not_sent` records `request_deadline_insufficient` before HTTP, request counting
or unknown-usage charging. Cancellation after sending retains the existing
conservative unknown charge and never automatically retries. Token limits,
scientific gates and historical terminal records remain unchanged.

**24 new regressions; 4,082 passed / 25 deselected / 8 warnings** in the final
default offline suite; 23 changed Python files pass Black/isort and diff checks.
Four historical config fingerprints and fourteen historical preservation groups
remain valid. The initial eight fixture failures are retained locally; fixtures
now explicitly initialize omitted client policy fields and give synthetic graphs
the current configured Worker window before freezing. Short real deadlines still
reject without HTTP. No paid call or new visual validation was performed; the
previous failed trial was not resumed. New limits require a live-service restart
and do not yet demonstrate successful real body generation. Phase 14.5 remains
`in_progress`.
