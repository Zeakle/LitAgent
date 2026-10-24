# Design Truth

Claims below refer to implementation and recorded evidence. Offline fixtures do
not establish model quality. Public presentation is an engineering release;
Phase 14.5 scientific acceptance remains in progress.

| Claim | Status | Repository evidence |
| --- | --- | --- |
| Multi-agent Survey DAG | implemented | [runner](../src/litagent/runner.py), [orchestrator tests](../tests/test_orchestrator.py) |
| Cross-run Paper Corpus | implemented | [corpus](../src/litagent/rag/corpus.py), [tests](../tests/test_phase14_1.py) |
| PDF ingestion/provenance | implemented | [parser](../src/litagent/rag/pdf_parser.py), [tests](../tests/test_phase14_2.py) |
| RAG benchmark | completed | [measured results](../benchmarks/rag/RESULTS.md), [machine-readable evidence](../benchmarks/rag/phase14_3_results.json) |
| 15-case Survey benchmark | implementation and diagnostic pilots complete; formal controlled baseline pending | [dataset](../benchmarks/survey/dataset.yaml), [methodology](../benchmarks/survey/README.md) |
| Evidence / fact-card preparation | verified offline; semantic fields remain unverified candidates | [quality](../src/litagent/quality.py), [source binding tests](../tests/test_grounded_extraction_claims.py) |
| Scoped Evaluation v3 and strict delivery | verified offline; full coverage and scientific support 1.0 required; no formal live baseline | [batched evaluator](../src/litagent/eval/batched.py), [tests](../tests/test_batched_evaluation.py) |
| Converged writer with fixed chapter requests | execution-v6 verified offline; live draft compiled, Reviewer budget blocked; science unverified | [writer](../src/litagent/writing_v2.py), [chapter regressions](../tests/test_writer_sections.py), [bounded result](../benchmarks/survey/results/frozen-writer-three-domain-20261024-005/README.md) |
| Complete-input budget adapter | execution-v7 verified offline; live follow-up stopped at summary JSON before Reviewer; science unverified | [review](native-input-review-20261024.md), [regressions](../tests/test_native_report_input.py), [bounded result](../benchmarks/survey/results/frozen-writer-three-domain-20261024-006/README.md) |
| Native writer intent and summary boundary | execution-v8 verified offline; live follow-up failed fifth chapter JSON before summaries/Evaluation; science unverified | [review](native-intent-review-20261024.md), [regressions](../tests/test_writer_summary_protocol.py), [bounded result](../benchmarks/survey/results/frozen-writer-three-domain-20261024-007/README.md) |
| Native thinking configuration | execution-v9 verified offline; configured modes reach generation/summary/repair and independent Reviewer; default remains disabled; no new live quality claim | [review](native-thinking-review-20261024.md), [SDK and flow regressions](../tests/test_writer_thinking.py) |
| Reasoning-inclusive output reservation | execution-v10 verified offline; live taxonomy exhausted the 20k total cap with no answer, so no final report/Evaluation; disabled baseline preserved, science unverified | [budget and profile review](native-thinking-review-20261024.md), [actual SDK budget tests](../tests/test_completion_output_limit.py), [bounded result](../benchmarks/survey/results/frozen-writer-three-domain-20261024-008/README.md) |
| Fixed low reasoning and optional usage breakdown | execution-v11 verified offline; live low and reasoning metadata read back, but first introduction JSON invalid before output/time caps; science unverified | [review](native-thinking-review-20261024.md), [regressions](../tests/test_writer_low_effort.py), [bounded result](../benchmarks/survey/results/frozen-writer-three-domain-20261025-009/README.md) |
| Same-schema Responses transport mapping | optional runtime adapter with real SDK/HTTP-mock regression; default Chat and schema unchanged; no real Responses request or quality claim | [compatibility audit](native-transport-review-20261025.md) |
| Lite / Pro stage routing | live calls observed; no controlled quality gain established | [routing](../src/litagent/llm/routing.py), [tests](../tests/test_stage_model_routing.py), [public diagnostic](../src/litagent/static/presentation/manifest.json) |
| Native array writing and inline schema | verified offline; original invalid response remains rejected | [codec](../src/litagent/native_wire.py), [regressions](../tests/test_native_schema_inline.py) |
| Staged body / summary generation | verified offline; first staged body JSON failed; flat frozen-input follow-up JSON parsed but card schema failed | [staged writer](../src/litagent/staged_generation.py), [SDK regression tests](../tests/test_staged_native_generation.py), [bounded result](../benchmarks/survey/results/survey-reliability-20261021-001/README.md) |
| Chapter-scoped short card references | verified offline only; exact stable-ID restoration and original source permissions; no new live result | [staged codec](../src/litagent/staged_generation.py), [real scope regressions](../tests/test_section_card_refs.py) |
| Isolated public presentation | verified by fresh-process import, lifespan and read/export tests; no private archive scan, paid service or recovery | [presentation](../src/litagent/presentation.py), [tests](../tests/test_presentation.py) |
| Academic structures and host retrieval appendix | implemented; version-bound, no full PRISMA claim | [structure](../src/litagent/review_structure.py), [tests](../tests/test_academic_v2.py) |
| Citation styles and six export types | implemented; deterministic, no LLM on style change | [formats](../src/litagent/citation_formats.py), [tests](../tests/test_citation_formats.py) |
| Independent Judge / extraction Jev switches | implemented; new UI defaults off; enabled checks bind exact source/version | [options](../src/litagent/workbench.py), [optional verification tests](../tests/test_optional_verification.py) |
| Budget / cancellation / no automatic replay | implemented and verified offline; unknown usage retained conservatively | [budget](../src/litagent/safety/budget.py), [checkpoint tests](../tests/test_quality_protocol_stop.py) |
| Flow replay / quality visualization | implemented; distinguishes execution, verification and delivery | [demo](../src/litagent/demo.py), [UI tests](../tests/test_quality_view.py) |

## Measured boundaries

Retrieval uses 25 papers and 15 source-reviewed queries, each repeated three times.
RRF Recall@5 is 0.8867 / p50 14 ms; reranking is 0.9089 / 647 ms. This demonstrates
an observed ranking/latency tradeoff, not broad generalization. Full-text profiles
change indexed material and are not a pure ranking ablation.

An earlier 2026-10-21 diagnostic used 26 Lite requests and one Pro request,
126,023 known tokens. Invalid JSON stopped generation before Evaluation, so no
scientific score is available. One of three pilots ran; formal cases are 0/15.
Original response and private evidence remain local; the public record is an
explicitly sanitized replay, identified by its source archive SHA.

An older single [NeRF observation](../benchmarks/survey/results/post-repair-pilot-2026-09-08.md)
reached faithfulness 0.8958 and delivery ready under a different policy/model.
It is self-judged, non-formal and not comparable with current strict runs.

There is no independent human calibration. Scientific accuracy and false
acceptance/rejection rates are unavailable. Passing tests or publishing a
Benchmark bundle does not complete scientific acceptance.

The [frozen-input protocol follow-up](../benchmarks/survey/results/flat-body-protocol-20261021-001/README.md) used one Pro request / 28,230 tokens. Valid JSON still contained seven invalid card references; summary and scientific Evaluation did not run. It is not an additional Survey success.

Stage timeout configuration and inherited deadline admission are implemented and
offline verified (4,082 default tests). The latest settings have **not** undergone
a new paid Survey. [Timeout controls and remaining guarantees](./timeouts.md).

The v11 staged writer has an offline-verified reference-only repair boundary:
exact bad rows and original chapter permissions are frozen, and patches cannot
rewrite scientific prose or expand permissions. It shares the existing two-repair
limit; local repair success is not scientific acceptance. The original 003
failure is preserved. The subsequent v11 live attempt returned a valid repair
refusal and did not reach Evaluation.

## Converged frozen-pool writer (2026-10-23)

`body_summary_v2` uses `grounded-writer-v1`: global short card IDs refer to an
immutable selected pool; per-chapter lists are recommendations. The model writes
only text and card references. The host owns headings, citation spans and summary
dependencies. Evaluation and automatic acceptance consume the same native unit
bindings. Reviewer, local and Evaluation repairs share two attempts, followed by
complete reassessment. Old explicit strategies and API defaults remain readable
and retain their identity semantics. New UI runs select v2.

The fixed three-domain validation reuses existing public evidence snapshots,
without new retrieval/extraction. Judge/Jev are disabled for this experiment.
The [fixed validation](../benchmarks/survey/results/frozen-writer-three-domain-20261023-001/README.md) stopped after its first Pro request: 12,477 known tokens, valid provider completion but invalid JSON at character 6,089. Summary, Reviewer and Evaluation did not run; ViT and NeRF were not requested. Strict delivery is 0/3 planned cases, with no scientific score. No extra attempts were added. This remains a protocol reliability blocker, not a measured scientific failure rate.
The earlier v11 real run returned a valid repair refusal; it did not reach
Evaluation. Its original failure and historical costs remain preserved.

The subsequent [writer plan review](writer-plan-review-20261023.md) closes local
preflight, repair-target, no-progress, budget-ownership and response-accounting
gaps. Offline verification passed 4,181 tests (25 deselected). The original
invalid JSON was replayed unchanged through the real SDK with mock HTTP only;
no new paid request was made. This does not resolve the provider output failure
or establish a live quality improvement.

A [subsequent identical full-input request](../benchmarks/survey/results/writer-v2-protocol-20261023-001/README.md)
completed JSON/schema and local body checks with 66 units, one Pro request and
14,578 known tokens. Provider HTTP content matched the SDK content. No scientific
Evaluation was performed; the previous malformed response remains a failure.
This observation does not establish an improvement in Survey quality or reliability.

The next [fixed three-domain campaign](../benchmarks/survey/results/frozen-writer-three-domain-20261023-002/README.md)
stopped after one Pro response was invalid JSON again (12,422 known tokens).
The request fields and input SHA matched the successful protocol observation.
Reviewer and Evaluation did not run; two cases were not started, and strict
delivery is 0/3. Its trace was read back with one generation, matching usage
and empty scientific I/O. The standalone protocol observer's missing trace
parent was separately corrected offline; its historical trace remains unavailable.

The [provider-schema compatibility observation](../benchmarks/survey/results/writer-v2-schema-compat-20261023-001/README.md)
retains complete host validation while projecting only the native decoder schema.
Offline regression passed 4,211 tests (25 deselected). Its single live Pro request
still returned malformed JSON: HTTP content exactly matched SDK content, with
12,537 known tokens and complete usage. This rules out local SDK content rewriting
for that response, without identifying the remote cause. No retry, summary or
Evaluation followed. Compatibility adaptation has not demonstrated improved
protocol reliability or scientific quality; Phase 14.5 remains in progress.

The [layout-guided fixed campaign](../benchmarks/survey/results/frozen-writer-three-domain-20261023-003/README.md)
completed one full draft, summary and Reviewer, then stopped after independent
extraction produced 132 assertions against the unchanged 128 limit. Support
verification and global checks did not run; 116 provisionally scientific
assertions remain unverified. Four requests used 89,534 known tokens with complete
usage; two planned cases were not started. Strict delivery remains 0/3. This is
progress beyond the prior JSON blocker for one sample, not proof of scientific
reliability. A later read-only view fix exposes the preserved capacity reason and
distinguishes unsent repair preflight from actual repair requests. It does not
alter archived scores, report SHA or execution status.

The [content-guided fixed campaign](../benchmarks/survey/results/frozen-writer-three-domain-20261024-004/README.md)
again stopped on the first Pro response: 13,179 known tokens, malformed JSON, no
Reviewer or Evaluation, strict delivery 0/3. Early chapter nesting was observed;
the 2,380 output tokens did not reach the 20,000 output allowance. This does not
establish a specific remote cause or measure scientific ability.

The subsequent [chapter-request repair](writer-sections-review-20261024.md) uses
the same Synthesis Worker and full document contract, deriving five request schemas
from existing chapter fields. All original sources remain available in every
request; the host assembles and validates the complete body before summaries.
Execution-v6 passed 4,274 default offline tests. SDK regressions cover Pro/Lite
routing, separate request accounting, no downstream budget borrowing and no replay
of a cancelled prefix. Existing scientific gates and two shared repairs remain.
No paid request was made during that offline repair. Its subsequent
[fixed live campaign](../benchmarks/survey/results/frozen-writer-three-domain-20261024-005/README.md)
completed five chapters and summaries with six Pro requests / 76,146 known tokens.
Reviewer input required a 151,731 reservation against 117,964 context allowance;
it was refused before transmission. Evaluation did not run, strict delivery is
0/3, and the two other cases were not started. This establishes one compiled draft,
not scientific quality. A read-only replay of all six bound inputs reproduced its
compiled SHA. The generic archived failure is preserved alongside the underlying
Reviewer budget diagnostic. A separate mocked inventory exposed another oversized
global request; those scripted verdicts are not scientific measurements.
