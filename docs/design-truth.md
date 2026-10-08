# Design Truth

Only claims supported by current repository code, tests, or committed benchmark
artifacts are listed. Planned capacity and unexecuted live results are not
presented as completed behavior.

| Claim | Status | Repository evidence |
| --- | --- | --- |
| Multi-agent Survey DAG | implemented | [`runner.py`](../src/litagent/runner.py), [`test_orchestrator.py`](../tests/test_orchestrator.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Evidence-grounded evaluation/rewrite | implemented | [`evidence.py`](../src/litagent/evidence.py), [`test_phase13_9.py`](../tests/test_phase13_9.py) |
| Scoped Evaluation v2, complete coverage and attribution gates | verified offline; host fragment IDs, two calls, shared materials with actual citation allowlists, 120 items/80,000 chars; live pass-rate improvement unverified | [`scoped.py`](../src/litagent/eval/scoped.py), [`test_scoped_evaluation.py`](../tests/test_scoped_evaluation.py) |
| Host-selected source passages and support-aspect gates | host-support-passages-v2 verified offline; source-owned offsets, citation ownership, explicit abbreviations and conservative relation guards. 1618 regression tests passed; run-004 live semantic acceptance failed, run-005 preflight only | [`scoped.py`](../src/litagent/eval/scoped.py), [`test_support_passages.py`](../tests/test_support_passages.py), [`test_support_basis.py`](../tests/test_support_basis.py) |
| Host-managed method disclosures and short provider-specific queries | verified offline; preserves scientific claims; historical failed reports unchanged | [`review_structure.py`](../src/litagent/review_structure.py), [`test_review_structure.py`](../tests/test_review_structure.py), [`test_quality_search.py`](../tests/test_quality_search.py) |
| Version-bound quality visualization with shared Judge/Jev results | verified offline; bilingual desktop/390px, keyboard and source/evidence navigation; independent audits remain outside delivery gates | [`quality-view.js`](../src/litagent/static/flow-demo/quality-view.js), [`test_quality_view.py`](../tests/test_quality_view.py), [`test_audit_controls.py`](../tests/test_audit_controls.py) |
| Invalid evidence-selection contract stops automatic review, revision and evaluation | verified offline; preserves diagnostic draft and blocks trust/automatic audits; live validation pending | [`adversarial.py`](../src/litagent/agents/adversarial.py), [`runner.py`](../src/litagent/runner.py), [`test_evidence_contract_guard.py`](../tests/test_evidence_contract_guard.py) |
| Flow replay | implemented | [`demo.py`](../src/litagent/demo.py), [`recorder.py`](../src/litagent/observability/recorder.py), [`test_phase14_6.py`](../tests/test_phase14_6.py) |
| Four-layer Memory wiring | implemented with backend degradation | [`manager.py`](../src/litagent/memory/manager.py), [`test_memory.py`](../tests/test_memory.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Versioned review structures | academic_v2 verified offline: seven body sections, host appendix, bibliography before appendix, version-bound workbench records; old profiles preserved. Local ViT structure draft is unverified, with no new paid calls; live content acceptance remains pending | [`review_structure.py`](../src/litagent/review_structure.py), [`test_academic_v2.py`](../tests/test_academic_v2.py), [`test_academic_v2_ui.py`](../tests/test_academic_v2_ui.py) |
| Deterministic citation styles and bibliography exports | implemented; no model calls on format switch | [`citation_formats.py`](../src/litagent/citation_formats.py), [`test_citation_formats.py`](../tests/test_citation_formats.py), [`test_format_controls.py`](../tests/test_format_controls.py) |
| Version-bound independent Judge/Jev workbench audits | implemented and verified offline; live permission/semantic validation pending | [`report_audit.py`](../src/litagent/report_audit.py), [`audit_context.py`](../src/litagent/audit_context.py), [`test_report_audit.py`](../tests/test_report_audit.py), [`test_audit_controls.py`](../tests/test_audit_controls.py) |
| Optional automatic audits after final Evaluation passes | implemented and verified offline; off by default, 1–3 version budget, no restart replay | [`interactive_service.py`](../src/litagent/interactive_service.py), [`report_audit.py`](../src/litagent/report_audit.py), [`test_automatic_audit_integration.py`](../tests/test_automatic_audit_integration.py), [`test_automatic_report_audit.py`](../tests/test_automatic_report_audit.py) |
| Cross-run Paper Corpus | implemented | [`corpus.py`](../src/litagent/rag/corpus.py), [`ingest.py`](../src/litagent/rag/ingest.py), [`test_phase14_1.py`](../tests/test_phase14_1.py) |
| PDF ingestion/provenance | implemented | [`pdf_parser.py`](../src/litagent/rag/pdf_parser.py), [`quality.py`](../src/litagent/rag/quality.py), [`test_phase14_2.py`](../tests/test_phase14_2.py) |
| Incremental hybrid index | implemented | [`vector_store.py`](../src/litagent/rag/vector_store.py), [`retriever.py`](../src/litagent/rag/retriever.py), [`test_phase14_3.py`](../tests/test_phase14_3.py) |
| RAG benchmark | completed | [`RESULTS.md`](../benchmarks/rag/RESULTS.md), [`phase14_3_results.json`](../benchmarks/rag/phase14_3_results.json) |
| Reproducible runtime gate | implemented | [`ci.yml`](../.github/workflows/ci.yml), [`test_phase14_4_e2e.py`](../tests/test_phase14_4_e2e.py), [`test_tools.py`](../tests/test_tools.py), [`test_mcp.py`](../tests/test_mcp.py) |
| 15-case Survey benchmark | implementation and diagnostic pilots complete; formal controlled baseline pending | [`2026-08-25 pilot`](../benchmarks/survey/results/pilot-2026-08-25.md), [`2026-09-08 post-repair pilot`](../benchmarks/survey/results/post-repair-pilot-2026-09-08.md), [`dataset.yaml`](../benchmarks/survey/dataset.yaml), [`test_phase14_5.py`](../tests/test_phase14_5.py) |

## Interpretation Boundary

The Phase 14.3 retrieval summary is committed measured evidence. Phase 14.5 also
has a measured 9-run live pilot with verified local artifacts and LangFuse traces,
but every profile comparison was non-comparable because external inputs drifted,
and one full-text observation was partial. No winning Survey profile or full-text
product gain may be claimed until a controlled, accepted result exists.

The current worktree repairs the observed control and failure paths with frozen
external-input replay, stage-level paper-flow metrics, provider-aware evaluators,
bounded long-context fallback, terminal-result gates, and explicit paid-run
budgets. One post-repair NeRF full-text pilot completed with final runtime
faithfulness 0.8958 and delivery `ready`. It was a self-judged dirty-worktree
single observation, so it validates the repaired path but does not establish a
profile comparison, a full-text product gain, or a formal baseline.
