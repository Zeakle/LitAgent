# Design Truth

Only claims supported by current repository code, tests, or committed benchmark
artifacts are listed. Planned capacity and unexecuted live results are not
presented as completed behavior.

| Claim | Status | Repository evidence |
| --- | --- | --- |
| Multi-agent Survey DAG | implemented | [`runner.py`](../src/litagent/runner.py), [`test_orchestrator.py`](../tests/test_orchestrator.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Evidence-grounded evaluation/rewrite | implemented | [`evidence.py`](../src/litagent/evidence.py), [`test_phase13_9.py`](../tests/test_phase13_9.py) |
| Flow replay | implemented | [`demo.py`](../src/litagent/demo.py), [`recorder.py`](../src/litagent/observability/recorder.py), [`test_phase14_6.py`](../tests/test_phase14_6.py) |
| Four-layer Memory wiring | implemented with backend degradation | [`manager.py`](../src/litagent/memory/manager.py), [`test_memory.py`](../tests/test_memory.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Cross-run Paper Corpus | implemented | [`corpus.py`](../src/litagent/rag/corpus.py), [`ingest.py`](../src/litagent/rag/ingest.py), [`test_phase14_1.py`](../tests/test_phase14_1.py) |
| PDF ingestion/provenance | implemented | [`pdf_parser.py`](../src/litagent/rag/pdf_parser.py), [`quality.py`](../src/litagent/rag/quality.py), [`test_phase14_2.py`](../tests/test_phase14_2.py) |
| Incremental hybrid index | implemented | [`vector_store.py`](../src/litagent/rag/vector_store.py), [`retriever.py`](../src/litagent/rag/retriever.py), [`test_phase14_3.py`](../tests/test_phase14_3.py) |
| RAG benchmark | completed | [`RESULTS.md`](../benchmarks/rag/RESULTS.md), [`phase14_3_results.json`](../benchmarks/rag/phase14_3_results.json) |
| Reproducible runtime gate | implemented | [`ci.yml`](../.github/workflows/ci.yml), [`test_phase14_4_e2e.py`](../tests/test_phase14_4_e2e.py), [`test_tools.py`](../tests/test_tools.py), [`test_mcp.py`](../tests/test_mcp.py) |
| 15-case Survey benchmark | implementation complete, live acceptance pending | [`dataset.yaml`](../benchmarks/survey/dataset.yaml), [`test_phase14_5.py`](../tests/test_phase14_5.py) |

## Interpretation Boundary

The Phase 14.3 retrieval summary is committed measured evidence. Phase 14.5 has
versioned inputs, execution code, and deterministic tests, but its 39 live Survey
runs have not been accepted. No winning Survey profile or full-text product gain
may be claimed until those artifacts and traces exist.
