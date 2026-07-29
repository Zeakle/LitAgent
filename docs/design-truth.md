# Design Truth

Only claims provable with current repository artifacts are listed. No aspirational
metrics or future capacity are included without evidence.

| Claim | Status | Repository evidence |
| --- | --- | --- |
| Multi-agent Survey DAG | implemented | [`runner.py`](../src/litagent/runner.py), [`test_orchestrator.py`](../tests/test_orchestrator.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Evidence-grounded evaluation/rewrite | implemented | [`evidence.py`](../src/litagent/evidence.py), [`test_phase13_9.py`](../tests/test_phase13_9.py) |
| Flow replay | implemented | [`recorder.py`](../src/litagent/observability/recorder.py), [`test_run_recorder.py`](../tests/test_run_recorder.py), [`test_api.py`](../tests/test_api.py) |
| Four-layer Memory wiring | implemented with backend degradation | [`manager.py`](../src/litagent/memory/manager.py), [`test_memory.py`](../tests/test_memory.py), [`test_assembly.py`](../tests/test_assembly.py) |
| Cross-run Paper Corpus | pending 14.1 | no production corpus state or ingestion service |
| PDF ingestion/provenance | pending 14.1/14.2 | no production parser |
| RAG benchmark | pending 14.3 | no committed benchmark dataset or result |
| 15-case Survey benchmark | pending 14.5 | no reviewed ground truth or result |
