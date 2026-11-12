# Capabilities and verification

| Capability | Status | Where to check |
| --- | --- | --- |
| Survey task orchestration | Implemented and tested offline | [Orchestration tests](../tests/test_orchestrator.py) |
| Independent editorial review and Evaluation repair loops | Available through the independent-loop profile | [Pipeline tests](../tests/test_separate_review_pipeline.py) |
| Cross-run Paper Corpus | implemented | [Corpus tests](../tests/test_phase14_1.py) |
| PDF ingestion/provenance | implemented | [PDF ingestion tests](../tests/test_phase14_2.py) |
| RAG benchmark | completed | [Measured results](../benchmarks/rag/RESULTS.md) |
| 15-case Survey benchmark | implementation and diagnostic pilots complete; formal controlled baseline pending | [Benchmark methodology](../benchmarks/survey/README.md) |
| Evidence preparation and Jev admission | Implemented for extraction before writing | [Verification options](../src/litagent/workbench.py) |
| Evaluation and strict delivery | Implemented with cited-context assessment and separate report checks | [Metric definitions](evaluation-results.md) |
| Report revision | Observed in a live case using frozen evidence | [Revision case results](../src/litagent/static/presentation/frozen-repair-377-380.json) |
| Citation styles and report exports | Implemented and tested offline | [Citation format tests](../tests/test_citation_formats.py) |
| Request budgets and interruption recovery | Implemented and tested offline | [Protocol stop tests](../tests/test_quality_protocol_stop.py) |
| Offline workbench | Tested with packaged samples and public records | [Presentation tests](../tests/test_presentation.py) |

Live Survey is experimental. Published runs describe their sources, assessment scope and delivery status. The offline workbench includes both a successful revision case and [a run that stopped before Evaluation](closeout.md).
