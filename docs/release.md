# Release checks

1. Install from a clean checkout with Python 3.11 and [release dependency constraints](../requirements/release-constraints.txt).
2. Run the offline test suite with `python -m pytest -v`. Check formatting with Black and isort, then run `git diff --check`.
3. Start the [offline workbench](presentation-guide.md) without API keys. Open a report, follow a citation to its evidence, and check the quality and execution views in both languages. Repeat at desktop and mobile widths.
4. Export Markdown, PDF, Word and bibliography data. Confirm that all exports use the selected report version and that Chinese text renders where applicable.
5. Compare the public experiment pages with their JSON results. Check that unpublished reports, credentials and local run directories are excluded from the release.
6. Check the [CI result for the release commit](release-status.md) before tagging.

The offline workbench uses packaged samples and public records. Live Survey
requires a configured model service and may incur charges. Live results and their
limits are recorded in the [Survey Benchmark](../benchmarks/survey/README.md).
