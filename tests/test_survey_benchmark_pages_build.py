"""Guard the files placed in the public Survey Benchmark Pages bundle."""

import re
import shutil
from pathlib import Path

import pytest

from scripts.build_survey_benchmark_pages import PAGE_FILES, WORKBENCH_FILES, build

ROOT = Path(__file__).resolve().parents[1]
PAGES = Path("benchmarks/survey/pages")
WORKBENCH = Path("src/litagent/static/flow-demo")
RUNS = Path("benchmarks/survey/results/latency-faithfulness-95/execution-data")


def test_pages_bundle_contains_only_public_assets(tmp_path):
    """Package all execution records and resolve the actual module imports."""
    output = tmp_path / "site"
    build(ROOT, output)

    assert {path.name for path in output.iterdir()} == {
        *PAGE_FILES,
        ".nojekyll",
        "runs",
        "workbench",
    }
    assert {path.name for path in (output / "workbench").iterdir()} == set(
        WORKBENCH_FILES
    )
    assert len(list((output / "runs").glob("*.json"))) == 30
    for source in output.rglob("*.js"):
        for reference in re.findall(
            r'\bfrom\s+"(\./[^"?]+)', source.read_text(encoding="utf-8")
        ):
            assert (source.parent / reference).is_file(), (source, reference)


@pytest.mark.parametrize("invalid_source", ["unindexed_file", "linked_directory"])
def test_pages_build_rejects_unexpected_execution_inputs(tmp_path, invalid_source):
    """Fail before publishing when unapproved data could escape the checkout."""
    checkout = tmp_path / "checkout"
    for directory in (PAGES, WORKBENCH, RUNS):
        shutil.copytree(ROOT / directory, checkout / directory)
    if invalid_source == "unindexed_file":
        (checkout / RUNS / "raw-run.json").write_text("{}", encoding="utf-8")
        expected = "unindexed files"
    else:
        outside = tmp_path / "outside"
        source = checkout / RUNS.parent
        source.rename(outside)
        try:
            source.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Symlinks are unavailable: {exc}")
        expected = "inside the repository"

    output = tmp_path / "site"
    with pytest.raises(ValueError, match=expected):
        build(checkout, output)
    assert not output.exists()
