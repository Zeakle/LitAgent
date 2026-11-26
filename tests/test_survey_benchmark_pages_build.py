"""Guard the files placed in the public Survey Benchmark Pages bundle."""

import json
import re
import shutil
from pathlib import Path

import pytest

from scripts.build_survey_benchmark_pages import PAGE_FILES, WORKBENCH_FILES, build

ROOT = Path(__file__).resolve().parents[1]
PAGES = Path("benchmarks/survey/pages")
WORKBENCH = Path("src/litagent/static/flow-demo")
RUNS = Path("benchmarks/survey/results/latency-faithfulness-95/execution-data")


@pytest.fixture
def checkout(tmp_path):
    """Copy only the Pages build inputs for negative tests."""
    root = tmp_path / "checkout"
    for directory in (PAGES, WORKBENCH, RUNS):
        shutil.copytree(ROOT / directory, root / directory)
    return root


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
    views = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "runs").glob("*.json")
        if path.name != "index.json"
    ]
    assert any(view["report"] is not None for view in views)
    assert any(view["step_io"] for view in views)
    for source in output.rglob("*.js"):
        for reference in re.findall(
            r'\bfrom\s+"(\./[^"?]+)', source.read_text(encoding="utf-8")
        ):
            assert (source.parent / reference).is_file(), (source, reference)


@pytest.mark.parametrize("invalid_source", ["unindexed_file", "linked_directory"])
def test_pages_build_rejects_unexpected_execution_inputs(
    checkout, tmp_path, invalid_source
):
    """Fail before publishing when unapproved data could escape the checkout."""
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


@pytest.mark.parametrize(
    ("invalid_data", "expected"),
    [
        ("report_hash", "Public report hash mismatch"),
        ("unknown_step", "Invalid public step IO"),
        ("empty_step", "Invalid public step payload"),
        ("scalar_payload", "Invalid public step payload"),
        ("duplicate_payload", "Duplicate member in public execution JSON"),
    ],
)
def test_pages_build_rejects_invalid_report_and_step_data(
    checkout, tmp_path, invalid_data, expected
):
    """Reject invalid or unreferenced details before the site is assembled."""
    path = checkout / RUNS / "89dc828fd629.json"
    view = json.loads(path.read_text(encoding="utf-8"))
    if invalid_data == "report_hash":
        view["report"]["markdown"] += "\nChanged source"
    elif invalid_data == "unknown_step":
        view["step_io"]["event:sequence:999999"] = {"status": "recorded", "input": 1}
    elif invalid_data in ("empty_step", "scalar_payload"):
        step = next(iter(view["graph"]["tasks"]))
        view["step_io"][step] = {"status": "recorded"}
        if invalid_data == "scalar_payload":
            view["step_io"][step]["input"] = "unprojected archive text"
    payload = json.dumps(view, ensure_ascii=False)
    if invalid_data == "duplicate_payload":
        payload = '{"step_io":{},' + payload[1:]
    path.write_text(payload, encoding="utf-8")

    output = tmp_path / "site"
    with pytest.raises(ValueError, match=expected):
        build(checkout, output)
    assert not output.exists()
