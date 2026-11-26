"""Package the public Survey execution viewer for static GitHub Pages hosting."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

PAGE_FILES = ("index.html", "app.js", "styles.css")
WORKBENCH_FILES = (
    "evidence-model.js",
    "execution-content.js",
    "execution-flow.js",
    "execution-inspector.js",
    "execution-model.js",
    "execution-overview.js",
    "execution-timeline.js",
    "execution-view.js",
    "execution-workspace.js",
    "i18n.js",
    "report-view.js",
    "state.js",
    "styles.css",
)
RUN_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RUN_COUNT = 29
RUN_INDEX_FIELDS = {
    "run_id",
    "case_id",
    "profile_id",
    "repetition",
    "status",
    "delivery",
}
RUN_VIEW_FIELDS = {
    "identity",
    "summary",
    "graph",
    "events",
    "delivery",
    "quality",
    "metrics",
    "report",
    "step_io",
}
REPORT_FIELDS = {"markdown", "source_sha256", "state", "html"}
STEP_SIDES = {"input", "output"}
STEP_FIELDS = STEP_SIDES | {"status", "omitted"}


def _regular_file(path: Path) -> Path:
    """Require an ordinary source file without copying symlinks into the site."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular source file at {path}")
    return path


def _read_json(path: Path) -> object:
    """Read a single expected public data file."""
    return json.loads(
        _regular_file(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_members,
        parse_constant=_invalid_json_number,
    )


def _unique_members(members: list[tuple[str, object]]) -> dict[str, object]:
    """Prevent duplicate keys from hiding unvalidated data in the published file."""
    result = dict(members)
    if len(result) != len(members):
        raise ValueError("Duplicate member in public execution JSON")
    return result


def _invalid_json_number(number: str) -> None:
    """Reject Python JSON extensions that cannot be read by the browser."""
    raise ValueError(f"Invalid number in public execution JSON: {number}")


def _validate_report(report: object, run_id: str) -> None:
    """Check the report fields and verify its exact archived Markdown hash."""
    if report is None:
        return
    if (
        not isinstance(report, dict)
        or report.keys() != REPORT_FIELDS
        or report["state"] not in ("complete", "partial")
        or not isinstance(report["markdown"], str)
        or not report["markdown"].strip()
        or not isinstance(report["html"], str)
        or not report["html"].strip()
    ):
        raise ValueError(f"Invalid public report for {run_id}")
    if (
        hashlib.sha256(report["markdown"].encode("utf-8")).hexdigest()
        != report["source_sha256"]
    ):
        raise ValueError(f"Public report hash mismatch for {run_id}")


def _validate_step_io(view: dict, run_id: str) -> None:
    """Require every detail record to resolve to a visible task, node, or event."""
    graph, events, steps = view["graph"], view["events"], view["step_io"]
    if (
        not isinstance(graph, dict)
        or not isinstance(graph.get("tasks"), dict)
        or not isinstance(graph.get("nodes"), dict)
        or not isinstance(events, list)
        or not isinstance(steps, dict)
    ):
        raise ValueError(f"Invalid public step IO for {run_id}")
    sequences = [
        event.get("sequence") if isinstance(event, dict) else None for event in events
    ]
    if any(type(sequence) is not int or sequence < 1 for sequence in sequences):
        raise ValueError(f"Invalid public event sequence for {run_id}")
    event_ids = {f"event:sequence:{sequence}" for sequence in sequences}
    if len(event_ids) != len(events):
        raise ValueError(f"Duplicate public event sequence for {run_id}")
    references = graph["tasks"].keys() | graph["nodes"].keys() | event_ids
    for key, detail in steps.items():
        if (
            key not in references
            or not isinstance(detail, dict)
            or not detail.keys() <= STEP_FIELDS
            or detail.get("status") not in ("recorded", "reused", "planned")
        ):
            raise ValueError(f"Invalid public step IO for {run_id}")
        omitted = detail.get("omitted", [])
        recorded = STEP_SIDES & detail.keys()
        if (
            not isinstance(omitted, list)
            or any(side not in ("input", "output") for side in omitted)
            or len(omitted) != len(set(omitted))
            or not (recorded or omitted)
            or any(not isinstance(detail[side], (dict, list)) for side in recorded)
        ):
            raise ValueError(f"Invalid public step payload for {run_id}")


def _run_files(directory: Path) -> list[Path]:
    """Check that the public index names each archived run exactly once."""
    index = directory / "index.json"
    payload = _read_json(index)
    if not isinstance(payload, dict) or payload.keys() != {"runs"}:
        raise ValueError("The execution index must contain a runs list")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != RUN_COUNT:
        raise ValueError(f"Expected exactly {RUN_COUNT} indexed runs")

    files = {index}
    for row in runs:
        if not isinstance(row, dict) or row.keys() != RUN_INDEX_FIELDS:
            raise ValueError("Invalid field set in the execution index")
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run ID in the execution index")
        path = directory / f"{run_id}.json"
        if path in files:
            raise ValueError(f"Duplicate execution data file for {run_id}")
        view = _read_json(path)
        if (
            not isinstance(view, dict)
            or view.keys() != RUN_VIEW_FIELDS
            or not isinstance(view["identity"], dict)
            or view["identity"].get("runId") != run_id
        ):
            raise ValueError(f"Invalid public execution view for {run_id}")
        _validate_report(view["report"], run_id)
        _validate_step_io(view, run_id)
        files.add(path)
    if set(directory.iterdir()) != files:
        raise ValueError("Execution data contains unindexed files")
    return sorted(files)


def build(root: Path, output: Path) -> None:
    """Assemble an exact static bundle and publish it to a new output directory."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output directory already exists at {output}")
    root = root.resolve()
    pages = root / "benchmarks/survey/pages"
    workbench = root / "src/litagent/static/flow-demo"
    runs = root / "benchmarks/survey/results/latency-faithfulness-95/execution-data"
    for directory in (pages, workbench, runs):
        if not directory.is_dir() or directory.resolve() != directory:
            raise ValueError(
                f"Expected a regular source directory inside the repository at {directory}"
            )
    file_groups = {
        "": [_regular_file(pages / name) for name in PAGE_FILES],
        "workbench": [_regular_file(workbench / name) for name in WORKBENCH_FILES],
        "runs": _run_files(runs),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        site = Path(temporary) / "site"
        for subdirectory, files in file_groups.items():
            destination = site / subdirectory
            destination.mkdir(parents=True)
            for path in files:
                shutil.copyfile(path, destination / path.name)
        (site / ".nojekyll").touch()
        site.rename(output)


def main() -> None:
    """Build the Pages bundle using only the Python standard library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output)


if __name__ == "__main__":
    main()
