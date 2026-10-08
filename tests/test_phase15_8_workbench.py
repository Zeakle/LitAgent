"""Verify Phase 15.8 workbench projections and safe report rendering."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from litagent.demo import project_flow_demo_view
from litagent.report_render import render_report_html


def test_report_renderer_supports_research_markdown_and_escapes_raw_html():
    rendered = render_report_html(
        "# Survey\n\n## Findings\n\n- Result [E:ev-1]\n\n"
        "| Metric | Value |\n| --- | --- |\n| Faithfulness | 0.91 |\n\n"
        "<script>alert('x')</script>"
    )

    assert '<h1 id="survey">Survey</h1>' in rendered.html
    assert '<h2 id="findings">Findings</h2>' in rendered.html
    assert "<table>" in rendered.html
    assert "<script>" not in rendered.html
    assert "&lt;script&gt;" in rendered.html
    assert [item.text for item in rendered.headings] == ["Survey", "Findings"]


def test_report_renderer_blocks_unsafe_link_protocols():
    rendered = render_report_html(
        "[safe](https://example.com) [unsafe](javascript:alert(1))"
    )

    assert 'href="https://example.com"' in rendered.html
    assert 'href="javascript:' not in rendered.html


def test_archive_projection_includes_safe_rendered_report():
    view = project_flow_demo_view(
        {
            "run_id": "run-1",
            "status": "completed",
            "report": {"survey": "# Result\n\nSupported [E:ev-1]."},
        },
        source="archive",
    )

    assert '<h1 id="result">Result</h1>' in view["rendered_report"]["html"]
    assert view["rendered_report"]["headings"][0]["anchor"] == "result"


def _run_module(module: str, expression: str) -> object:
    """Execute one browser-independent ES module contract with local Node."""
    if shutil.which("node") is None:
        pytest.skip("Node is unavailable for the ES module contract test")
    uri = (Path("src/litagent/static/flow-demo") / module).resolve().as_uri()
    script = (
        f"import * as subject from {json.dumps(uri)}; "
        f"console.log(JSON.stringify({expression}));"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_run_adapter_orders_sources_and_selects_default_section() -> None:
    result = _run_module(
        "adapters.js",
        "({runs: subject.normalizeRunList("
        "[{run_id: 'live-1', query: 'new', status: 'awaiting_human', "
        "updated_at: '2026-09-17T10:00:00Z'}], "
        "[{run_id: 'old-1', query: 'old', status: 'completed', "
        "completed_at: '2026-09-16T10:00:00Z', source: 'archive'}]"
        ").map((item) => item.key), "
        "review: subject.chooseDefaultSection({identity: {status: 'awaiting_human'}, "
        "review: {stage: 'paper_selection'}, report: null}), "
        "terminal: subject.chooseDefaultSection({identity: {status: 'completed'}, "
        "review: null, report: {survey: '# done'}})})",
    )

    assert result == {
        "runs": ["interactive:live-1", "archive:old-1"],
        "review": "review",
        "terminal": "report",
    }


def test_execution_graph_reports_missing_dependencies_and_cycles() -> None:
    result = _run_module(
        "execution-view.js",
        "subject.buildExecutionGraph("
        "{a: {task_id: 'a'}, b: {task_id: 'b'}}, "
        "{a: ['b'], b: ['a', 'missing']}, {})",
    )

    assert result["cyclic"] is True
    assert result["missingDependencies"] == [{"taskId": "b", "dependencyId": "missing"}]
    assert {node["id"] for node in result["nodes"]} == {"a", "b"}


def test_evidence_map_uses_exact_citation_marker_identity() -> None:
    result = _run_module(
        "evidence-view.js",
        "({raw: subject.evidenceMap([{evidence_id: 'arxiv:1:claim:a'}]).has("
        "'E:arxiv:1:claim:a'), prefixed: subject.evidenceMap("
        "[{evidence_id: 'E:demo-1'}]).has('E:demo-1')})",
    )

    assert result == {"raw": True, "prefixed": True}


def test_benchmark_run_heading_does_not_reuse_observation_count_suffix() -> None:
    labels = _run_module(
        "i18n.js",
        "({zhCount: subject.translator('zh-CN')('observations'), "
        "zhHeading: subject.translator('zh-CN')('benchmarkRuns'), "
        "enHeading: subject.translator('en')('benchmarkRuns')})",
    )
    benchmark_view = Path("src/litagent/static/flow-demo/benchmark-view.js").read_text(
        encoding="utf-8"
    )

    assert labels == {
        "zhCount": "次运行",
        "zhHeading": "运行记录",
        "enHeading": "Run records",
    }
    assert 'el("h4", t("benchmarkRuns"))' in benchmark_view


def test_benchmark_observation_renders_citation_audit_sidecars() -> None:
    labels = _run_module(
        "i18n.js",
        "({zh: subject.translator('zh-CN')('citationAuditExperiment'), "
        "en: subject.translator('en')('citationAuditExperiment'), "
        "stale: subject.translator('en')('staleDiagnostic')})",
    )
    benchmark_view = Path("src/litagent/static/flow-demo/benchmark-view.js").read_text(
        encoding="utf-8"
    )

    assert labels == {
        "zh": "引用核验试验",
        "en": "Citation audit experiment",
        "stale": "Stale diagnostic",
    }
    assert "payload.citation_audits || []" in benchmark_view
    assert "association_status" in benchmark_view
    assert "audit.decisions || []" in benchmark_view
    assert "audit.inputs || []" in benchmark_view


def test_workbench_shell_is_report_first_and_modular() -> None:
    root = Path("src/litagent/static/flow-demo")
    html = (root / "index.html").read_text(encoding="utf-8")
    app = (root / "app.js").read_text(encoding="utf-8")

    for filename in (
        "api-client.js",
        "state.js",
        "adapters.js",
        "history-view.js",
        "report-view.js",
        "execution-view.js",
        "review-view.js",
        "evidence-view.js",
        "quality-view.js",
        "i18n.js",
    ):
        assert (root / filename).is_file(), filename
    assert 'id="run-sidebar"' in html
    assert 'id="report-document"' in html
    assert 'id="inspector-drawer"' in html
    assert 'id="new-run-dialog"' in html
    assert re.search(r'from "\./report-view\.js(?:\?v=[\d.]+)?"', app)
    assert len(app.splitlines()) < 360
