"""Verify terminal rejection accounting using actual SDK, journal and recorder output."""

import asyncio
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from model_config_helpers import load_previous_default_config as load_config
from test_native_responses import owner_for, session_for
from test_writer_v2 import case  # noqa: F401

from litagent.contracts import build_config_summary
from litagent.llm import responses as provider
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.report_audit import atomic_json, digest
from litagent.safety import budget as budget_module
from litagent.writing_v2 import schema_for
from scripts import run_frozen_writing as campaign


@pytest_asyncio.fixture
async def rejected_parent(case, tmp_path, monkeypatch):
    """Reproduce v1's real unsupported field at HTTP; all host consumers remain real."""
    monkeypatch.chdir(tmp_path)
    original_builder = provider.build_responses_request

    def legacy_builder(*args, **kwargs):
        value = original_builder(*args, **kwargs)
        value.update(background=False, truncation="disabled")
        return value

    monkeypatch.setattr(provider, "build_responses_request", legacy_builder)
    monkeypatch.setattr(provider, "RESPONSES_CONTRACT", "ark-responses-mapping-v1")
    owner, budget, _ = await owner_for(
        lambda request: pytest.fail("vendor rejects before inference")
    )
    root = tmp_path / "parent"
    root.mkdir()
    run_id = root.name + "-few-shot-metric"
    recorder = RunRecorder(run_id, case.packet["query"], ArchiveRepository())
    owner._trace_hook = recorder
    session = session_for(owner, case, tmp_path)
    # Keep the writer's durable path aligned with the actual campaign consumer.
    session.journal_path = Path(
        "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
    )
    context = set_task_id("synthesis")
    try:
        with pytest.raises(ValueError, match="strict_usage_incomplete"):
            await session.call(
                "body:introduction",
                "Full source contract",
                case.packet,
                schema_for("body", case.aliases),
            )
    finally:
        reset_task_id(context)
        await owner.close()
    config = load_config()
    config.llm.base_url = "https://offline.invalid/api/v3"
    config.llm.generation_model = "pro"
    config.llm.evaluation_model = "lite"
    config.adversarial.thinking_mode = "enabled"
    config.llm.max_tokens = 20000
    summary = build_config_summary(config, "strict_auto_v2")
    recorder.set_config_summary(summary)
    artifact = recorder.finalize(report=None, error="usage_unknown")
    source = Path(budget_module.__file__)
    budget_name = "src/litagent/safety/budget.py"
    saved = root / "source" / budget_name
    saved.parent.mkdir(parents=True)
    saved.write_bytes(source.read_bytes())
    history = {
        "model_requests": 752,
        "model_known_tokens": 4238381,
        "model_unknown_conservative_tokens": 1030414,
    }
    manifest = {
        "version": campaign.VERSION,
        "history": history,
        "config": summary,
        "limits": dict(campaign.LIMITS),
        "source_hashes": {budget_name: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    manifest["identity"] = digest(manifest)
    row = {
        "run_id": run_id,
        "requests": 1,
        "known_tokens": 0,
        "unknown_hold": budget.used,
        "usage_complete": False,
        "publishable": False,
    }
    state = {"identity": manifest["identity"], "status": "stopped", "cases": [row]}
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "state.json", state)
    return SimpleNamespace(
        root=root,
        history=history,
        manifest=manifest,
        state=state,
        row=row,
        artifact=artifact,
        journal=session.journal_path,
        archive=Path("artifacts/runs", run_id + ".json"),
        snapshot=budget.snapshot(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [False, True])
async def test_unknown_rejection_is_carried_without_a_refund_or_fabricated_snapshot(
    rejected_parent, persisted
):
    """Only a named new policy can carry proven uncertain charges; the old default refuses."""
    p = rejected_parent
    if persisted:
        p.row["budget_snapshot"] = p.snapshot
        atomic_json(p.root / "state.json", p.state)
    before = {
        f: f.read_bytes()
        for f in (p.root / "manifest.json", p.root / "state.json", p.archive, p.journal)
    }
    with pytest.raises(ValueError, match="usage_unknown"):
        campaign.accounting_history(p.history, [p.root])
    result, receipts = campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    )
    assert result == {
        **p.history,
        "model_requests": 753,
        "model_unknown_conservative_tokens": 1030414 + p.row["unknown_hold"],
    }
    proof = receipts[0]["conservative_rejections"][0]
    assert proof["unknown_hold"] == p.row["unknown_hold"]
    assert proof["budget_evidence"] == (
        "persisted_snapshot" if persisted else "reconstructed_from_frozen_records"
    )
    assert receipts[0]["known_tokens"] == 0 and receipts[0]["unknown_hold"] > 0
    assert all(f.read_bytes() == value for f, value in before.items())
    assert campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    ) == (result, receipts)
    with pytest.raises(ValueError, match="duplicate"):
        campaign.accounting_history(
            p.history, [p.root, p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [False, True])
async def test_legacy_budget_source_revalidates_without_using_current_code(
    rejected_parent, persisted
):
    """Carry old uncertain charges across the planned-budget module upgrade."""
    p = rejected_parent
    expected, _ = campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    )
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/legacy-budget-pre-planned-v2.json"
        ).read_text()
    )
    original = budget_module.CostBudget
    saved = p.root / "source/src/litagent/safety/budget.py"
    saved.write_bytes(fixture["source"].encode())
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == fixture["source_sha256"]
    p.manifest["source_hashes"]["src/litagent/safety/budget.py"] = fixture[
        "source_sha256"
    ]
    p.manifest["identity"] = digest(
        {k: v for k, v in p.manifest.items() if k != "identity"}
    )
    p.state["identity"] = p.manifest["identity"]
    if persisted:
        p.row["budget_snapshot"] = p.snapshot
    atomic_json(p.root / "manifest.json", p.manifest)
    atomic_json(p.root / "state.json", p.state)
    paths = [
        p.root / "manifest.json",
        p.root / "state.json",
        p.archive,
        p.journal,
        saved,
    ]
    before = {f: f.read_bytes() for f in paths}
    result, receipts = campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    )
    assert result == expected
    proof = receipts[0]["conservative_rejections"][0]
    assert proof["budget_source_sha256"] == fixture["source_sha256"]
    assert proof["unknown_hold"] == p.row["unknown_hold"]
    assert campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    ) == (result, receipts)
    assert all(f.read_bytes() == content for f, content in before.items())
    assert budget_module.CostBudget is original


@pytest.mark.parametrize("rehashed", [False, True])
def test_unknown_historical_source_is_never_executed(tmp_path, rehashed):
    saved = tmp_path / "source/src/litagent/safety/budget.py"
    saved.parent.mkdir(parents=True)
    marker = tmp_path / "unexpected-execution"
    saved.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    sha = hashlib.sha256(saved.read_bytes()).hexdigest() if rehashed else "wrong"
    manifest = {"source_hashes": {"src/litagent/safety/budget.py": sha}}
    with pytest.raises(ValueError, match="frozen_accounting_"):
        campaign.historical_accounting_budget(tmp_path, manifest)
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "short_hold",
        "large_hold",
        "known",
        "complete",
        "running",
        "input",
        "schema",
        "input_sha",
        "receipt_bytes",
        "receipt_hash",
        "missing_receipt",
        "source",
        "snapshot",
        "operation",
        "retry",
        "model",
        "endpoint",
        "output",
        "non_rejection",
        "count",
        "origin",
        "config",
    ],
)
async def test_conservative_receipt_rejects_inconsistent_or_incomplete_evidence(
    rejected_parent, fault
):
    """Rehashing individual files cannot bypass their independent producer bindings."""
    p = rejected_parent
    journal = json.loads(p.journal.read_text())
    call = journal["calls"]["body:introduction"]
    receipt = call["provider_receipt"]
    artifact = json.loads(p.archive.read_text())
    start, end = [
        e["data"] for e in artifact["events"] if e["event"].startswith("llm.")
    ]
    if fault == "short_hold":
        p.row["unknown_hold"] -= 1
    if fault == "large_hold":
        p.row["unknown_hold"] += 1
    if fault == "known":
        p.row["known_tokens"] = 1
    if fault == "complete":
        p.row["usage_complete"] = True
    if fault == "running":
        p.state["status"] = "running"
    if fault == "count":
        p.row["requests"] = 0
    if fault == "input_sha":
        call["input_sha256"] = "wrong"
    if fault == "receipt_bytes":
        receipt["request_body_base64"] = "!"
    if fault == "receipt_hash":
        receipt["request_sha256"] = "wrong"
    if fault == "missing_receipt":
        del call["provider_receipt"]
    if fault in {"input", "schema", "model"}:
        body = json.loads(base64.b64decode(receipt["request_body_base64"]))
        if fault == "input":
            body["input"][1]["content"] = "replaced sources"
        if fault == "schema":
            body["text"]["format"]["strict"] = False
        if fault == "model":
            body["model"] = "wrong"
        raw = json.dumps(body).encode()
        receipt["request_body_base64"] = base64.b64encode(raw).decode()
        receipt["request_sha256"] = hashlib.sha256(raw).hexdigest()
    if fault == "source":
        (p.root / "source/src/litagent/safety/budget.py").write_text("changed formula")
    if fault == "snapshot":
        p.row["budget_snapshot"] = {**p.snapshot, "used_tokens": 0}
    if fault == "operation":
        end["operation_id"] = "different"
    if fault == "retry":
        artifact["events"].append({"event": "llm.retry", "data": {}})
    if fault == "endpoint":
        receipt["endpoint"] = "/other/responses"
    if fault == "origin":
        receipt["origin"] = "https://other.invalid"
    if fault == "config":
        artifact["config"] = {}
    if fault == "output":
        body = {"error": {}, "output": [{"type": "message"}]}
        raw = json.dumps(body).encode()
        receipt["response_body_base64"] = base64.b64encode(raw).decode()
        receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
    if fault == "non_rejection":
        receipt["status_code"] = 500
        end["provider_error"] = "responses_http_500"
    atomic_json(p.root / "state.json", p.state)
    atomic_json(p.journal, journal)
    atomic_json(p.archive, artifact)
    with pytest.raises(ValueError):
        campaign.accounting_history(
            p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
        )


@pytest.mark.asyncio
async def test_http_root_cause_is_safe_and_version_bound_in_all_exports(
    rejected_parent, monkeypatch
):
    """The financial stop remains visible beside the technical cause, without raw prose."""
    p = rejected_parent
    diagnostic = campaign.transport_diagnostic(p.row["run_id"])
    assert diagnostic["http_status"] == 400
    assert diagnostic["provider_error_code"] == "InvalidParameter"
    assert diagnostic["rejected_parameter"] == "background"
    assert "message" not in diagnostic
    manifest = {
        **p.manifest,
        "source_commit": "offline",
        "writing_profile": campaign.LEGACY_RESPONSES_PROFILE,
    }
    p.row.update(
        case_id="few-shot-metric",
        status="blocked",
        generated=False,
        reviewer_completed=False,
        evaluation="not_complete",
        failure_code="usage_unknown",
    )
    campaign.export(p.root, p.state, manifest)
    out = Path("benchmarks/survey/results") / p.root.name
    data = json.loads((out / "benchmark.json").read_text())
    assert data["cases"][0]["http_status"] == 400
    assert data["cases"][0]["failure_code"] == "usage_unknown"
    assert data["planned"] == 3 and data["passed"] == 0 and data["not_run"] == 2
    assert all(
        s in (out / "benchmark.csv").read_text()
        for s in ("InvalidParameter", "background", "400")
    )
    assert "HTTP 400" in (out / "benchmark.md").read_text()
    page = (out / "benchmark.html").read_text()
    embedded = page.split('<script id="data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == data
    assert "request_body_base64" not in page and "response_body_base64" not in page
    # Corrupting this version must remove its projection, never borrow another run.
    atomic_json(p.archive, {**p.artifact, "run_id": "other"})
    assert campaign.transport_diagnostic(p.row["run_id"]) is None


@pytest.mark.asyncio
async def test_known_receipt_after_uncertain_prefix_keeps_the_entire_hold(
    rejected_parent,
):
    """A later known bill cannot overwrite or refund an earlier conservative charge."""
    from test_writer_v2_campaign import accounting_parent

    p = rejected_parent
    after, _ = campaign.accounting_history(
        p.history, [p.root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    )
    second = accounting_parent(p.root.parent, "known-followup", after, 30)
    final, receipts = campaign.accounting_history(
        p.history, [p.root, second], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
    )
    assert final["model_requests"] == 754
    assert final["model_known_tokens"] == 4238411
    assert final["model_unknown_conservative_tokens"] == 1030414 + p.row["unknown_hold"]
    assert "unknown_hold" not in receipts[1]


@pytest.mark.asyncio
async def test_public_diagnostic_does_not_expose_hostile_error_prose(rejected_parent):
    """Provider errors remain untrusted data even when their receipt is intact."""
    p = rejected_parent
    journal = json.loads(p.journal.read_text())
    receipt = journal["calls"]["body:introduction"]["provider_receipt"]
    raw = json.dumps(
        {
            "error": {
                "code": "<script>PRIVATE</script>",
                "message": "PRIVATE secret text and scientific content",
                "param": "</script>",
            }
        }
    ).encode()
    receipt["response_body_base64"] = base64.b64encode(raw).decode()
    receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
    atomic_json(p.journal, journal)
    value = campaign.transport_diagnostic(p.row["run_id"])
    assert value["http_status"] == 400
    assert value["provider_error_code"] is value["rejected_parameter"] is None
    assert "PRIVATE" not in json.dumps(value)
