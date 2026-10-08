"""Exercise durable automatic audit reservations with offline model replies."""

import asyncio
import copy
import hashlib
import html
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.config import AppConfig
from litagent.llm.client import LLMResponse
from litagent.report_audit import ReportAuditService


def source(revision=1):
    """Freeze a passed report whose sole fact is backed by selected evidence."""
    survey = "Claim A [E:ev-1]."
    return {
        "source_run_id": "automatic-report",
        "source_report_revision": revision,
        "source_report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
        "report": {
            "survey": survey,
            "partial": False,
            "quality": {"status": "passed"},
            "metadata": {"human_review": {"status": "pending"}},
            "delivery": {"status": "needs_review"},
        },
        "ledger": {"ev-1": {"paper_id": "paper-1", "text": "Claim A."}},
        "context": {"query": "Question", "report_language": "en"},
        "artifact": {"run_id": "automatic-report"},
    }


class FakeJudge:
    """Reply through the real Judge schema without reaching a network."""

    def __init__(self):
        """Track calls and evidence aliases selected by the host."""
        self.calls = 0
        self.aliases = []

    async def chat(self, messages, **kwargs):
        """Return one faithful scientific assertion and its assessment."""
        self.calls += 1
        if self.calls == 1:
            units = json.loads(
                html.unescape(
                    messages[1]["content"]
                    .split("<source_units>\n", 1)[1]
                    .split("\n</source_units>", 1)[0]
                )
            )
            unit = next(unit for unit in units if "Claim A" in unit["text"])
            self.aliases = unit["evidence_aliases"]
            payload = {
                "claims": [
                    {
                        "claim_text": "Claim A.",
                        "source_unit_id": unit["source_unit_id"],
                        "evidence_aliases": self.aliases,
                    }
                ],
                "method_claims": [],
            }
        else:
            payload = {
                "topic_assessments": [],
                "claim_assessments": [
                    {
                        "claim_id": "claim_0",
                        "source_faithful": True,
                        "classification_valid": True,
                        "supported": True,
                        "evidence_ids": self.aliases,
                        "explanation": "Supported by the supplied evidence.",
                    }
                ],
                "contradictions": [],
                "method_assessments": [],
            }
        return LLMResponse(
            content=json.dumps(payload),
            model="offline-fixture",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def close(self):
        """Satisfy the executor-owned transport cleanup contract."""


@pytest.fixture
def configured(monkeypatch):
    """Provide fake credential presence without reading the project's .env."""
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")
    monkeypatch.setenv("LLM_API_KEY", "offline-fixture-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return AppConfig(agent={}, logging={}, observability={"enabled": False})


async def completed(service, automatic):
    """Wait only for a task owned by the injected offline executor."""
    task = service._tasks.get(automatic["audit_id"])
    if task:
        await task
    return await service.get_automatic(
        automatic["source_run_id"], automatic["source_report_revision"]
    )


@pytest.mark.asyncio
async def test_off_creates_neither_index_nor_provider_task(tmp_path, configured):
    root = tmp_path / "off"
    factory = AsyncMock()
    service = ReportAuditService(configured, root, client_factory=factory)
    result = await service.submit_automatic(source(), mode="off", max_versions=1)
    assert result["status"] == "disabled"
    assert result["reason_codes"] == ["auto_audit_off"]
    assert not result["budget_reserved"] and result["audit_id"] is None
    assert not root.exists() and not service._tasks
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,budget,error",
    [
        ("automatic", 1, "automatic_mode_invalid"),
        ("judge", 0, "automatic_max_versions_invalid"),
        ("judge", 4, "automatic_max_versions_invalid"),
        ("judge", True, "automatic_max_versions_invalid"),
    ],
)
async def test_invalid_policy_is_rejected_before_reservation(
    tmp_path, configured, mode, budget, error
):
    root = tmp_path / "invalid"
    service = ReportAuditService(configured, root)
    with pytest.raises(ValueError, match=error):
        await service.submit_automatic(source(), mode=mode, max_versions=budget)
    assert not root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,reason",
    [
        ("quality_missing", "automatic_quality_not_passed"),
        ("quality_failed", "automatic_quality_not_passed"),
        ("partial", "automatic_report_partial"),
        ("rejected", "automatic_report_rejected"),
        ("cancelled", "automatic_report_cancelled"),
    ],
)
async def test_ineligible_reports_are_saved_as_skipped_without_spending_budget(
    tmp_path, configured, fault, reason
):
    snapshot = source()
    if fault == "quality_missing":
        snapshot["report"].pop("quality")
    elif fault == "quality_failed":
        snapshot["report"]["quality"]["status"] = "failed"
    elif fault == "partial":
        snapshot["report"]["partial"] = True
    else:
        snapshot["report"]["metadata"]["human_review"]["status"] = fault
    client = FakeJudge()
    service = ReportAuditService(configured, tmp_path, client_factory=lambda *_: client)
    skipped = await service.submit_automatic(snapshot, mode="judge", max_versions=1)
    assert skipped["status"] == "skipped"
    assert skipped["reason_codes"] == [reason] and not skipped["budget_reserved"]
    assert client.calls == 0 and not service._tasks
    next_version = await service.submit_automatic(
        source(2), mode="judge", max_versions=1
    )
    assert (await completed(service, next_version))["status"] == "completed"
    assert client.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,fault,reason",
    [
        ("judge", "key", "missing_key:LLM_API_KEY"),
        ("judge", "disabled", "audit_offline_or_disabled"),
        ("judge", "offline", "audit_offline_or_disabled"),
        ("judge_jev", "jev_key", "missing_key:TYPESAFE_API_KEY"),
        ("judge_jev", "jev_disabled", "jev_disabled"),
    ],
)
async def test_capability_gates_precede_paid_version_reservation(
    tmp_path, configured, monkeypatch, mode, fault, reason
):
    if fault == "key":
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    elif fault == "disabled":
        configured.report_audit.enabled = False
    elif fault == "offline":
        monkeypatch.setenv("LITAGENT_DEMO_MODE", "offline")
    elif fault == "jev_disabled":
        configured.report_audit.jev_enabled = False
    factory = AsyncMock()
    service = ReportAuditService(configured, tmp_path, client_factory=factory)
    result = await service.submit_automatic(source(), mode=mode, max_versions=1)
    assert result["status"] == "skipped" and result["reason_codes"] == [reason]
    assert not result["budget_reserved"] and result["audit_id"] is None
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_sha_conflicts_cannot_reuse_or_schedule_another_version(
    tmp_path, configured
):
    service = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    invalid = source()
    invalid["source_report_sha256"] = "0" * 64
    result = await service.submit_automatic(invalid, mode="judge", max_versions=1)
    assert result["status"] == "failed" and result["reason_codes"] == [
        "report_sha_mismatch"
    ]
    assert not result["budget_reserved"] and not service._tasks
    good = await service.submit_automatic(source(2), mode="judge", max_versions=1)
    await completed(service, good)
    changed = source(2)
    changed["report"]["survey"] += " Altered."
    with pytest.raises(ValueError, match="report_sha_mismatch"):
        await service.submit_automatic(changed, mode="judge", max_versions=1)
    changed["source_report_sha256"] = hashlib.sha256(
        changed["report"]["survey"].encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="report_sha_mismatch"):
        await service.submit_automatic(changed, mode="judge", max_versions=1)


@pytest.mark.asyncio
async def test_concurrent_executors_reserve_one_version_before_submit(
    tmp_path, configured
):
    entered, release = asyncio.Event(), asyncio.Event()
    first = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    second = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    await first.initialize()
    await second.initialize()
    original = first.submit

    async def blocked(snapshot, **kwargs):
        entered.set()
        await release.wait()
        return await original(snapshot, **kwargs)

    first.submit = AsyncMock(side_effect=blocked)
    second.submit = AsyncMock(wraps=second.submit)
    pending = asyncio.create_task(
        first.submit_automatic(source(), mode="judge", max_versions=1)
    )
    await entered.wait()
    duplicate = await second.submit_automatic(source(), mode="judge", max_versions=1)
    assert duplicate["status"] == "reserved" and duplicate["budget_reserved"]
    second.submit.assert_not_called()
    release.set()
    result = await pending
    result = await completed(first, result)
    assert result["status"] == "completed"
    first.submit.assert_awaited_once()
    with first._db() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM automatic_audits WHERE budget_reserved=1"
            ).fetchone()[0]
            == 1
        )
    assert (await second.get_automatic("automatic-report", 1))["audit_id"] == result[
        "audit_id"
    ]


@pytest.mark.asyncio
async def test_budget_counts_failed_attempts_and_allows_only_remaining_versions(
    tmp_path, configured
):
    service = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    original = service.submit
    service.submit = AsyncMock(side_effect=RuntimeError("private provider message"))
    failed = await service.submit_automatic(source(), mode="judge", max_versions=2)
    assert failed["status"] == "failed" and failed["budget_reserved"]
    assert failed["reason_codes"] == ["automatic_submit_failed:RuntimeError"]
    reused = await service.submit_automatic(source(), mode="judge", max_versions=2)
    assert reused == failed
    service.submit.assert_awaited_once()
    assert "private provider message" not in json.dumps(reused)
    service.submit = original
    second = await service.submit_automatic(source(2), mode="judge", max_versions=2)
    assert (await completed(service, second))["status"] == "completed"
    third = await service.submit_automatic(source(3), mode="judge", max_versions=2)
    assert third["status"] == "skipped"
    assert third["reason_codes"] == ["automatic_version_budget_exhausted"]
    assert not third["budget_reserved"] and third["audit_id"] is None


@pytest.mark.asyncio
async def test_competing_versions_cannot_exceed_a_run_wide_reservation(
    tmp_path, configured
):
    entered, release = asyncio.Event(), asyncio.Event()
    first = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    second = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    await first.initialize()
    await second.initialize()
    original = first.submit

    async def blocked(snapshot, **kwargs):
        entered.set()
        await release.wait()
        return await original(snapshot, **kwargs)

    first.submit = AsyncMock(side_effect=blocked)
    second.submit = AsyncMock(wraps=second.submit)
    pending = asyncio.create_task(
        first.submit_automatic(source(), mode="judge", max_versions=1)
    )
    await entered.wait()
    denied = await second.submit_automatic(source(2), mode="judge", max_versions=1)
    assert denied["status"] == "skipped"
    assert denied["reason_codes"] == ["automatic_version_budget_exhausted"]
    second.submit.assert_not_called()
    release.set()
    assert (await completed(first, await pending))["status"] == "completed"


@pytest.mark.asyncio
async def test_submission_cancellation_is_persisted_and_budget_not_refunded(
    tmp_path, configured
):
    entered = asyncio.Event()
    service = ReportAuditService(configured, tmp_path)

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    service.submit = AsyncMock(side_effect=blocked)
    task = asyncio.create_task(
        service.submit_automatic(source(), mode="judge", max_versions=1)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await service.get_automatic("automatic-report", 1)
    assert result["status"] == "cancelled" and result["budget_reserved"]
    assert result["reason_codes"] == ["automatic_submission_cancelled"]
    assert (
        await service.submit_automatic(source(), mode="judge", max_versions=1) == result
    )
    service.submit.assert_awaited_once()
    next_version = await service.submit_automatic(
        source(2), mode="judge", max_versions=1
    )
    assert next_version["reason_codes"] == ["automatic_version_budget_exhausted"]


@pytest.mark.asyncio
async def test_restart_does_not_reissue_a_reserved_call_or_refund_budget(
    tmp_path, configured
):
    service = ReportAuditService(configured, tmp_path)
    service.submit = AsyncMock(side_effect=RuntimeError("crash fixture"))
    await service.submit_automatic(source(), mode="judge", max_versions=1)
    with service._db() as db:
        db.execute("UPDATE automatic_audits SET status='reserved', reason_codes='[]'")
    restarted = ReportAuditService(configured, tmp_path)
    restarted.submit = AsyncMock()
    recovered = await restarted.get_automatic("automatic-report", 1)
    assert recovered["status"] == "interrupted" and recovered["budget_reserved"]
    assert recovered["reason_codes"] == ["automatic_reservation_interrupted_no_retry"]
    assert (
        await restarted.submit_automatic(source(), mode="judge", max_versions=1)
        == recovered
    )
    assert (await restarted.submit_automatic(source(2), mode="judge", max_versions=1))[
        "status"
    ] == "skipped"
    restarted.submit.assert_not_called()


@pytest.mark.asyncio
async def test_restart_does_not_reissue_an_already_linked_provider_call(
    tmp_path, configured
):
    service = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    automatic = await completed(
        service,
        await service.submit_automatic(source(), mode="judge", max_versions=1),
    )
    result = await service.get("automatic-report", 1, automatic["audit_id"])
    result["status"] = "running"
    service._save(result)
    factory = AsyncMock()
    restarted = ReportAuditService(configured, tmp_path, client_factory=factory)
    recovered = await restarted.get_automatic("automatic-report", 1)
    assert recovered["status"] == "interrupted"
    assert recovered["reason_codes"] == ["process_interrupted_no_automatic_retry"]
    assert recovered["budget_reserved"]
    assert (
        await restarted.submit_automatic(source(), mode="judge", max_versions=1)
        == recovered
    )
    factory.assert_not_called()
    assert not restarted._tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "mismatch"])
async def test_automatic_result_binding_never_falls_back_to_other_versions(
    tmp_path, configured, fault
):
    service = ReportAuditService(
        configured, tmp_path, client_factory=lambda *_: FakeJudge()
    )
    automatic = await completed(
        service,
        await service.submit_automatic(source(), mode="judge", max_versions=1),
    )
    if fault == "missing":
        service._path(automatic["audit_id"]).unlink()
    else:
        result = await service.get("automatic-report", 1, automatic["audit_id"])
        result["source_report_revision"] = 2
        service._save(result)
    result = await service.get_automatic("automatic-report", 1)
    assert result["status"] == "failed" and result["budget_reserved"]
    assert result["reason_codes"] == [f"automatic_bound_audit_{fault}"]
    assert await service.get_automatic("automatic-report", 2) is None
    assert not service._tasks


@pytest.mark.asyncio
async def test_task_cancellation_and_list_project_the_exact_bound_version(
    tmp_path, configured
):
    entered = asyncio.Event()

    class WaitingJudge(FakeJudge):
        """Hold an emitted provider call until audit cancellation."""

        async def chat(self, messages, **kwargs):
            """Represent one in-flight paid request with unknown usage."""
            self.calls += 1
            entered.set()
            await asyncio.Future()

    client = WaitingJudge()
    service = ReportAuditService(configured, tmp_path, client_factory=lambda *_: client)
    result = await service.submit_automatic(source(), mode="judge", max_versions=1)
    await entered.wait()
    assert (await service.list("automatic-report", 1))["automatic"][
        "status"
    ] == "running"
    await service.cancel("automatic-report", 1, result["audit_id"])
    cancelled = (await service.list("automatic-report", 1))["automatic"]
    assert cancelled["status"] == "cancelled" and cancelled["budget_reserved"]
    assert (await service.list("automatic-report", 2))["automatic"] is None
    assert client.calls == 1
    assert (await service.submit_automatic(source(2), mode="judge", max_versions=1))[
        "status"
    ] == "skipped"


@pytest.mark.asyncio
async def test_config_changes_never_retrigger_existing_automatic_version(
    tmp_path, configured
):
    client = FakeJudge()
    service = ReportAuditService(configured, tmp_path, client_factory=lambda *_: client)
    source_copy = source()
    original = copy.deepcopy(source_copy)
    result = await completed(
        service,
        await service.submit_automatic(source_copy, mode="judge", max_versions=1),
    )
    assert source_copy == original and client.calls == 2
    configured.report_audit.judge_model = "different-offline-model"
    service.submit = AsyncMock()
    reused = await service.submit_automatic(
        source_copy, mode="judge_jev", max_versions=3
    )
    assert reused["audit_id"] == result["audit_id"]
    assert reused["mode"] == "judge" and reused["max_versions"] == 1
    service.submit.assert_not_called()


@pytest.mark.asyncio
async def test_judge_jev_uses_frozen_claims_and_keeps_report_unchanged(
    tmp_path, configured, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fixture-key")
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        answers = {}
        for identity, question in payload["questions"].items():
            choices = list(question["criteria"])
            answers[identity] = {
                "type": "choice",
                "choice": choices[0],
                "confidence": 0.97,
                "probabilities": {
                    choice: float(choice == choices[0]) for choice in choices
                },
            }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        )

    snapshot = source()
    original = copy.deepcopy(snapshot)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        service = ReportAuditService(
            configured,
            tmp_path,
            client_factory=lambda *_: FakeJudge(),
            jev_factory=lambda *_: JevCitationAuditor(client=transport),
        )
        automatic = await completed(
            service,
            await service.submit_automatic(snapshot, mode="judge_jev", max_versions=1),
        )
        audit = await service.get("automatic-report", 1, automatic["audit_id"])
    assert automatic["status"] == "completed" and automatic["budget_reserved"]
    assert len(requests) == 1 and audit["jev"]["status"] == "completed"
    assert audit["jev"]["source_report_sha256"] == snapshot["source_report_sha256"]
    assert (
        audit["judge"]["inventory_snapshot"]["inventory_sha256"]
        == audit["jev"]["inventory_sha256"]
    )
    assert snapshot == original
