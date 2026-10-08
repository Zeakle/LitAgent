"""Exercise version-bound report audits entirely with offline transports."""

import asyncio
import copy
import hashlib
import html
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from litagent.audit_context import SurveyAuditContext, check_structure
from litagent.benchmark.citation_audit import (
    JevCitationAuditor,
    derive_citation_audit_summary,
)
from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
from litagent.benchmark.survey_models import SurveyJudgeConfig
from litagent.config import AppConfig
from litagent.llm.client import LLMResponse
from litagent.report_audit import ReportAuditService, atomic_json
from litagent.report_render import build_report_source_units


def config():
    """Disable tracing transport while preserving production capability checks."""
    return AppConfig(agent={}, logging={}, observability={"enabled": False})


def snapshot(survey="Claim A [E:ev-1].", revision=1, methods=None, structure="legacy"):
    """Construct an immutable report and a host-owned evidence ledger."""
    return {
        "source_run_id": "report-1",
        "source_report_revision": revision,
        "source_report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
        "report": {
            "survey": survey,
            "quality": {"status": "passed"},
            "metadata": {"human_review": {"status": "approved"}},
        },
        "ledger": {"ev-1": {"paper_id": "paper-1", "text": "Claim A."}},
        "context": {
            "query": "Question",
            "report_language": "en",
            "review_structure": structure,
            "review_methodology": methods or {},
        },
        "artifact": {"run_id": "report-1"},
    }


class OfflineJudge:
    """Reply through the real two-stage validator without any network access."""

    def __init__(
        self,
        *,
        method_status="verified",
        faithful=True,
        invalid_path=False,
        missing_usage=False,
        invalid_classification=False,
    ):
        """Configure deliberate protocol faults and method judgments."""
        self.calls = 0
        self.closed = False
        self.method_status = method_status
        self.faithful = faithful
        self.invalid_path = invalid_path
        self.missing_usage = missing_usage
        self.invalid_classification = invalid_classification
        self.units = []
        self.methods = []

    async def chat(self, messages, **kwargs):
        """Extract supplied unit identities rather than manufacturing source offsets."""
        self.calls += 1
        if self.calls == 1:
            content = messages[1]["content"]
            payload = json.loads(
                html.unescape(
                    content.split("<source_units>\n", 1)[1].split(
                        "\n</source_units>", 1
                    )[0]
                )
            )
            self.units = [unit for unit in payload if "Claim A" in unit["text"]]
            self.methods = [
                unit for unit in payload if "abstract evidence" in unit["text"]
            ]

            def item(unit):
                return {
                    "claim_text": (
                        "Claim A."
                        if "Claim A" in unit["text"]
                        else "This run used abstract evidence."
                    ),
                    "source_unit_id": unit["source_unit_id"],
                    "evidence_aliases": unit["evidence_aliases"],
                }

            result = {
                "claims": [item(unit) for unit in self.units[:1]],
                "method_claims": [item(unit) for unit in self.methods[:1]],
            }
        else:
            result = {
                "topic_assessments": [],
                "claim_assessments": (
                    [
                        {
                            "claim_id": "claim_0",
                            "classification_valid": True,
                            "source_faithful": self.faithful,
                            "supported": True,
                            "evidence_ids": self.units[0]["evidence_aliases"],
                            "explanation": "Supported.",
                        }
                    ]
                    if self.units
                    else []
                ),
                "contradictions": [],
                "method_assessments": (
                    [
                        {
                            "claim_id": "method_0",
                            "source_faithful": True,
                            "classification_valid": not self.invalid_classification,
                            "status": self.method_status,
                            "record_paths": (
                                ["invented" if self.invalid_path else "content_mode"]
                                if self.method_status != "unverifiable"
                                else []
                            ),
                            "explanation": "Compared with actual run records.",
                        }
                    ]
                    if self.methods
                    else []
                ),
            }
        return LLMResponse(
            content=json.dumps(result),
            model="offline",
            usage=(
                {}
                if self.missing_usage
                else {"prompt_tokens": 10, "completion_tokens": 5}
            ),
        )

    async def close(self):
        """Record client ownership cleanup."""
        self.closed = True


async def wait_result(service, result):
    """Await an owned offline task without polling delays."""
    task = service._tasks.get(result["audit_id"])
    if task:
        await task
    return await service.get(
        result["source_run_id"], result["source_report_revision"], result["audit_id"]
    )


@pytest.mark.asyncio
async def test_ordinary_judge_separates_methods_and_preserves_source_anchor():
    survey = "# Review\n\n## Search and Evidence Selection Methods\nThis run used abstract evidence.\n\nClaim A [E:ev-1]."
    state = {}
    client = OfflineJudge()
    result, usage = await SurveyBenchmarkJudge(
        client, SurveyJudgeConfig(base_url="https://offline.example", model="offline")
    ).evaluate(
        case=SurveyAuditContext(
            query="Question",
            review_structure="method_transparent_v1",
            review_methodology={"content_mode": "abstract"},
        ),
        survey=survey,
        ledger=snapshot()["ledger"],
        checkpoint=state,
    )
    assert result.status == "completed"
    assert result.topic_coverage is None
    assert result.factual_claim_count == 1
    assert len(state["method_assessments"]) == 1
    assert state["method_assessments"][0]["status"] == "verified"
    claim = result.inventory_snapshot.claims[0]
    assert survey[claim.source_start : claim.source_end] == claim.source_quote
    assert "Claim A" in claim.source_quote
    assert usage.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("missing", "completed"),
        ("false_method", "completed"),
        ("invalid_path", "failed"),
        ("classification", "failed"),
    ],
)
async def test_method_record_missing_fabricated_or_invalid(mode, expected):
    client = OfflineJudge(
        method_status=(
            "unverifiable"
            if mode == "missing"
            else "unsupported" if mode == "false_method" else "verified"
        ),
        invalid_path=mode == "invalid_path",
        invalid_classification=mode == "classification",
    )
    state = {}
    result, _ = await SurveyBenchmarkJudge(
        client, SurveyJudgeConfig(base_url="https://offline.example", model="offline")
    ).evaluate(
        case=SurveyAuditContext(
            query="Question",
            review_methodology=(
                {} if mode == "missing" else {"content_mode": "full_text"}
            ),
        ),
        survey="This run used abstract evidence.\n\nClaim A [E:ev-1].",
        ledger=snapshot()["ledger"],
        checkpoint=state,
    )
    assert result.status == expected
    if mode == "missing":
        assert state["method_assessments"][0]["status"] == "unverifiable"
    if mode == "false_method":
        assert state["method_assessments"][0]["status"] == "unsupported"
        assert result.faithfulness == 1


@pytest.mark.parametrize(
    "structure,language",
    [
        ("academic_v1", "en"),
        ("academic_v1", "zh-CN"),
        ("method_transparent_v1", "en"),
        ("method_transparent_v1", "zh-CN"),
    ],
)
def test_structure_order_missing_duplicate_empty(structure, language):
    context = SurveyAuditContext(
        query="q", review_structure=structure, report_language=language
    )
    titles = check_structure("", context)["expected_sections"]
    report = "# Review\n\n" + "\n\n".join(f"## {title}\nBody" for title in titles)
    assert check_structure(report, context)["status"] == "passed"
    assert (
        check_structure(report.replace("\nBody", "", 1), context)["status"] == "failed"
    )
    assert (
        check_structure(report + f"\n## {titles[0]}\nRepeated", context)["status"]
        == "failed"
    )
    assert (
        check_structure(
            "# Review\n" + "\n".join(f"## {title}\nBody" for title in reversed(titles)),
            context,
        )["status"]
        == "failed"
    )
    assert (
        check_structure(report.replace(f"## {titles[0]}", "## Other", 1), context)[
            "status"
        ]
        == "failed"
    )
    assert (
        check_structure(report, SurveyAuditContext(query="q"))["status"]
        == "not_applicable"
    )


@pytest.mark.asyncio
async def test_audit_dedup_version_binding_and_immutability(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    client = OfflineJudge()
    service = ReportAuditService(config(), tmp_path, client_factory=lambda *_: client)
    original = snapshot()
    before = copy.deepcopy(original)
    first, second = await asyncio.gather(
        service.submit(original), service.submit(original)
    )
    assert first["audit_id"] == second["audit_id"]
    result = await wait_result(service, first)
    assert result["status"] == "completed"
    assert result["self_judged"] is True
    assert result["judge"]["topic_coverage"] is None
    assert result["method_status"] == "unverifiable"
    assert result["jev"]["status"] == "not_requested"
    assert client.calls == 2 and client.closed
    assert original == before
    again = await service.submit(original)
    assert again["reused"] and again["audit_id"] == first["audit_id"]
    with pytest.raises(KeyError):
        await service.get("report-1", 2, first["audit_id"])
    assert not (await service.list("report-1", 2))["audits"]
    bad = snapshot()
    bad["source_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="report_sha_mismatch"):
        await service.submit(bad)
    await service.shutdown()


@pytest.mark.asyncio
async def test_new_attempt_missing_key_offline_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    clients = []

    def factory(*_):
        client = OfflineJudge()
        clients.append(client)
        return client

    service = ReportAuditService(config(), tmp_path, client_factory=factory)
    first = await wait_result(service, await service.submit(snapshot()))
    second = await wait_result(
        service, await service.submit(snapshot(), new_attempt=True)
    )
    assert first["audit_id"] != second["audit_id"] and len(clients) == 2
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        await service.submit(snapshot(), include_jev=True)
    changed = snapshot(methods={"content_mode": "abstract"})
    third = await wait_result(service, await service.submit(changed))
    assert third["binding"] != first["binding"]
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "offline")
    with pytest.raises(ValueError, match="audit_offline_or_disabled"):
        await service.submit(snapshot(revision=2))
    assert service.dry_run(snapshot(), include_jev=True)["paid_calls"] == 0


@pytest.mark.asyncio
async def test_budget_and_missing_usage_do_not_start_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    client = OfflineJudge(missing_usage=True)
    service = ReportAuditService(config(), tmp_path, client_factory=lambda *_: client)
    result = await wait_result(service, await service.submit(snapshot()))
    assert client.calls == 1
    assert result["judge"]["reason_codes"] == ["judge_budget_exhausted"]
    assert result["judge_usage"]["usage_complete"] is False
    context = SurveyAuditContext(query="q")
    no_call = AsyncMock()
    judge, usage = await SurveyBenchmarkJudge(
        no_call, SurveyJudgeConfig(base_url="https://offline.example", model="offline")
    ).evaluate(
        case=context,
        survey="Claim A [E:ev-1].",
        ledger=snapshot()["ledger"],
        input_budget_tokens=1,
    )
    assert judge.status == "failed" and usage.call_count == 0
    no_call.chat.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_and_restart_never_repeat_inflight(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    entered = asyncio.Event()

    class WaitingJudge(OfflineJudge):
        async def chat(self, *args, **kwargs):
            self.calls += 1
            entered.set()
            await asyncio.Event().wait()

    client = WaitingJudge()
    service = ReportAuditService(config(), tmp_path, client_factory=lambda *_: client)
    result = await service.submit(snapshot())
    await entered.wait()
    cancelled = await service.cancel("report-1", 1, result["audit_id"])
    assert cancelled["status"] == "cancelled" and client.calls == 1 and client.closed
    assert cancelled["judge_usage"]["usage_complete"] is False
    # Simulate a crash after the persisted reservation, without restarting a provider.
    cancelled["status"] = "running"
    service._save(cancelled)
    factory = AsyncMock()
    restarted = ReportAuditService(config(), tmp_path, client_factory=factory)
    await restarted.initialize()
    assert (await restarted.get("report-1", 1, result["audit_id"]))[
        "status"
    ] == "interrupted"
    factory.assert_not_called()
    reused = await restarted.submit(snapshot())
    assert (
        reused["audit_id"] == result["audit_id"] and reused["status"] == "interrupted"
    )
    factory.assert_not_called()
    assert not restarted._tasks


@pytest.mark.asyncio
async def test_jev_same_inventory_and_confidence_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            options = list(question["criteria"])
            answers[key] = {
                "type": "choice",
                "choice": options[0],
                "probabilities": {
                    option: 1.0 if option == options[0] else 0 for option in options
                },
                "confidence": 0.97,
            }
            assert question["instructions"]["state_pointer"] == f"audit_items.{key}"
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        service = ReportAuditService(
            config(),
            tmp_path,
            client_factory=lambda *_: OfflineJudge(),
            jev_factory=lambda *_: JevCitationAuditor(client=transport),
        )
        source = snapshot(
            "This run used abstract evidence.\n\nClaim A [E:ev-1].",
            methods={"content_mode": "abstract"},
        )
        result = await wait_result(
            service, await service.submit(source, include_jev=True)
        )
        assert result["status"] == "completed"
        assert (
            result["jev"]["inventory_sha256"]
            == result["judge"]["inventory_snapshot"]["inventory_sha256"]
        )
        assert result["jev"]["source_report_sha256"] == source["source_report_sha256"]
        assert result["jev"]["summary"]["agreement"] == 1
        assert (
            len(requests[0]["questions"]) == 2
        )  # Scientific group + evidence member; no runtime assertions.
        from litagent.benchmark.citation_audit import CitationAuditResult

        jev = CitationAuditResult.model_validate(result["jev"])
        jev.decisions[0].uncertain = True
        assert (
            derive_citation_audit_summary(jev, reference_judge=result["judge"])[
                "agreement"
            ]
            is None
        )
        jev.decisions[0].uncertain = False
        changed = copy.deepcopy(result["judge"])
        changed["inventory_snapshot"]["inventory_sha256"] = "wrong"
        assert (
            derive_citation_audit_summary(jev, reference_judge=changed)["agreement"]
            is None
        )
        assert result["jev"]["summary"]["accuracy"] is None


@pytest.mark.asyncio
async def test_jev_blocked_by_unfaithful_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    jev = AsyncMock()
    service = ReportAuditService(
        config(),
        tmp_path,
        client_factory=lambda *_: OfflineJudge(faithful=False),
        jev_factory=jev,
    )
    result = await wait_result(
        service, await service.submit(snapshot(), include_jev=True)
    )
    assert result["jev"]["status"] == "blocked"
    assert result["status"] == "failed"
    jev.assert_not_called()


@pytest.mark.asyncio
async def test_api_exact_sha_auth_version_and_no_paid_calls(tmp_path, monkeypatch):
    from litagent import api as module

    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    service = ReportAuditService(
        config(), tmp_path, client_factory=lambda *_: OfflineJudge()
    )
    monkeypatch.setattr(module.app.state, "report_audit_service", service)

    # Keep the actual resolver's 404 contract for version-missing tests.
    async def bound(run_id, revision):
        if run_id != "report-1" or revision != 1:
            from fastapi import HTTPException

            raise HTTPException(404, "report_revision_not_found")
        return snapshot()

    monkeypatch.setattr(module, "_report_audit_snapshot", bound)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=module.app), base_url="http://test"
    ) as client:
        prefix = "/survey/report-1/versions/1/audits"
        dry = await client.get(prefix + "/dry-run?include_jev=true")
        assert dry.json()["paid_calls"] == 0 and not service._tasks
        body = {"report_sha256": snapshot()["source_report_sha256"]}
        assert (await client.post(prefix, json=body)).status_code == 403
        headers = {"X-LitAgent-Operation-Token": module.app.state.operation_token}
        bad = await client.post(
            prefix, json={"report_sha256": "0" * 64}, headers=headers
        )
        assert bad.status_code == 409
        missing = await client.get("/survey/report-1/versions/2/audits")
        assert missing.status_code == 404
        invalid = await client.post(
            prefix, json={**body, "model": "evil"}, headers=headers
        )
        assert invalid.status_code == 422
        started = await client.post(prefix, json=body, headers=headers)
        assert started.status_code == 202
        result = await wait_result(service, started.json())
        assert (await client.get(prefix + "/" + result["audit_id"])).json()[
            "status"
        ] == "completed"
        assert (
            await client.get("/survey/report-1/versions/2/audits/" + result["audit_id"])
        ).status_code == 404
    await service.shutdown()


@pytest.mark.asyncio
async def test_unknown_and_combined_citations_keep_jev_local_error_denominator(
    tmp_path, monkeypatch
):
    """Unknown refs remain in the frozen inventory instead of disappearing."""
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    source = snapshot("Claim A [E:ev-1] [E:missing].\n\nClaim A [E:ev-1].")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: (_ for _ in ()).throw(
                AssertionError("No local-error claim may be sent")
            )
        )
    ) as transport:
        service = ReportAuditService(
            config(),
            tmp_path,
            client_factory=lambda *_: OfflineJudge(),
            jev_factory=lambda *_: JevCitationAuditor(client=transport),
        )
        result = await wait_result(
            service, await service.submit(source, include_jev=True)
        )
        assert result["judge"]["status"] == "completed"
        assert result["judge"]["faithfulness"] == 0
        assert result["judge"]["inventory_snapshot"]["claims"][0]["evidence_ids"] == [
            "ev-1",
            "missing",
        ]
        assert result["jev"]["summary"]["planned_claims"] == 1
        assert result["jev"]["summary"]["local_error_questions"] == 1
        assert result["jev"]["summary"]["claim_completion_rate"] == 0


def test_source_model_comparison_uses_historical_provider(tmp_path):
    """Provider changes cannot relabel an old report as self-judged."""
    service = ReportAuditService(config(), tmp_path)
    source = snapshot()
    source["generation_provider"] = {
        "model": "other-model",
        "origin": config().llm.base_url,
    }
    assert service.model_comparison(source)["self_judged"] is False
    source["generation_provider"] = {"model": None, "origin": None}
    assert service.model_comparison(source)["self_judged"] is None


@pytest.mark.asyncio
async def test_actual_snapshot_resolver_uses_raw_version_and_host_evidence(monkeypatch):
    """Exercise the producer without hiding it behind an API snapshot mock."""
    from litagent import api as module

    source = snapshot()
    source["report"]["metadata"].update(
        query="q",
        review_structure="academic_v1",
        review_methodology={"content_mode": "abstract"},
    )
    artifact = {
        "run_id": "report-1",
        "report": source["report"],
        "nodes": {
            "evidence.retrieve": {
                "output": {
                    "selected_items": {
                        "ev-1": {"evidence_id": "ev-1", **source["ledger"]["ev-1"]}
                    }
                }
            }
        },
    }
    service = AsyncMock()
    service.get_version.side_effect = KeyError("report-1")
    service.get_audit_artifact.side_effect = KeyError("report-1")
    monkeypatch.setattr(module, "_interactive_service", AsyncMock(return_value=service))
    monkeypatch.setattr(module, "_flow_artifact", lambda _: artifact)
    before = copy.deepcopy(artifact)
    frozen = await module._report_audit_snapshot("report-1", 1)
    assert frozen["report"]["survey"] == source["report"]["survey"]
    assert frozen["context"]["review_methodology"] == {"content_mode": "abstract"}
    assert frozen["ledger"]["ev-1"]["text"] == "Claim A."
    assert frozen["source_report_sha256"] == source["source_report_sha256"]
    assert artifact == before
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await module._report_audit_snapshot("report-1", 2)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_method_only_report_has_no_scientific_score():
    """A procedure disclosure must not become an unsupported scientific claim."""
    state = {}
    result, _ = await SurveyBenchmarkJudge(
        OfflineJudge(),
        SurveyJudgeConfig(base_url="https://offline.example", model="offline"),
    ).evaluate(
        case=SurveyAuditContext(
            query="q", review_methodology={"content_mode": "abstract"}
        ),
        survey="This run used abstract evidence.",
        ledger={},
        checkpoint=state,
    )
    assert result.status == "completed" and result.factual_claim_count == 0
    assert result.faithfulness is None
    assert state["method_assessments"][0]["status"] == "verified"


def test_fenced_or_quoted_headings_cannot_satisfy_structure():
    context = SurveyAuditContext(query="q", review_structure="academic_v1")
    sections = check_structure("", context)["expected_sections"]
    fake = (
        "# Review\n\n```markdown\n"
        + "\n".join(f"## {title}\nBody" for title in sections)
        + "\n```"
    )
    assert check_structure(fake, context)["status"] == "failed"
    quoted = "# Review\n\n" + "\n".join(f"> ## {title}\n> Body" for title in sections)
    assert check_structure(quoted, context)["status"] == "failed"


@pytest.mark.asyncio
async def test_explicit_new_attempt_still_deduplicates_pending_work(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LLM_API_KEY", "offline-key")
    entered = asyncio.Event()

    class WaitingJudge(OfflineJudge):
        async def chat(self, *args, **kwargs):
            self.calls += 1
            entered.set()
            await asyncio.Event().wait()

    client = WaitingJudge()
    service = ReportAuditService(config(), tmp_path, client_factory=lambda *_: client)
    first = await service.submit(snapshot(), new_attempt=True)
    await entered.wait()
    second = await service.submit(snapshot(), new_attempt=True)
    assert second["audit_id"] == first["audit_id"] and second["reused"]
    await service.cancel("report-1", 1, first["audit_id"])
    assert client.calls == 1


def test_judge_inherits_existing_key_fallback_and_explicit_override(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    service = ReportAuditService(config(), tmp_path)
    assert service.judge_config().api_key_env == "OPENAI_API_KEY"
    assert service.capabilities()["judge_available"] is True
    service.config.report_audit.judge_api_key_env = "BENCHMARK_JUDGE_API_KEY"
    monkeypatch.delenv("BENCHMARK_JUDGE_API_KEY", raising=False)
    assert service.capabilities()["judge_available"] is False


def test_separate_judge_provider_requires_its_explicit_key_environment():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="explicit judge_api_key_env"):
        AppConfig(
            agent={},
            logging={},
            report_audit={"judge_base_url": "https://independent.example"},
        )
    cfg = AppConfig(
        agent={},
        logging={},
        report_audit={
            "judge_base_url": "https://independent.example",
            "judge_api_key_env": "BENCHMARK_JUDGE_API_KEY",
        },
    )
    assert cfg.report_audit.judge_api_key_env == "BENCHMARK_JUDGE_API_KEY"
