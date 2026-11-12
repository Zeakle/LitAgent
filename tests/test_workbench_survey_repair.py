"""Exercise normal Survey submission with real workers and offline provider HTTP."""

import asyncio
import copy
import hashlib
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI
from test_native_ragas_pipeline import NativeGateway
from test_writer_v2 import case  # noqa: F401

from litagent import api
from litagent.config import load_config
from litagent.eval.engine_policy import CONTRACT, KEYED_GLOBAL_QUOTED_OMISSIONS
from litagent.eval.ragas_global_wire import unpack_global_input
from litagent.observability.recorder import ArchiveRepository
from litagent.runner import LitAgent
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_v2 import BODY

QUERY = "Explain mechanisms and limitations of Relation Networks in few-shot learning."
ABSTRACT = "In few-shot learning, Relation networks learn a comparison function."
PAPER = "1711.06025"


class WorkbenchGateway(NativeGateway):
    """Replace search, full-text and model HTTP without bypassing report workers."""

    def __init__(
        self, fixture, failing_candidates, *, mixed_summary=False, repetition=False
    ):
        """Align the reusable writer fixture with the one selected source."""
        fixture = copy.deepcopy(fixture)
        fixture.by_eid["b"] = "c1"
        for section in BODY:
            fixture.body[section][0]["sentences"][0]["fact_card_refs"] = ["c1"]
        super().__init__(fixture, failures=failing_candidates)
        self.discovery = []
        self.extractions = 0
        self.unexpected = []
        self.mixed_summary = mixed_summary
        self.repetition = repetition
        self.protected_finding = "Relation networks learn a comparison function."
        self.unique_finding = "The comparison function operates in few-shot learning."
        if repetition:
            original = self.semantic.chat

            async def repetition_response(messages):
                response = await original(messages)
                payload = json.loads(messages[1]["content"])
                raw = json.loads(response.content)
                if "jobs" in payload and "context_pool" not in payload:
                    for item in raw.values():
                        if item["statements"] == [
                            self.protected_finding + " " + self.unique_finding
                        ]:
                            item["statements"] = [
                                self.protected_finding,
                                self.unique_finding,
                            ]
                if "source_units" in payload and self.global_calls == 1:
                    if "global_input_contract" in payload:
                        payload = unpack_global_input(payload)
                    ids = [
                        row["claim_id"]
                        for row in payload["assertions"]
                        if row["section"] == "Representative Methods"
                        and row["claim_text"] == self.protected_finding
                    ]
                    assert len(ids) == 2
                    raw["issues"] = [
                        {
                            "code": "repetition",
                            "anchor_ids": ids,
                            "explanation": "The same finding appears in two methods paragraphs.",
                        }
                    ]
                response.content = json.dumps(raw)
                return response

            self.semantic.chat = repetition_response

    def answer(self, payload, native_schema):
        """Put a separate unsupported finding in the initial abstract."""
        value = super().answer(payload, native_schema)
        if self.mixed_summary and "abstract" in value and self.repair_count == 0:
            value["abstract"][0].update(
                text="Relation networks reduce FLOPs by 61%.",
                body_unit_refs=["u1"],
            )
        if self.repetition and payload.get("requested_body_section") == "methods":
            value["units"].insert(
                0, {**value["units"][0], "text": self.protected_finding}
            )
        if self.repetition and "targets" in payload:
            for index, patch in enumerate(value["patches"]):
                patch["text"] = (
                    self.protected_finding + " " + self.unique_finding
                    if self.repair_count == 1
                    else self.protected_finding if index == 0 else self.unique_finding
                )
        return value

    async def __call__(self, request):
        """Serve fixture data at the actual HTTP transport boundary."""
        host, path = request.url.host, request.url.path
        if host == "export.arxiv.org" and path == "/api/query":
            self.discovery.append("arxiv")
            return httpx.Response(
                200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>'
            )
        if host == "huggingface.co" and path == "/api/papers/search":
            self.discovery.append("huggingface")
            return httpx.Response(
                200,
                json=[
                    {
                        "paper": {
                            "id": PAPER,
                            "title": "Relation Networks for Few-Shot Learning",
                            "summary": ABSTRACT,
                        }
                    }
                ],
            )
        if host == "arxiv.org" and path == f"/pdf/{PAPER}":
            return httpx.Response(404)
        if host != "offline.invalid":
            self.unexpected.append((host, path))
            return httpx.Response(403)
        if path.endswith("/responses"):
            return await super().__call__(request)
        body = json.loads(request.content)
        system, user = body["messages"][:2]
        if "query planning assistant" in system["content"]:
            result = {
                "sub_queries": ["Relation Networks few-shot learning"],
                "named_methods": [],
                "coverage_constraints": [],
                "original_paper_queries": [],
            }
        else:
            assert path.endswith("/chat/completions"), path
            rf = body["response_format"]["json_schema"]
            assert rf["name"] == "host_grounded_claims_v1", rf["name"]
            self.extractions += 1
            (member,) = json.loads(user["content"])["papers"].items()
            alias, paper = member
            assert ABSTRACT in paper["source_context"]
            result = {
                "papers": {
                    alias: {
                        "claim_records": [
                            {
                                "finding": "Relation networks learn a comparison function.",
                                "source_chunk_key": paper["allowed_source_chunk_keys"][
                                    0
                                ],
                                "confidence": 0.95,
                                "fact_fields": {
                                    "method": "Relation networks",
                                    "conditions": "few-shot learning",
                                    "dataset": None,
                                    "metric": None,
                                    "value": None,
                                    "unit": None,
                                    "qualifiers": None,
                                },
                            }
                        ]
                    }
                }
            }
            Draft202012Validator(rf["schema"]).validate(result)
        return httpx.Response(
            200,
            json={
                "id": "offline-workbench",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing_candidates", "expanded", "repetition"),
    [(0, False, False), (1, False, False), (0, True, False), (0, True, True)],
    ids=["stop-early", "retry", "expanded-live-default", "repetition-followup"],
)
async def test_workbench_survey_repairs_and_stops_after_full_reassessment(
    case, tmp_path, monkeypatch, failing_candidates, expanded, repetition
):
    """Submit the normal workbench payload and keep only the passing version."""
    monkeypatch.delenv("LITAGENT_CONFIG", raising=False)
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")
    config = (
        api._load_runtime_config() if expanded else load_config("config/default.yaml")
    )
    max_requests, max_repairs = (226, 11) if expanded else (200, 2)
    config.observability.enabled = False
    config.llm.base_url = "https://offline.invalid/api/v3"
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    repository = ArchiveRepository(tmp_path / "archive")
    for key, value in {
        "config": config,
        "flow_repository": repository,
        "tasks": {},
        "survey_semaphore": None,
        "report_audit_service": None,
    }.items():
        monkeypatch.setattr(api.app.state, key, value)

    async def unavailable_infra(self, config, *, infra):
        """Simulate optional backing services being absent."""
        return infra

    monkeypatch.setattr(LitAgent, "_connect_infra", unavailable_infra)
    gateway = WorkbenchGateway(
        case,
        failing_candidates,
        mixed_summary=expanded and not repetition,
        repetition=repetition,
    )
    if expanded and not repetition:
        gateway.recovery_case = "combined"
    real_init = httpx.AsyncClient.__init__

    def offline_client(self, *args, **kwargs):
        """Prevent every non-ASGI HTTP client from leaving the test."""
        if not isinstance(kwargs.get("transport"), httpx.ASGITransport):
            kwargs.update(transport=httpx.MockTransport(gateway), trust_env=False)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", offline_client)

    def offline_sdk(**kwargs):
        """Inject HTTP explicitly because the SDK can use a different HTTP backend."""
        return AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(gateway), trust_env=False
            ),
        )

    monkeypatch.setattr("litagent.llm.client.AsyncOpenAI", offline_sdk)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://localhost"
    ) as client:
        context = (await client.get("/workbench/context")).json()
        automatic = context["automatic_acceptance"]
        assert automatic["maximum_survey_requests"] == max_requests
        assert context["capacity_capabilities"]["maximum_requests"] == max_requests
        if expanded:
            assert automatic["maximum_survey_tokens"] == 47_420_000
            assert automatic["maximum_repair_attempts_by_category"] == {
                "capacity": 3,
                "inventory": 3,
                "prose": 5,
            }
            assert automatic["jev_requests"] == 240
            assert automatic["jev_input_tokens"] == 4_800_000
        response = await client.post(
            "/survey",
            json={
                "query": QUERY,
                "quality_policy": context["default_quality_policy"],
                "report_language": "en",
                "review_structure": "academic_v2",
                "citation_style": "ieee",
                "human_review": False,
                "body_quality": automatic["default_body_quality"],
                "generation_strategy": automatic["default_generation_strategy"],
                "budget_policy": automatic["default_budget_policy"],
                "capacity_policy": "adaptive-capacity-v1",
                "verification": {
                    **automatic["default_verification"],
                    "jev": False,
                },
                "auto_audit_mode": "off",
                "auto_audit_max_versions": 1,
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["task_id"]
        try:
            await asyncio.wait_for(api.app.state.tasks[run_id].task, timeout=240)
            state = (await client.get(f"/survey/{run_id}")).json()
            terminal = await client.get(f"/survey/{run_id}/report")
        finally:
            if api.app.state.report_audit_service is not None:
                await api.app.state.report_audit_service.shutdown()

    assert not gateway.unexpected
    assert state["status"] == "completed" and terminal.status_code == 200, state
    report = terminal.json()
    assert set(gateway.discovery) == {"arxiv", "huggingface"}
    assert gateway.extractions == 1
    history = report["metadata"]["evaluation_history"]
    initial = history["initial_evaluation"]["faithfulness"]["details"]
    assert initial["implementation"] == CONTRACT
    assert initial["unsupported_count"] == (2 if expanded and not repetition else 1)
    assert initial["coverage"]["complete"]
    assert initial["stages"]["global_check"] == "completed"
    if expanded:
        assert (
            initial["stages"]["global_unit_response_contract"]
            == KEYED_GLOBAL_QUOTED_OMISSIONS
        )
    if expanded and not repetition:
        (omission,) = initial["global_issues"]
        assert omission["code"] == "missing_assertion"
        assert omission["source_unit_id"].startswith("unit:")
        assert omission["claim_ids"] == []
    *rejected, passed = history["attempts"]
    assert len(rejected) == failing_candidates + repetition
    if repetition:
        (draft,) = rejected
        assert not draft["committed"]
        assert draft["candidate_evaluation"]["faithfulness"]["score"] == 1
        assert draft["next_repair_base_sha256"] == (
            passed["recovery_decision"]["source_report_sha256"]
        )
        assert gateway.unique_finding in report["survey"]
        assert report["survey"].count(gateway.protected_finding) >= 1
    elif rejected:
        (failed,) = rejected
        assert failed["reason_code"] == "writer_candidate_failed"
        assert not failed["committed"] and "FLOPs by 45%" in failed["candidate"]
    assert passed["committed"] and passed["candidate"] == report["survey"]
    if expanded and not repetition:
        assert passed["recovery_decision"]["action"] == "combined"
    assert "FLOPs by" not in report["survey"]
    expected_attempts = failing_candidates + 1 + repetition
    expected_assessments = expected_attempts + 1
    assert gateway.repair_count == expected_attempts
    assert gateway.global_calls == expected_assessments
    assert (
        sum("body_units" in row["payload"] for row in gateway.sent)
        == expected_assessments
    )
    assessments = [
        row["payload"]["context_pool"]
        for row in gateway.sent
        if "context_pool" in row["payload"]
    ]
    assert len(assessments) == expected_assessments
    assert all(ABSTRACT in json.dumps(pool) for pool in assessments)
    for index, row in enumerate(history["attempts"]):
        assert row["recovery_decision"]["source_report_sha256"] == (
            rejected[0]["next_repair_base_sha256"]
            if repetition and index == 1
            else initial["source_report_sha256"]
        )
        details = row["candidate_evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
    sha = hashlib.sha256(report["survey"].encode()).hexdigest()
    acceptance = report["metadata"]["automatic_acceptance"]
    assert report["delivery"]["publishable"] and acceptance["publishable"]
    assert state["delivery_status"] == "ready"
    assert sha == history["source_report_sha256"] == acceptance["report_sha256"]
    final = report["evaluation"]["faithfulness"]["details"]
    assert sha == final["source_report_sha256"]
    assert final["supported_count"] == final["factual_claim_count"] > 0
    assert final["unsupported_count"] == final["unverified_count"] == 0
    path = (
        tmp_path
        / "artifacts"
        / "targeted-repairs"
        / f"{run_id}.json.budget-ledger.json"
    )
    ledger = json.loads(path.read_bytes())
    validate_accounting_snapshot(ledger, expected_plan=ledger["plan"])
    assert ledger["plan"]["repair_limit"] == max_repairs
    assert ledger["plan"]["maximum_requests"] == max_requests
    if expanded:
        assert ledger["plan"]["maximum_tokens"] == 47_420_000
        assert ledger["plan"]["envelope"]["run_limits"]["syntax_policy"] == (
            "agent-format-recovery-v1"
        )
        assert (
            final["stages"]["capacity_admission"]["complete_inventory"][
                "statement_count"
            ]
            == final["factual_claim_count"]
        )
        journal = json.loads(
            path.with_name(f"{run_id}.json.writer-v2.json").read_bytes()
        )
        if repetition:
            first, second = journal["candidates"]
            assert first["committed"] is False and second["committed"] is True
            assert first["report_sha256"] == second["source_report_sha256"]
            assert (
                journal["repetition_retry"]["candidate_sha256"]
                == first["report_sha256"]
            )
            assert all(
                target.startswith("writer:methods:")
                for target in second["repair_receipt"]["authorized_targets"]
            )
            assert (
                sum(
                    claim["section"] == "Representative Methods"
                    and claim["claim_text"] == gateway.protected_finding
                    for claim in final["assertions"]
                )
                == 1
            )
        else:
            summary = journal["summary_regeneration_guidance"]["summary_repair_1"]
            assert [d["claim_text"] for d in summary["diagnostics"]] == [
                "Relation networks reduce FLOPs by 61%."
            ]
            assert summary["host_directory"]["issues"][0]["target_ids"] == [
                "writer:abstract:0:0"
            ]
            assert (
                summary["host_directory"]["source_report_sha256"]
                == initial["source_report_sha256"]
            )
    assert (
        sum(e["stage"] == "repair" for e in ledger["entries"].values())
        == expected_attempts
    )
    assert not ledger["usage_uncertain"]
    artifact = repository.get(run_id)
    assert artifact["report"]["survey"] == report["survey"]
    assert artifact["report"]["metadata"]["automatic_acceptance"] == acceptance
