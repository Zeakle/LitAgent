"""Exercise both loops through the ordinary Survey API and real consumers."""

import asyncio
import copy
import json

import httpx
import pytest
from openai import AsyncOpenAI
from test_native_responses import envelope
from test_workbench_survey_repair import QUERY, WorkbenchGateway
from test_writer_v2 import case  # noqa: F401

from litagent import api
from litagent.config import load_config
from litagent.observability.recorder import ArchiveRepository
from litagent.runner import LitAgent


class SeparateGateway(WorkbenchGateway):
    """Provide only HTTP responses; application control flow stays real."""

    def __init__(
        self,
        fixture,
        *,
        reviews=1,
        failures=0,
        keep=False,
        improve=False,
        format_fault=None,
        editorial_fault=None,
    ):
        super().__init__(fixture, failures)
        self.reviews = reviews
        self.review_calls = 0
        self.editorial_edits = 0
        self.keep = keep
        self.improve = improve
        self.events = []
        self.format_fault = format_fault
        self.format_calls = 0
        self.format_injected = False
        self.editorial_fault = editorial_fault

    async def __call__(self, request):
        """Recover a damaged editorial reply through the actual format agent."""
        if request.url.path.endswith("/responses"):
            body = json.loads(request.content)
            payload = json.loads(body["input"][1]["content"])
            if "anchors" in payload:
                self.format_calls += 1
                raw = payload["raw"]
                response = envelope(
                    json.dumps(
                        {
                            "outcome": "patch",
                            "edits": [
                                {
                                    "start": len(raw) - 1,
                                    "end": len(raw),
                                    "replacement": "",
                                }
                            ],
                        }
                    ),
                    model=body["model"],
                )
                response["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
                return httpx.Response(200, json=response)
            result = await super().__call__(request)
            if (
                self.editorial_fault
                and "response_target_keys" in payload
                and self.editorial_edits == 3
                and self.global_calls == 0
            ):
                wire = result.json()
                content = wire["output"][-1]["content"][0]
                patch = json.loads(content["text"])
                patch["patches"][0]["text"] = ""
                patch["patches"][0]["fact_card_refs"] = []
                if self.editorial_fault == "missing_field":
                    del patch["explanation"]
                if self.editorial_fault == "unknown_usage":
                    wire.pop("usage")
                content["text"] = json.dumps(patch)
                return httpx.Response(200, json=wire)
            if (
                self.format_fault
                and not self.format_injected
                and self.editorial_edits
                and self.global_calls == 0
            ):
                wire = result.json()
                content = wire["output"][-1]["content"][0]
                assert content["text"].endswith("}")
                if self.format_fault == "missing_root_brace":
                    content["text"] = content["text"][:-1]
                else:
                    content["text"] += "}"
                self.format_injected = True
                return httpx.Response(200, json=wire)
            return result
        return await super().__call__(request)

    def answer(self, payload, native_schema):
        if "issues" in native_schema["properties"]:
            self.review_calls += 1
            self.events.append("review")
            uid = next(k for k in payload["units"] if ":introduction:" in k)
            return {
                "issues": (
                    [{"unit_id": uid, "explanation": "Clarify the review scope."}]
                    if self.review_calls <= self.reviews
                    else []
                ),
                "explanation": "Review the current draft only.",
            }
        if "target_responses" in native_schema["properties"]:
            editorial = self.global_calls == 0
            if editorial:
                self.editorial_edits += 1
            else:
                self.repair_count += 1
            self.events.append("editorial_edit" if editorial else "evaluation_edit")
            directory = payload["response_target_keys"]
            field = "body_unit_refs" if "body_units" in payload else "fact_card_refs"
            refs = ["u1"] if field == "body_unit_refs" else ["c1"]
            keep = self.keep == "all" or (self.keep and editorial)
            text = (
                f"This survey examines Relation networks in few-shot learning (revision {self.editorial_edits})."
                if editorial
                else (
                    f"Relation networks reduce FLOPs by {44 + self.repair_count}%."
                    if self.repair_count <= self.failing_candidates
                    else "In the cited few-shot setting, Relation networks learn a comparison function."
                )
            )
            result = {
                "outcome": "patched",
                "explanation": "Independently respond to authorized targets.",
                "target_responses": {
                    key: {
                        "action": "keep" if keep else "replace",
                        "reason": "Inspected the supplied original evidence.",
                        "reference_ids": refs,
                    }
                    for key in directory
                },
                "patches": (
                    []
                    if keep
                    else [
                        {"target_id": key, "text": text, field: refs}
                        for key in directory
                    ]
                ),
            }
            if self.improve and not editorial and self.repair_count == 1:
                for key, tid in directory.items():
                    if ":experiments:" in tid:
                        result["target_responses"][key]["action"] = "keep"
                        result["patches"] = [
                            p for p in result["patches"] if p["target_id"] != key
                        ]
            return result
        result = super().answer(payload, native_schema)
        if self.improve and payload.get("requested_body_section") == "experiments":
            result["units"][0]["text"] = "Relation networks reduce FLOPs by 12%."
        return result


async def execute(case, tmp_path, monkeypatch, gateway):
    """Submit the server-selected policy without replacing any report consumers."""
    config = load_config("config/separate-review-evaluation.yaml")
    config.observability.enabled = False
    config.llm.base_url = "https://offline.invalid/api/v3"
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "config": config,
        "flow_repository": ArchiveRepository(tmp_path / "archive"),
        "tasks": {},
        "survey_semaphore": None,
        "report_audit_service": None,
    }.items():
        monkeypatch.setattr(api.app.state, key, value)

    async def unavailable_infra(self, config, *, infra):
        return infra

    monkeypatch.setattr(LitAgent, "_connect_infra", unavailable_infra)
    real_init = httpx.AsyncClient.__init__

    def offline_client(self, *args, **kwargs):
        if not isinstance(kwargs.get("transport"), httpx.ASGITransport):
            kwargs.update(transport=httpx.MockTransport(gateway), trust_env=False)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", offline_client)
    monkeypatch.setattr(
        "litagent.llm.client.AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(gateway), trust_env=False
            ),
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://localhost"
    ) as client:
        context = (await client.get("/workbench/context")).json()
        options = context["automatic_acceptance"]
        response = await client.post(
            "/survey",
            json={
                "query": QUERY,
                "quality_policy": context["default_quality_policy"],
                "report_language": "en",
                "review_structure": "academic_v2",
                "citation_style": "ieee",
                "human_review": False,
                "body_quality": options["default_body_quality"],
                "generation_strategy": options["default_generation_strategy"],
                "budget_policy": options["default_budget_policy"],
                "capacity_policy": "adaptive-capacity-v1",
                "verification": {**options["default_verification"], "jev": False},
                "auto_audit_mode": "off",
                "auto_audit_max_versions": 1,
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["task_id"]
        try:
            # Full request construction is CPU-bound under parallel regression;
            # this test watchdog must cover all six edits, not one repair cycle.
            await asyncio.wait_for(api.app.state.tasks[run_id].task, timeout=900)
            state = (await client.get(f"/survey/{run_id}")).json()
            terminal = await client.get(f"/survey/{run_id}/report")
        finally:
            if api.app.state.report_audit_service is not None:
                await api.app.state.report_audit_service.shutdown()
    assert not gateway.unexpected
    assert state["status"] == "completed" and terminal.status_code == 200, state
    return terminal.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reviews,failures,keep,format_fault",
    [
        (1, 0, False, "extra_brace"),
        (1, 0, False, "missing_root_brace"),
        (9, 2, False, None),
        (9, 0, True, None),
        (0, 3, False, None),
        (0, 0, "all", None),
    ],
)
async def test_editorial_then_evaluation_independent_real_chain(
    case, tmp_path, monkeypatch, reviews, failures, keep, format_fault
):
    gateway = SeparateGateway(
        case, reviews=reviews, failures=failures, keep=keep, format_fault=format_fault
    )
    report = await execute(case, tmp_path, monkeypatch, gateway)
    (tmp_path / "observed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    assert report["delivery"]["publishable"] is (
        failures < 3 and keep != "all"
    ), report.get("metadata", {}).get("writer_failure")
    assert gateway.review_calls == (1 if keep else min(4, reviews + 1))
    assert gateway.editorial_edits == (1 if keep and reviews else min(3, reviews))
    assert gateway.repair_count == min(3, failures + 1)
    assert gateway.global_calls == (1 if keep == "all" else min(3, failures + 1) + 1)
    assert gateway.events.index("evaluation_edit") > max(
        i for i, e in enumerate(gateway.events) if e == "review"
    )
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert (
        sum(a["category"] == "review_revision" for a in attempts)
        == gateway.editorial_edits
    )
    assert sum(a["category"] == "evaluation_revision" for a in attempts) == min(
        3, failures + 1
    )
    assert ("FLOPs by" not in report["survey"]) is report["delivery"]["publishable"]
    assert gateway.format_calls == int(format_fault == "extra_brace")
    if format_fault == "missing_root_brace":
        recovered = report["metadata"]["writer_format"]
        assert recovered["recovered_count"] == 1
        assert recovered["requests_used"] == 0
        assert recovered["calls"][0]["kinds"] == ["local_eof · accepted"]
    workflow = report["metadata"]["review_workflow"]
    # The baseline 0.8 metric can pass while strict scientific acceptance fails.
    assert workflow["evaluations"][0]["status"] == "failed"
    assert workflow["evaluations"][0]["metric_status"] == "passed"
    assert workflow["allowances"]["review_revision"]["used"] == gateway.editorial_edits
    assert workflow["allowances"]["evaluation_revision"]["used"] == gateway.repair_count
    assert workflow["publishable"] is report["delivery"]["publishable"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["empty_replacement", "missing_field", "unknown_usage"]
)
async def test_third_editorial_rejection_only_retains_draft_for_settled_empty_patch(
    case, tmp_path, monkeypatch, fault
):
    gateway = SeparateGateway(case, reviews=3, editorial_fault=fault)
    report = await execute(case, tmp_path, monkeypatch, gateway)
    meta = report["metadata"]
    assert gateway.editorial_edits == 3
    if fault != "empty_replacement":
        assert report["partial"] and not report["delivery"]["publishable"]
        assert gateway.global_calls == gateway.repair_count == 0
        assert meta["writer_failure"]["code"] == (
            "writer_flat_schema_invalid"
            if fault == "missing_field"
            else "strict_usage_incomplete"
        )
        return

    first, second, rejected = meta["editorial_history"]
    assert first["status"] == second["status"] == "revised_unverified"
    assert rejected["status"] == "candidate_rejected"
    assert rejected["reason_code"] == "writer_flat_schema_invalid"
    retained_sha = second["candidate_sha256"]
    assert rejected["source_report_sha256"] == retained_sha
    initial = meta["evaluation_history"]["initial_evaluation"]
    assert initial["faithfulness"]["details"]["source_report_sha256"] == retained_sha
    assert "revision 2" in report["survey"] and "revision 3" not in report["survey"]
    assert gateway.global_calls == 2 and gateway.repair_count == 1
    assert gateway.format_calls == 0
    assert meta["review_workflow"]["allowances"]["review_revision"]["used"] == 3
    assert report["delivery"]["publishable"]


@pytest.mark.asyncio
async def test_large_revised_inventory_reaches_automatic_delivery(
    case, tmp_path, monkeypatch
):
    """Exercise budget-partition validation through the real report outlet."""
    gateway = SeparateGateway(case, reviews=0)
    gateway.statements_per_unit = 25
    report = await execute(case, tmp_path, monkeypatch, gateway)
    metric = report["evaluation"]["faithfulness"]
    assert metric["details"]["factual_claim_count"] > 128
    assert metric["score"] == 1.0
    assert gateway.repair_count == 1 and gateway.global_calls == 2
    assert report["metadata"]["automatic_acceptance"]["inventory_sha256"]
    assert report["delivery"]["publishable"], report["delivery"]


@pytest.mark.asyncio
async def test_improved_unpublished_candidate_becomes_next_repair_base(
    case, tmp_path, monkeypatch
):
    gateway = SeparateGateway(case, reviews=0, improve=True)
    report = await execute(case, tmp_path, monkeypatch, gateway)
    (tmp_path / "observed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    history = report["metadata"]["evaluation_history"]["attempts"]
    assert len(history) == 2, report.get("metadata", {}).get("writer_failure")
    first, last = history
    assert first["working_draft_decision"]["promote"], first.get(
        "working_draft_decision"
    )
    assert not first["committed"]
    assert (
        last["recovery_decision"]["source_report_sha256"]
        == first["next_repair_base_sha256"]
    )
    assert last["committed"] and report["delivery"]["publishable"]
    assert gateway.repair_count == 2 and gateway.global_calls == 3
