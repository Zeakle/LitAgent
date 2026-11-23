"""Exercise the question-only global request, host gates and historical defaults."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_native_ragas_evaluation import SemanticTransport, native_case  # noqa: F401
from test_quality_view import render_repair_state

from litagent.config import EvalConfig, load_config
from litagent.eval import question_coverage, ragas_native
from litagent.eval.conflict_review import CONTRACT as CONFLICT_ONLY_SCOPE
from litagent.eval.engine_policy import configured_engine, validate_engine
from litagent.eval.responsibilities import REPORT_REVIEW_SCOPE
from litagent.eval.results import (
    EvaluationRunResult,
    acceptance,
    derive_quality,
    public_summary,
)
from litagent.llm.client import LLMResponse
from litagent.safety.budget import strict_request_reservation


@pytest.mark.parametrize("live", [False, True])
def test_default_freezes_scope_and_isolates_previous_checkpoint(live):
    profile = load_config(live_workbench=live)
    engine = configured_engine(profile)
    assert engine["global_scope"] == CONFLICT_ONLY_SCOPE
    assert validate_engine(engine) == engine
    profile.eval.native_global_scope = None
    old = configured_engine(profile)
    assert "global_scope" not in old and old["identity"] != engine["identity"]
    with pytest.raises(ValueError, match="identity_invalid"):
        validate_engine({**old, "global_scope": REPORT_REVIEW_SCOPE})


def test_request_reservation_covers_every_verdict_and_uses_only_coverage_schema():
    units = [SimpleNamespace(text="Measurements under X and Y.")]
    ctx = {
        "query": "Compare X and Y",
        "coverage_requirements": [
            {"requirement_id": "question", "text": "Compare X and Y"}
        ],
    }
    claims = [
        {
            "kind": "scientific",
            "claim_id": "claim_0",
            "claim_text": "Under X.",
            "status": "unverifiable",
        }
    ]

    def request():
        payload, messages, fmt = ragas_native.global_review_request(
            "",
            ctx,
            EvalConfig(),
            units,
            claims,
            global_scope=question_coverage.CONTRACT,
        )
        assert set(payload) == {
            "query",
            "coverage_requirements",
            "report_passages",
            "claims",
        }
        assert list(fmt["json_schema"]["schema"]["properties"]) == ["coverage"]
        assert json.loads(messages[1]["content"]) == payload
        return strict_request_reservation(messages, 16384, fmt)

    preflight = request()
    for verdict in ("supported", "unsupported"):
        claims[0]["status"] = verdict
        assert request() <= preflight


class CoverageTransport(SemanticTransport):
    """Replace only external generation, NLI and global question responses."""

    def __init__(self, answered, reject):
        super().__init__(reject=reject)
        self.answered = answered

    async def chat(self, messages, **kwargs):
        if (
            kwargs["response_format"]["json_schema"]["name"]
            != "ragas_question_coverage"
        ):
            return await super().chat(messages, **kwargs)
        data = json.loads(messages[1]["content"])
        self.calls.append((data, kwargs))
        cid = next(iter(data["claims"]))
        raw = {
            "coverage": [
                {
                    "requirement_id": r["requirement_id"],
                    "answered": self.answered,
                    "claim_ids": [cid],
                    "explanation": (
                        "Responds to the method question."
                        if self.answered
                        else "Only the setup is described."
                    ),
                }
                for r in data["coverage_requirements"]
            ]
        }
        return LLMResponse(
            json.dumps(raw),
            model="offline",
            finish_reason="stop",
            usage={
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answered", "reject", "blocker"),
    [
        (True, None, None),
        (False, None, "core_coverage_unverified"),
        (True, "Method A learns an embedding.", "science_not_fully_supported"),
    ],
)
async def test_only_question_and_measured_science_reach_delivery(
    native_case, monkeypatch, answered, reject, blocker
):
    text, context = native_case
    monkeypatch.setattr(
        ragas_native,
        "locked_writer_envelope",
        lambda _: {"semantic_engine": {"global_scope": question_coverage.CONTRACT}},
    )
    llm = CoverageTransport(answered, reject)
    run = await ragas_native.execute_native(
        llm=llm, survey=text, context=context, config=EvalConfig()
    )
    assert not run.execution_error, run.execution_error
    assert len(llm.calls) == 3
    assert run.execution_complete() and run.inventory_valid()
    assert run.checks()["core_coverage"]["status"] == (
        "passed" if answered else "failed"
    )
    assert all(
        run.checks()[key]["status"] == "not_applicable"
        for key in ("analysis", "limitations", "global_report")
    )
    metrics = run.metrics()
    assert metrics["internal_consistency"]["score"] is None
    assert metrics["internal_consistency"]["status"] == "not_applicable"
    metrics["citation_accuracy"] = {
        "passed": True,
        "skipped": False,
        "score": 1.0,
        "status": "passed",
    }
    quality = derive_quality(metrics)
    assert quality["status"] == ("failed" if blocker else "passed")
    assert acceptance(run.details)["failure_codes"] == ([blocker] if blocker else [])
    projection = public_summary({"survey": text, "evaluation": metrics})
    assert projection["metrics"]["internal_consistency"]["score"] is None
    assert projection["checks"]["analysis"]["status"] == "not_applicable"
    tampered = copy.deepcopy(run.details)
    del tampered["stages"]["global_scope"]
    with pytest.raises(ValueError, match="evaluation_result_contract_identity_invalid"):
        EvaluationRunResult.from_details(tampered, run.report_sha256)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "extra",
        "unknown",
        "duplicate",
        "missing_claim",
        "unknown_claim",
        "false_string",
        "legacy",
        "duplicate_json_key",
    ],
)
def test_question_review_refuses_unbound_or_invalid_response(fault):
    data = {
        "coverage_requirements": [{"requirement_id": "question"}],
        "claims": {"claim_1": "A fact"},
    }
    row = {
        "requirement_id": "question",
        "answered": True,
        "claim_ids": ["claim_1"],
        "explanation": "The report explains the effect.",
    }
    raw = {"coverage": [copy.deepcopy(row)]}
    if fault == "empty":
        raw["coverage"] = []
    elif fault == "extra":
        raw["analyses"] = []
    elif fault == "unknown":
        raw["coverage"][0]["requirement_id"] = "other"
    elif fault == "duplicate":
        raw["coverage"] *= 2
    elif fault == "missing_claim":
        raw["coverage"][0]["claim_ids"] = []
    elif fault == "unknown_claim":
        raw["coverage"][0]["claim_ids"] = ["claim_2"]
    elif fault == "false_string":
        raw["coverage"][0]["answered"] = "false"
    elif fault == "legacy":
        raw = {"units": [], "conflicts": []}
    content = json.dumps(raw)
    if fault == "duplicate_json_key":
        content = content.replace(
            '"answered": true', '"answered": false, "answered": true'
        )
    with pytest.raises(ValueError, match="evaluation_core_coverage"):
        question_coverage.parse_response(content, data)


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_workbench_explains_missing_answer_in_both_languages(locale):
    details = {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "evaluation_result_contract": "independent-evaluation-v1",
        "stages": {
            "global_scope": question_coverage.CONTRACT,
            "global_check": "completed",
        },
        "coverage_requirements": [{"requirement_id": "q", "text": "Compare X and Y"}],
        "core_coverage": [
            {
                "requirement_id": "q",
                "answered": False,
                "claim_ids": [],
                "explanation": "Y is missing.",
            }
        ],
    }
    text = render_repair_state(
        {"evaluation": {"faithfulness": {"details": details}}}, locale
    )["text"]
    assert ("用户问题覆盖" if locale == "zh-CN" else "User question coverage") in text
    assert "Y is missing." in text
    assert "0.2 ×" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "clear",
        "contradiction",
        "unsupported",
        "missing",
        "reversed",
        "self",
        "unknown",
        "units",
        "omission",
    ],
)
async def test_report_review_measures_conflicts_without_auditing_extraction(
    native_case, monkeypatch, mode
):
    """Exercise extraction, NLI, global response, scoring and bound repair diagnostics."""
    from litagent.eval.ragas_repair import native_repair_diagnostics

    text, context = native_case
    if mode == "contradiction":
        head, sep, tail = text.rpartition(
            "Method A reaches 80 percent under setting X."
        )
        assert sep
        text = head + "Method A reaches 60 percent under setting X." + tail
    llm = SemanticTransport(
        reject="Method A learns an embedding." if mode == "unsupported" else None
    )
    transport = llm.chat

    async def chat(messages, **kwargs):
        response = await transport(messages, **kwargs)
        if kwargs["response_format"]["json_schema"]["name"] != "ragas_report_review":
            return response
        schema = kwargs["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == {
            "coverage",
            "conflicts",
            "issues",
            "analyses",
            "limitations",
        }
        assert "missing_assertion" not in json.dumps(schema)
        payload = json.loads(messages[1]["content"])
        assert len(payload["assertions"]) == 7
        raw = json.loads(response.content)
        pair = {
            "claim_a": "claim_4",
            "claim_b": "claim_6",
            "reason": "Accuracy differs for the same method and setting.",
        }
        if mode in {"contradiction", "reversed", "self", "unknown"}:
            raw["conflicts"] = [pair]
        if mode == "reversed":
            raw["conflicts"].append(
                {**pair, "claim_a": pair["claim_b"], "claim_b": pair["claim_a"]}
            )
        if mode == "self":
            pair["claim_b"] = pair["claim_a"]
        if mode == "unknown":
            pair["claim_b"] = "claim_999"
        if mode == "missing":
            del raw["conflicts"]
        if mode == "units":
            raw["units"] = []
        if mode == "omission":
            raw["issues"] = [
                {
                    "code": "missing_assertion",
                    "anchor_ids": ["u1"],
                    "explanation": "Missing fact.",
                }
            ]
        response.content = json.dumps(raw)
        return response

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(
        ragas_native,
        "locked_writer_envelope",
        lambda _: {"semantic_engine": {"global_scope": REPORT_REVIEW_SCOPE}},
    )
    run = await ragas_native.execute_native(
        llm=llm, survey=text, context=context, config=EvalConfig()
    )
    assert len(llm.calls) == 3
    consistency = run.metrics()["internal_consistency"]
    gate = acceptance(run.details)
    if mode in {"reversed", "self", "unknown", "units", "omission", "missing"}:
        assert consistency["score"] is None
        assert gate["status"] == "failed"
        return
    assert run.execution_complete() and run.inventory_valid()
    assert run.checks()["inventory"]["scope"] == "processed_assertions"
    assert "units" not in run.details["stages"]["global_sections"]
    assert run.checks()["core_coverage"]["status"] == "passed"
    if mode == "contradiction":
        assert (
            consistency["score"] == 0.8
            and consistency["details"]["conflict_count"] == 1
        )
        assert gate["failure_codes"] == ["global_inconsistency"]
        assert {
            d["claim_id"]
            for d in native_repair_diagnostics(run.details)
            if d["reason_code"] == "evaluation_report_contradiction"
        } == {"claim_4", "claim_6"}
    else:
        assert consistency["score"] == 1.0
        assert (gate["status"] == "passed") == (mode == "clear")


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_workbench_shows_measured_consistency_and_scope(locale):
    """Keep omitted recall separate from the displayed contradiction score."""
    details = {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "evaluation_result_contract": "independent-evaluation-v1",
        "stages": {"global_scope": REPORT_REVIEW_SCOPE, "global_check": "completed"},
    }
    text = render_repair_state(
        {
            "evaluation": {
                "faithfulness": {"details": details},
                "internal_consistency": {"score": 0.8, "status": "passed"},
            }
        },
        locale,
    )["text"]
    assert "0.2 ×" in text
    assert (
        "未评测断言抽取召回率"
        if locale == "zh-CN"
        else "Extraction recall is unmeasured"
    ) in text


def test_archived_question_only_contract_does_not_gain_a_consistency_score():
    from litagent.eval.results import contract_identity

    cfg = load_config(live_workbench=True)
    cfg.eval.native_global_scope = "question-coverage-v1"
    previous = {
        **configured_engine(cfg),
        "identity": "5292f369cdc647277a0825af93ab54e9071aeef1336aa5e74cefb458a1d1c830",
    }
    assert validate_engine(previous, archived=True) == previous
    with pytest.raises(ValueError, match="identity_invalid"):
        validate_engine(
            {**previous, "global_scope": REPORT_REVIEW_SCOPE}, archived=True
        )
    assert (
        contract_identity("question-coverage-v1")
        == "938db468a626ef977d3a2274f8c50a486236147b037b9cfb349e5771fb335783"
    )
