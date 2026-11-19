"""Exercise candidate-stage repair with real consumers and mocked transport."""

import copy

import pytest
from test_inline_citations import InlineGateway
from test_separate_review_pipeline import execute
from test_writer_v2 import case  # noqa: F401

from litagent.inline_citations import compile_text, references
from litagent.stage_repair import SCOPE_CONTRACT
from litagent.writing_v2 import WriterError


def stage_config():
    from litagent.config import load_config

    return load_config("config/llm-stage-repair.yaml")


def test_marker_ends_range_without_borrowing_tail():
    text = "减少token[C:c15]、处理冗余[C:c18]以及尾句。无引用句。"
    rendered, spans = compile_text(
        text, {"c15": "a", "c18": "b"}, scope_contract=SCOPE_CONTRACT
    )
    assert [s["evidence_ids"] for s in spans] == [["a"], ["b"], [], []]
    assert "".join(s["source_quote"] for s in spans) == rendered
    with pytest.raises(WriterError):
        references(text)
    with pytest.raises(WriterError, match="without_prose"):
        references("无引文句。[C:c15]", scope_contract=SCOPE_CONTRACT)


def test_guidance_replay_reenters_frozen_citation_policy(monkeypatch):
    from litagent import stage_repair, writing_v2

    seen = []
    monkeypatch.setattr(writing_v2, "repair_prompt_fields", lambda envelope: {})

    def consume(envelope, record, **kwargs):
        seen.append(stage_repair.current_scope())
        return compile_text(
            "机制[C:c1]以及后文。",
            {"c1": "e1"},
            scope_contract=stage_repair.current_scope(),
        )

    monkeypatch.setattr(writing_v2, "check_guidance_wire_record", consume)
    envelope = {"run_limits": {"candidate_recovery": stage_repair.CONTRACT}}
    _, scopes = writing_v2.check_repair_guidance_record(envelope, {})
    assert seen == [SCOPE_CONTRACT]
    assert [row["evidence_ids"] for row in scopes] == [["e1"], []]
    assert stage_repair.current_scope() is None


@pytest.mark.asyncio
async def test_guidance_replay_accepts_marker_before_connective(
    case, tmp_path, monkeypatch
):
    class Gateway(InlineGateway):
        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            for unit in result.get("units", []) + result.get("patches", []):
                if "[C:" in unit.get("text", ""):
                    unit["text"] = unit["text"].removesuffix(".") + "以及后续说明。"
            return result

    gateway = Gateway(case, reviews=1)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.editorial_edits == 1
    assert gateway.global_calls >= 1, report["metadata"].get("writer_failure")
    # The mock classifies the uncited tail as scientific: accepting its syntax
    # must not allow that tail to borrow the preceding source or pass delivery.
    assert report["metadata"]["writer_failure"]["code"] == "writer_repair_no_progress"
    assert not report["delivery"]["publishable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_conflicting_response_uses_remaining_model_attempt(
    case, tmp_path, monkeypatch, category
):
    class Gateway(InlineGateway):
        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            if "target_responses" in schema.get("properties", {}):
                first = (
                    self.editorial_edits == 1
                    if category == "review"
                    else self.repair_count == 1
                )
                if first:
                    key = result["patches"][0]["target_id"]
                    result["target_responses"][key]["action"] = "keep"
            return result

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.global_calls >= 1
    assert (
        gateway.editorial_edits == 2
        if category == "review"
        else gateway.repair_count == 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_summary_failure_only_regenerates_summary(
    case, tmp_path, monkeypatch, category
):
    class Gateway(InlineGateway):
        bad_summaries = 0
        feedback = None

        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            props = schema.get("properties", {})
            ready = (
                self.editorial_edits > 0
                if category == "review"
                else self.repair_count > 0
            )
            if "abstract" in props and ready and self.bad_summaries == 0:
                self.bad_summaries += 1
                result["abstract"][0]["text"] += "[C:c99998]"
                result["conclusion"][0]["text"] += " A false source[C:c99999]."
            elif (
                "abstract" in props
                and ready
                and payload.get("candidate_feedback", {}).get("category")
                == f"{category}_revision"
            ):
                self.feedback = payload["candidate_feedback"]
            return result

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.bad_summaries == 1
    assert gateway.feedback["diagnostics_complete"]
    assert {d.get("reference") for d in gateway.feedback["diagnostics"]} >= {
        "c99998",
        "c99999",
    }
    assert all("wire_path" in d for d in gateway.feedback["diagnostics"])
    assert gateway.global_calls >= 1
    assert (
        gateway.editorial_edits == 1
        if category == "review"
        else gateway.repair_count == 1
    )
    import json

    journals = list((tmp_path / "artifacts").rglob("*.writer-v2.json"))
    states = [json.loads(p.read_text()) for p in journals]
    journal = next(j for j in states if j.get("candidate_rejections"))
    assert journal["candidate_rejections"]
    assert not journal.get("pending_candidate")
    assert (
        len([key for key in journal["calls"] if key.startswith("summary_repair_")]) >= 2
    )


def test_vit10_t21_keeps_original_response_and_local_evidence():
    import json
    from pathlib import Path

    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/vit-235d5bf1dba2-inline-reassessment-10/repairs.json.writer-v2.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT 10")
    before = path.read_bytes()
    response = json.loads(json.loads(before)["calls"]["repair_2"]["raw"])
    text = next(p["text"] for p in response["patches"] if p["target_id"] == "t21")
    with pytest.raises(WriterError):
        references(text)
    refs = references(text, scope_contract=SCOPE_CONTRACT)
    rendered, ranges = compile_text(
        text, {r: r for r in refs}, scope_contract=SCOPE_CONTRACT
    )
    assert "".join(row["source_quote"] for row in ranges) == rendered
    assert any(row["evidence_ids"] == ["c5", "c15"] for row in ranges)
    assert ranges[-1]["evidence_ids"] == []
    assert ranges[-1]["source_quote"].startswith("对应的数据效率")
    assert path.read_bytes() == before


def test_feedback_and_public_rejection_keep_raw_material_private():
    from litagent.stage_repair import feedback_payload, make_feedback, public_rejection

    record = dict(
        contract="llm-stage-repair-v1",
        category="evaluation_revision",
        stage="summary",
        action="repair_summary",
        reason_code="new_candidate_check",
        call="summary_repair_2",
        retained_report_sha256="a" * 64,
        remaining=1,
        actual_tokens=123,
        body_sha256="b" * 64,
        raw_sha256="c" * 64,
        diagnostics=[{"path": ["conclusion", 0], "actual": "private" * 2000}],
    )
    raw = '正文和引用[C:c3]。"' * 2000
    feedback = make_feedback(record, raw)
    assert raw.startswith(feedback["failed_candidate_excerpt"])
    assert not feedback["excerpt_complete"]
    assert feedback["diagnostics_total"] == 1
    assert feedback["diagnostics_included"] == 0
    assert not feedback["diagnostics_complete"]
    feedback_payload({"candidate_feedback": feedback})
    public = public_rejection(record)
    assert public["remaining"] == 1
    assert "diagnostics" not in public and "raw_sha256" not in public


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_target", "missing_response"])
async def test_candidate_validator_does_not_need_error_code_retry_branch(
    case, tmp_path, monkeypatch, fault
):
    class Gateway(InlineGateway):
        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            if (
                "target_responses" in schema.get("properties", {})
                and self.editorial_edits == 1
            ):
                if fault == "unknown_target":
                    result["patches"][0]["target_id"] = "t999999"
                else:
                    result["target_responses"].pop(
                        next(iter(result["target_responses"]))
                    )
            return result

    gateway = Gateway(case, reviews=1)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.editorial_edits == 2
    assert gateway.global_calls >= 1
    assert (
        report["metadata"]["editorial_history"][0]["patch_rejection"]["action"]
        == "repair_response"
    )


@pytest.mark.asyncio
async def test_format_recovery_does_not_bypass_semantic_candidate_repair(
    case, tmp_path, monkeypatch
):
    class Gateway(InlineGateway):
        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            if (
                "target_responses" in schema.get("properties", {})
                and self.editorial_edits == 1
            ):
                result["target_responses"][result["patches"][0]["target_id"]][
                    "action"
                ] = "keep"
            return result

    gateway = Gateway(case, reviews=1, format_fault="extra_brace")
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.format_calls == 1
    assert gateway.editorial_edits == 2
    assert gateway.global_calls >= 1
    assert (
        report["metadata"]["editorial_history"][0]["patch_rejection"]["action"]
        == "repair_response"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_repeated_summary_repair_never_exceeds_its_three_attempts(
    case, tmp_path, monkeypatch, category
):
    class Gateway(InlineGateway):
        invalid_summaries = 0

        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            ready = (
                self.editorial_edits > 0
                if category == "review"
                else self.repair_count > 0
            )
            if "abstract" in schema.get("properties", {}) and ready:
                self.invalid_summaries += 1
                result["conclusion"][0][
                    "text"
                ] += f" Failure {self.invalid_summaries}[C:c99999]."
            return result

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    import json

    journal = next(
        json.loads(p.read_text())
        for p in (tmp_path / "artifacts").rglob("*.writer-v2.json")
        if "candidate_rejections" in p.read_text()
    )
    category_key = "review_revision" if category == "review" else "evaluation_revision"
    records = [
        r
        for r in journal["candidate_rejections"].values()
        if r["category"] == category_key
    ]
    assert len(records) == 3
    assert [r["remaining"] for r in records] == [2, 1, 0]
    assert records[-1]["action"] == "stop"
    assert not journal.get("pending_candidate")
    assert not report["delivery"]["publishable"]


@pytest.mark.asyncio
async def test_unknown_usage_is_not_candidate_repair(case, tmp_path, monkeypatch):
    import json

    import httpx

    class Gateway(InlineGateway):
        async def __call__(self, request):
            response = await super().__call__(request)
            if request.url.path.endswith("/responses") and self.editorial_edits:
                wire = json.loads(request.content)
                if "abstract" in wire["text"]["format"]["schema"]["properties"]:
                    value = response.json()
                    value.pop("usage")
                    return httpx.Response(200, json=value)
            return response

    gateway = Gateway(case, reviews=1)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stage_config())
    assert gateway.editorial_edits == 1
    assert gateway.global_calls == 0
    assert not report["delivery"]["publishable"]


def test_full_locked_budget_includes_new_feedback_and_all_existing_loops(case):
    from test_separate_review_budget import (
        test_all_independent_maxima_and_format_fallback_fit as verify_maxima,
    )

    from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
    from litagent.output_capacity import configured_output_tokens
    from litagent.run_limits import configured_run_limits

    cfg = stage_config()
    plan = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        methodology={},
        semantic_engine=configured_engine(cfg),
        capacity=cfg.capacity,
        run_limits=configured_run_limits(cfg),
        output_tokens=configured_output_tokens(cfg),
    )
    assert plan["status"] == "admitted", plan
    assert plan["envelope"]["run_limits"]["candidate_recovery"] == "llm-stage-repair-v1"
    verify_maxima(plan)


def test_pending_summary_with_unconsumed_later_attempt_cannot_send():
    from types import SimpleNamespace

    from litagent.quality import digest
    from litagent.stage_repair import resume_body

    pending = dict(
        category="review_revision",
        parent_body_sha256=digest({}),
        body={},
        body_sha256=digest({}),
        targets={},
        rejection_call="summary_repair_1",
    )
    pending["identity"] = digest(pending)
    session = SimpleNamespace(
        journal={"pending_candidate": pending}, state={"attempts": [{}, {}]}
    )
    with pytest.raises(ValueError, match="pending_summary_result_unconsumed"):
        resume_body(session, {}, {}, category="review_revision")


def test_previous_stage_identity_is_read_only_and_preserved():
    from litagent.run_limits import (
        ARCHIVED_STAGE_REPAIR_IDENTITIES,
        configured_run_limits,
        validate_run_limits,
    )

    old = configured_run_limits(stage_config())
    keys = ("candidate_recovery_identity", "citation_identity", "workflow_identity")
    old.update(zip(keys, ARCHIVED_STAGE_REPAIR_IDENTITIES))
    assert validate_run_limits(old, archived=True) == old
    with pytest.raises(ValueError, match="stage_repair_identity_invalid"):
        validate_run_limits(old)
    for key in keys:
        corrupt = {**old, key: "0" * 64}
        with pytest.raises(ValueError):
            validate_run_limits(corrupt, archived=True)


def test_vit11_summary_feedback_keeps_every_locatable_issue():
    import hashlib
    import json
    from pathlib import Path

    from litagent.stage_repair import make_feedback, summary_diagnostics

    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-11/repairs.json.writer-v2.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT 11")
    original = path.read_bytes()
    journal = json.loads(original)
    expected = {1: 11, 2: 2, 3: 2, 5: 10, 6: 11}
    for index, count in expected.items():
        name = f"summary_repair_{index}"
        call = journal["calls"][name]
        findings = summary_diagnostics(
            call["raw"],
            call["candidate_request"],
            {f"c{i}" for i in range(1, 30)},
            scope_contract=SCOPE_CONTRACT,
        )
        assert len(findings) == count
        assert all(len(d["wire_path"]) == 3 for d in findings)
        record = {**journal["candidate_rejections"][name], "diagnostics": findings}
        feedback = make_feedback(record, call["raw"])
        assert feedback["diagnostics_included"] == count
        assert feedback["diagnostics_complete"]
        assert record["raw_sha256"] == hashlib.sha256(call["raw"].encode()).hexdigest()
        if index == 3:
            assert {d["reference"]: d["reason"] for d in findings} == {
                "c29": "outside_body_dependencies",
                "c48": "unknown_frozen_card",
            }
        # A connective after a citation is legal in the new frozen scope.
        wire = json.loads(call["raw"])
        payload = json.loads(call["candidate_request"]["messages"][1]["content"])
        uid, unit = next(
            (u, v) for u, v in payload["body_units"].items() if v["fact_card_refs"]
        )
        for section in ("abstract", "conclusion"):
            for item in wire[section]:
                item.update(
                    text=f"机制[C:{unit['fact_card_refs'][0]}]以及未引用尾句。",
                    body_unit_refs=[uid],
                )
        assert not summary_diagnostics(
            json.dumps(wire, ensure_ascii=False),
            call["candidate_request"],
            {f"c{i}" for i in range(1, 30)},
            scope_contract=SCOPE_CONTRACT,
        )
    assert path.read_bytes() == original
