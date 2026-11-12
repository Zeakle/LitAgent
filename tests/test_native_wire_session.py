"""Exercise v4 wire decoding through real SDK, journals and writer consumers."""

import asyncio
import base64
import copy
import hashlib
import json

import httpx
import pytest
from test_budgeted_writer_envelope import native_fixture_value
from test_native_responses import envelope, owner_for
from test_writer_v2 import EvaluatorTransport, case  # noqa: F401

from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest, load_targeted_repairs, strict_acceptance
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_budget import (
    envelope_budget_plan,
    make_writer_envelope,
    planned_backend,
)
from litagent.writing_native import CONTRACT
from litagent.writing_v2 import (
    BODY,
    EOF_CONTAINER_POLICY,
    NATIVE_WIRE_CONTRACT,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    body_section_schema,
    body_units,
    chapter_request_input,
    compile_document,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    schema_for,
    summary_request_input,
    writer_budget_plan,
)


def reversed_objects(value):
    """Make physical key order differ while keeping logical array order intact."""
    if isinstance(value, dict):
        return {
            key: reversed_objects(child) for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [reversed_objects(child) for child in value]
    return copy.deepcopy(value)


def complete_plan(case, *, repairs=False):
    """Use the real plan, or a smaller declared fixture path for repair consumers."""
    plan = writer_budget_plan(
        case.packet, case.aliases, budget_policy="planned-budget-v4", methodology={}
    )
    assert plan["status"] == "admitted"
    if not repairs:
        return plan
    # The actual frozen three-domain scope has no repair allowance. This local
    # fixture declares a smaller complete path and still reserves every summary,
    # inventory, support batch and global check for two shared modifications.
    scope = make_writer_envelope(
        body_units=10,
        assertions=16,
        body_bytes=12000,
        source_identity=plan["envelope"]["source_identity"],
        builder_identity=plan["envelope"]["builder_identity"],
        body_cap=50000,
        frozen_component_reservation=plan["envelope"]["frozen_component_reservation"],
    )
    scope["request_caps"].update(
        summary=60000,
        reviewer=30000,
        repair=60000,
        inventory=30000,
        assessment=30000,
        **{"global": 30000},
    )
    scope["identity"] = digest(
        {key: value for key, value in scope.items() if key != "identity"}
    )
    system, payload, schema = chapter_request_input(
        case.packet, case.aliases, BODY[0], {}, {}, RANGE_WRITING_PLAN, scope
    )
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    plan = envelope_budget_plan(
        scope,
        identity=scope["source_identity"],
        first_request={
            "messages": request_messages(
                system, payload, schema, wire_contract=CONTRACT
            ),
            "response_format": provider_response_format(
                "synthesis", schema, wire_contract=CONTRACT
            ),
        },
        repairs=2,
    )
    assert plan["status"] == "admitted" and plan["planned_tokens"] <= 1000000
    assert plan["repair_limit"] == 2 and plan["planned_requests"] <= 64
    return plan


def make_session(owner, case, tmp_path, *, role="generation"):
    """Keep actual shared accounting and the host-owned stage/model binding."""
    return WriterSession(
        ModelBoundClient(owner, "lite" if role == "evaluation" else "pro", role),
        tmp_path / "run.json",
        "native-wire-test",
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )


class Gateway:
    """Replace external HTTP only; all writer and evaluator consumers remain real."""

    def __init__(self, case, *, fault=None, declined=False, unsupported=False):
        """Keep controlled scientific and protocol failures separate."""
        self.case = case
        self.fault = fault
        self.declined = declined
        self.evaluator = EvaluatorTransport(unsupported=unsupported)
        self.sent = []

    def answer(self, payload, native_schema):
        """Return canonical mock model values before following the requested wire."""
        if "targets" in payload:
            summary = "body_units" in payload
            field = "body_unit_refs" if summary else "fact_card_refs"
            return {
                "outcome": "declined" if self.declined else "patched",
                "explanation": "Controlled source-bound response.",
                "patches": (
                    []
                    if self.declined
                    else [
                        {
                            "target_id": tid,
                            "sentences": [
                                {
                                    "text": (
                                        "A revised source-bound abstract finding."
                                        if summary
                                        else "A revised source-bound relation-network comparison."
                                    ),
                                    field: ["u3" if summary else self.case.by_eid["b"]],
                                }
                            ],
                        }
                        for tid in payload["targets"]
                    ]
                ),
            }
        if "body_units" in payload:
            return copy.deepcopy(self.case.summaries)
        if "report_layout" in payload or "draft" in payload:
            return {"issues": [], "explanation": "No controlled review issue."}
        if "requested_body_section" in payload:
            return {
                key: copy.deepcopy(self.case.body[key])
                for key in native_schema["properties"]
            }
        return {
            key: copy.deepcopy(self.case.body[key])
            for key in native_schema["properties"]
        }

    async def __call__(self, request):
        """Serialize real SDK requests and return provider envelopes with usage."""
        body = json.loads(request.content)
        native = "input" in body
        messages = copy.deepcopy(body["input"] if native else body["messages"])
        rf = (
            {
                "type": "json_schema",
                "json_schema": {
                    key: value
                    for key, value in body["text"]["format"].items()
                    if key != "type"
                },
            }
            if native
            else body["response_format"]
        )
        payload = json.loads(messages[1]["content"])
        record = {
            "request": body,
            "request_bytes": request.content,
            "payload": copy.deepcopy(payload),
        }
        self.sent.append(record)
        if self.fault == "cancel":
            raise asyncio.CancelledError()
        if self.fault == "timeout":
            raise httpx.ReadTimeout("controlled timeout", request=request)
        is_evaluation = any(
            key in payload
            for key in (
                "allowed_citation_aliases",
                "cited_materials",
                "report_structure",
                "global_output_contract",
            )
        )
        if is_evaluation:
            # The old semantic fixture consumes the losslessly restored alias
            # view. This happens inside the mock provider, never the application.
            if (
                payload.get("assessment_input_contract")
                and "cited_materials" in payload
            ):
                for claim in payload["assertions"]:
                    if "evidence_aliases" in claim:
                        claim["evidence"] = [
                            {"evidence_alias": ref, "material_ref": ref}
                            for ref in claim.pop("evidence_aliases")
                        ]
                messages[1]["content"] = json.dumps(payload)
            response = await self.evaluator.chat(messages, response_format=rf)
            return httpx.Response(
                200,
                json={
                    "id": "controlled-evaluation",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": response.content,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 100,
                        "total_tokens": 120,
                    },
                },
            )
        assert native and body["thinking"] == {"type": "disabled"}
        assert "reasoning" not in body
        value = self.answer(payload, rf["json_schema"]["schema"])
        if self.fault == "surrogate":
            value["methods"][0]["sentences"][0]["text"] = "Malformed \ud800 value."
        wire = native_fixture_value(value, rf["json_schema"]["schema"])
        raw = json.dumps(reversed_objects(wire), ensure_ascii=False)
        if self.fault == "outer_eof":
            raw = raw[:-1]
        elif self.fault == "unit_eof":
            # Reversed object keys put the text string last, so this mock EOF
            # removes the unit's own closure as well as outer layout closures.
            raw = raw[: raw.rfind('"') + 1]
        provider = envelope(raw, model=body["model"])
        provider["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
        if self.fault == "missing_usage":
            provider.pop("usage")
        response = (
            httpx.Response(
                200, content=json.dumps(provider, ensure_ascii=True).encode("ascii")
            )
            if self.fault == "surrogate"
            else httpx.Response(200, json=provider)
        )
        record.update(raw=raw, canonical=value, response_bytes=response.content)
        return response


async def setup(case, tmp_path, gateway, *, repairs=False):
    """Activate a real complete plan without replacing the SDK or consumers."""
    owner, budget, events = await owner_for(gateway)
    budget.activate_plan(
        complete_plan(case, repairs=repairs), path=tmp_path / "ledger.json"
    )
    session = make_session(owner, case, tmp_path)
    assert session.wire_contract == CONTRACT == NATIVE_WIRE_CONTRACT
    return owner, budget, events, session


def assert_saved_wire(session, gateway, name, expected):
    """Verify raw receipt, canonical identity and original character bindings."""
    entry = session.journal["calls"][name]
    observed = next(row for row in gateway.sent if row.get("raw") == entry["raw"])
    assert entry["writer_wire_contract"] == CONTRACT
    assert entry["canonical_sha256"] == digest(expected)
    assert entry["raw"] == observed["raw"]
    assert entry["raw_sha256"] == hashlib.sha256(observed["raw"].encode()).hexdigest()
    receipt = entry["provider_receipt"]
    assert base64.b64decode(receipt["request_body_base64"]) == observed["request_bytes"]
    assert (
        base64.b64decode(receipt["response_body_base64"]) == observed["response_bytes"]
    )
    binding = entry["wire_bindings"]
    assert binding["coverage_complete"] is True
    assert binding["scientific_assessment"] == "not_verified"
    assert binding["raw_sha256"] == entry["raw_sha256"]
    for unit in binding["units"]:
        assert unit["end"] <= len(entry["raw"])
        assert entry["raw"][
            unit["text_span"]["start"] : unit["text_span"]["end"]
        ].startswith('"')
    assert entry["reservation"] == strict_request_reservation(
        observed["request"]["input"],
        entry["max_tokens"],
        {
            "type": "json_schema",
            "json_schema": {
                key: value
                for key, value in observed["request"]["text"]["format"].items()
                if key != "type"
            },
        },
    )


@pytest.mark.asyncio
async def test_real_chapter_summary_and_reviewer_calls_preserve_canonical_and_receipts(
    case, tmp_path
):
    """A complete generation chain never turns structural validity into science."""
    gateway = Gateway(case)
    owner, budget, _, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        assert body == case.body
        summaries = await session.summaries(body, "summary_initial")
        assert summaries == case.summaries
        for section in BODY:
            expected = {section: case.body[section]}
            if section == BODY[0]:
                expected["title"] = case.body["title"]
            assert_saved_wire(session, gateway, "body:" + section, expected)
        assert_saved_wire(session, gateway, "summary_initial", summaries)
        reviewer = make_session(owner, case, tmp_path, role="evaluation")
        schema = {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 128,
                },
                "explanation": {"type": "string", "minLength": 1},
            },
            "required": ["issues", "explanation"],
            "additionalProperties": False,
        }
        value = await reviewer.call(
            "reviewer",
            "Review source-bound writing.",
            {"report_layout": []},
            schema,
            stage="reviewer",
            max_tokens=4096,
        )
        assert value["issues"] == []
        assert_saved_wire(reviewer, gateway, "reviewer", value)
        assert len(gateway.sent) == budget.snapshot()["requests_sent"] == 7
        assert budget.used == 7 * 120 and budget.snapshot()["usage_uncertain"] is False
        assert reviewer.journal["writer_wire_contract"] == CONTRACT
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [False, True])
async def test_actual_patch_preflight_and_response_share_the_wire_contract(
    case, tmp_path, summary
):
    """Body and summary patches retain exact authorization and unaffected text."""
    gateway = Gateway(case)
    owner, budget, _, session = await setup(case, tmp_path, gateway, repairs=True)
    document = copy.deepcopy(case.summaries if summary else case.body)
    section = "abstract" if summary else "methods"
    target = {
        "authorized": {
            "path": [section, 0, 0],
            "original": copy.deepcopy(document[section][0]["sentences"][0]),
        }
    }
    original = copy.deepcopy(document)
    try:
        candidate = await session.patch(
            document,
            target,
            cause="evaluation",
            claim_count=7,
            summary_body=case.body if summary else None,
        )
        assert document == original
        assert candidate[section] != original[section]
        for key in original:
            if key != section:
                assert candidate[key] == original[key]
        calls = [
            name for name in session.journal["calls"] if name.startswith("repair_")
        ]
        assert len(calls) == 1
        assert_saved_wire(session, gateway, calls[0], gateway.sent[0]["canonical"])
        assert (
            len(session.state["attempts"]) == 1
            and session.state["attempts"][0]["status"] == "completed"
        )
        assert len(gateway.sent) == budget.snapshot()["requests_sent"] == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_declined_empty_patches_are_quality_terminal_and_not_replayed(
    case, tmp_path
):
    """A valid empty refusal remains a quality result, not a format repair loop."""
    gateway = Gateway(case, declined=True)
    owner, budget, _, session = await setup(case, tmp_path, gateway, repairs=True)
    target = {
        "authorized": {
            "path": ["methods", 0, 0],
            "original": copy.deepcopy(case.body["methods"][0]["sentences"][0]),
        }
    }
    try:
        with pytest.raises(WriterError, match="writer_repair_declined") as caught:
            await session.patch(case.body, target, cause="evaluation", claim_count=7)
        assert caught.value.family == "quality"
        assert gateway.sent[0]["canonical"]["patches"] == []
        name = next(iter(session.journal["calls"]))
        assert_saved_wire(session, gateway, name, gateway.sent[0]["canonical"])
        assert len(load_targeted_repairs(session.path, session.run_id)["attempts"]) == 1
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.call(
                name,
                "Never resend.",
                {},
                schema_for("repair", case.aliases, target),
                stage="repair",
            )
        assert len(gateway.sent) == 1 and budget.used == 120
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [False, True])
async def test_only_first_body_and_summary_eof_can_close_layout(
    case, tmp_path, summary
):
    """Container recovery preserves every original unit and its received spans."""
    gateway = Gateway(case, fault="outer_eof")
    owner, budget, _, session = await setup(case, tmp_path, gateway)
    try:
        if summary:
            result = await session.summaries(case.body, "summary_initial")
            expected, name = case.summaries, "summary_initial"
        else:
            schema = body_section_schema("methods", case.aliases, maximum=3)
            result = await session.call(
                "body:methods", "Write one chapter.", case.packet, schema
            )
            expected, name = {"methods": case.body["methods"]}, "body:methods"
        assert result == expected
        assert_saved_wire(session, gateway, name, expected)
        entry = session.journal["calls"][name]
        assert entry["normalized_raw"] == entry["raw"] + "}"
        assert entry["syntax_normalization"]["scientific_assessment"] == "not_verified"
        assert entry["syntax_normalization"]["coverage_complete"] is True
        assert budget.used == 120 and len(gateway.sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["summary_update", "repair", "reviewer"])
async def test_updated_summary_repair_and_reviewer_eof_do_not_gain_recovery(
    case, tmp_path, kind
):
    """The wire helper cannot widen the real Session's stage/name authorization."""
    gateway = Gateway(case, fault="outer_eof")
    owner, budget, _, session = await setup(case, tmp_path, gateway, repairs=True)
    try:
        if kind == "summary_update":
            system, payload, schema = summary_request_input(
                case.packet, body_units(case.body), session.envelope
            )
            name, stage, cap = "summary_after_repair", "summary", 20000
        elif kind == "repair":
            schema = schema_for("repair", case.aliases, ["authorized"])
            payload = {"targets": {"authorized": {}}, "query": case.packet["query"]}
            system, name, stage, cap = (
                "Repair only authorized text.",
                "repair_0",
                "repair",
                20000,
            )
        else:
            schema = {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["issues", "explanation"],
                "additionalProperties": False,
            }
            payload = {"report_layout": []}
            system, name, stage, cap = (
                "Review the frozen draft.",
                "reviewer",
                "reviewer",
                4096,
            )
        with pytest.raises(WriterError, match="writer_json_invalid"):
            await session.call(
                name, system, payload, schema, stage=stage, max_tokens=cap
            )
        entry = session.journal["calls"][name]
        assert "syntax_normalization" not in entry and "normalized_raw" not in entry
        assert entry["raw"] == gateway.sent[0]["raw"]
        assert entry["usage"]["usage_available"] is True
        assert budget.used == 120 and len(gateway.sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_eof_cannot_invent_a_scientific_unit_closure(case, tmp_path):
    """Even a schema-valid closed candidate cannot replace an incomplete raw unit."""
    gateway = Gateway(case, fault="unit_eof")
    owner, budget, _, session = await setup(case, tmp_path, gateway)
    schema = body_section_schema("methods", case.aliases, maximum=1)
    try:
        with pytest.raises(WriterError, match="writer_json_invalid"):
            await session.call(
                "body:methods", "Write one chapter.", case.packet, schema
            )
        entry = session.journal["calls"]["body:methods"]
        assert "syntax_normalization" not in entry
        assert entry["syntax_normalization_rejected"]["code"].endswith(
            "original_unit_incomplete"
        )
        assert entry["raw"] == gateway.sent[0]["raw"] and budget.used == 120
        assert len(gateway.sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_usage", "cancel", "timeout"])
async def test_unknown_responses_keep_reservation_and_never_restart_requests(
    case, tmp_path, fault
):
    """Cancellation and unknown usage keep a durable sent hold without retries."""
    gateway = Gateway(case, fault=fault)
    owner, budget, _, session = await setup(case, tmp_path, gateway)
    schema = body_section_schema("methods", case.aliases, maximum=1)
    try:
        expected = asyncio.CancelledError if fault == "cancel" else Exception
        with pytest.raises(expected):
            await session.call("body:methods", "Write once.", case.packet, schema)
        snapshot = budget.snapshot()
        assert len(gateway.sent) == snapshot["requests_sent"] == 1
        assert snapshot["usage_uncertain"] is True and budget.used == 0
        entry = next(iter(snapshot["entries"].values()))
        assert entry["sent"] is True and entry["status"] != "settled"
        assert entry["reserved_tokens"] > 20000
        restored = CostBudget.from_snapshot(snapshot)
        assert restored.snapshot() == snapshot
        owner._cost_budget = restored
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.call("body:methods", "Never retry.", case.packet, schema)
        with pytest.raises(ValueError, match="unknown"):
            await restarted.call(
                "body:taxonomy",
                "Do not continue.",
                case.packet,
                body_section_schema("taxonomy", case.aliases, maximum=1),
            )
        assert len(gateway.sent) == 1 and restored.snapshot() == snapshot
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_wire_identity_conflict_rejects_before_journal_write_or_http(
    case, tmp_path, monkeypatch
):
    """A version change cannot reinterpret a same-path frozen journal."""
    import litagent.writing_v2 as writer

    gateway = Gateway(case)
    owner, _, _, session = await setup(case, tmp_path, gateway)
    try:
        session.save()
        before = session.journal_path.read_bytes()
        monkeypatch.setattr(writer, "NATIVE_WIRE_CONTRACT", "future-wire-version")
        with pytest.raises(WriterError, match="writer_checkpoint_identity_conflict"):
            make_session(owner, case, tmp_path)
        assert session.journal_path.read_bytes() == before
        assert gateway.sent == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_invalid_unicode_keeps_known_usage_receipt_and_raw_before_rejection(
    case, tmp_path
):
    """A JSON-escaped lone surrogate cannot erase an already received bill."""
    gateway = Gateway(case, fault="surrogate")
    owner, budget, _, session = await setup(case, tmp_path, gateway)
    schema = body_section_schema("methods", case.aliases, maximum=1)
    try:
        with pytest.raises(WriterError, match="writer_native_wire_raw_unicode_invalid"):
            await session.call(
                "body:methods", "Write one chapter.", case.packet, schema
            )
        entry = session.journal["calls"]["body:methods"]
        assert entry["raw"] == gateway.sent[0]["raw"] and "\ud800" in entry["raw"]
        assert entry["raw_sha256"] is None
        assert (
            entry["raw_json_sha256"]
            == hashlib.sha256(
                json.dumps(entry["raw"], ensure_ascii=True).encode("ascii")
            ).hexdigest()
        )
        assert entry["usage"]["usage_available"] is True
        assert (
            base64.b64decode(entry["provider_receipt"]["response_body_base64"])
            == gateway.sent[0]["response_bytes"]
        )
        persisted = json.loads(session.journal_path.read_bytes())
        assert persisted["calls"]["body:methods"]["raw"] == entry["raw"]
        assert budget.used == 120 and budget.snapshot()["usage_uncertain"] is False
        assert len(gateway.sent) == budget.snapshot()["requests_sent"] == 1
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.call(
                "body:methods", "Never resend invalid Unicode.", case.packet, schema
            )
        assert len(gateway.sent) == 1 and budget.used == 120
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_wire_success_does_not_override_real_scientific_evaluation(
    case, tmp_path
):
    """Run the SDK-backed evaluator after canonical compilation; unsupported blocks."""
    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped

    gateway = Gateway(case, unsupported=True)
    owner, _, _, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summaries = await session.summaries(body, "summary_initial")
        survey, bindings = compile_document(
            body,
            summaries,
            case.cards,
            case.aliases,
            "en",
            writing_plan=RANGE_WRITING_PLAN,
            envelope=session.envelope,
        )
        context = {
            "query": case.packet["query"],
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "report_language": "en",
            "review_structure": "academic_v2",
            "writing_bindings": bindings,
            "scope_notice_contract": "scope-notice-v1",
            "fact_cards": case.cards,
            "evidence_snapshot_sha256": digest(case.cards),
            "evidence": case.ledger,
            "coverage_requirements": case.packet["requirements"],
            "structure_checks": {"status": "passed"},
            "method_checks": {"status": "not_applicable"},
            "language_checks": {"status": "passed"},
        }
        details = await evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=survey,
            context=context,
            config=EvalConfig(),
        )
        assert details["coverage"]["complete"] is True
        assert details["unsupported_count"] > 0
        assert strict_acceptance(details)["status"] == "failed"
        assert all(
            call["writer_wire_contract"] == CONTRACT
            for call in session.journal["calls"].values()
        )
        assert any("cited_materials" in row["payload"] for row in gateway.sent)
        assert (
            planned_backend(owner._cost_budget).snapshot()["usage_uncertain"] is False
        )
    finally:
        await owner.close()
