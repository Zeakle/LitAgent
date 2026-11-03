"""Keep sent summary constraints consistent with every real host consumer."""

import copy
import json

import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.quality import SUMMARY_RULES, digest
from litagent.writing_flat import parse, prepare
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    compile_document,
    effective_summary_limits,
    prepare_planned_input,
    provider_response_format,
    refresh_summaries,
    request_messages,
    summary_request_input,
    writer_budget_plan,
)


def plan_for(case, maximum=2):
    """Build the actual v7 plan, including its first-request contract preflight."""
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        capacity=CapacityConfig(max_summary_units=maximum),
    )


@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_actual_summary_messages_schema_and_compile_share_scope(case, maximum):
    plan = plan_for(case, maximum)
    assert plan["status"] == "admitted" and plan["repair_limit"] == 2
    envelope = plan["envelope"]
    limits = effective_summary_limits(envelope)
    units = body_units(case.body)
    system, payload, schema = summary_request_input(case.packet, units, envelope)
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    sent = json.loads(messages[1]["content"])
    assert sent["summary_limits_per_section"] == sent["summary_constraints"] == limits
    assert limits["min_units"] == 1 and limits["max_units"] == maximum
    assert "target_sentence_units" not in sent["summary_limits_per_section"]
    assert "two or three findings" not in messages[0]["content"]
    assert "no requirement to fill the capacity" in messages[0]["content"]
    assert sent["writing_plan"]["summary_section_max_units"] == maximum
    provider = provider_response_format(
        "summary", schema, wire_contract="flat-writer-v1"
    )
    wire = {}
    for section in ("abstract", "conclusion"):
        field = prepare(schema)["wire_schema"]["properties"][section]
        assert field["maxItems"] == maximum and field["minItems"] == 1
        description = provider["json_schema"]["schema"]["properties"][section][
            "description"
        ]
        assert json.dumps({"maxItems": maximum, "minItems": 1}) in description
        wire[section] = [
            {
                "text": f"{section} finding {i}.",
                "body_unit_refs": ["u3"],
                "paragraph_start": True,
            }
            for i in range(maximum)
        ]
    summaries = parse(json.dumps(wire), schema)["canonical"]
    report, bindings = compile_document(
        case.body, summaries, case.cards, case.aliases, "en", envelope=envelope
    )
    assert bindings["summary_constraints"] == limits
    assert bindings["summary_constraints_sha256"] == digest(limits)
    assert (
        bindings["report_sha256"]
        == __import__("hashlib").sha256(report.encode()).hexdigest()
    )
    wire["conclusion"].append(copy.deepcopy(wire["conclusion"][0]))
    with pytest.raises(WriterError, match="writer_flat_schema_invalid"):
        parse(json.dumps(wire), schema)


def test_v5_v6_and_unmarked_scope_keep_legacy_summary_bytes(case):
    for policy in ("planned-budget-v5", "planned-budget-v6"):
        plan = writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy=policy
        )
        scope = plan["envelope"]
        assert "summary_constraints_contract" not in scope
        assert effective_summary_limits(scope) is None
        system, payload, schema = summary_request_input(
            case.packet, body_units(case.body), scope
        )
        assert payload["summary_limits_per_section"] == SUMMARY_RULES
        messages = request_messages(
            system, payload, schema, wire_contract="flat-writer-v1"
        )
        assert "two or three findings" in messages[0]["content"]


@pytest.mark.parametrize("fault", ["payload", "schema", "plan"])
def test_conflicting_actual_builders_fail_before_first_dispatch(
    case, monkeypatch, fault
):
    import litagent.writing_v2 as writer

    original = writer.summary_request_input
    original_prepare = writer.prepare_planned_input

    def conflicting(*args, **kwargs):
        system, payload, schema = original(*args, **kwargs)
        if fault == "payload":
            payload["summary_limits_per_section"]["max_units"] = 6
        if fault == "schema":
            schema["properties"]["conclusion"]["prefixItems"][0]["properties"][
                "sentences"
            ]["prefixItems"].append(
                copy.deepcopy(
                    schema["properties"]["conclusion"]["prefixItems"][0]["properties"][
                        "sentences"
                    ]["prefixItems"][0]
                )
            )
        return system, payload, schema

    def conflicting_prepare(*args, **kwargs):
        system, payload = original_prepare(*args, **kwargs)
        if fault == "plan":
            payload["writing_plan"]["summary_section_max_units"] = 6
        return system, payload

    monkeypatch.setattr(writer, "summary_request_input", conflicting)
    monkeypatch.setattr(writer, "prepare_planned_input", conflicting_prepare)
    with pytest.raises(WriterError, match="writer_summary_constraints_conflict"):
        plan_for(case)


class SummaryGateway(FlatGateway):
    """Control only provider JSON, leaving compilation and validation untouched."""

    def __init__(self, case, *, overflow=False, patch_count=1):
        super().__init__(case)
        self.overflow, self.patch_count = overflow, patch_count

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if "targets" in payload:
            original = value["patches"][0]
            value["patches"] = [
                {**copy.deepcopy(original), "text": f"Replacement finding {i}."}
                for i in range(self.patch_count)
            ]
        elif "body_units" in payload and self.overflow:
            for section, count in (("abstract", 2), ("conclusion", 3)):
                unit = value[section][0]
                value[section] = [
                    {**copy.deepcopy(unit), "text": f"{section} finding {i}."}
                    for i in range(count)
                ]
        return value


async def setup(case, tmp_path, gateway):
    owner, budget, _ = await owner_for(gateway)
    budget.activate_plan(plan_for(case), path=tmp_path / "budget.json")
    return owner, make_session(owner, case, tmp_path)


@pytest.mark.asyncio
async def test_031_response_shape_still_rejected_with_precise_durable_diagnostic(
    case, tmp_path
):
    gateway = SummaryGateway(case, overflow=True)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        with pytest.raises(WriterError, match="writer_flat_schema_invalid") as caught:
            await session.summaries(body, "summary_initial")
        diagnostic = next(
            d for d in caught.value.diagnostics if d.get("section") == "conclusion"
        )
        assert diagnostic["actual"] == 3 and diagnostic["maximum"] == 2
        assert diagnostic["stage"] == "summary"
        assert len(gateway.sent) == 6 and not session.state["attempts"]
        call = session.journal["calls"]["summary_initial"]
        assert len(json.loads(call["raw"])["conclusion"]) == 3
        assert session.journal["summary_constraints"]["max_units"] == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_post_body_refresh_reuses_complete_effective_summary_contract(
    case, tmp_path
):
    gateway = SummaryGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summary = await session.summaries(body, "summary_initial")
        changed = copy.deepcopy(body)
        changed["methods"][0]["sentences"][0][
            "text"
        ] += " Under the recorded conditions."
        refreshed = await refresh_summaries(session, body, changed, summary, {})
        payload = gateway.sent[-1]["payload"]
        assert refreshed == case.summaries
        assert payload["body_units"] == body_units(changed)
        assert payload["summary_limits_per_section"] == payload["summary_constraints"]
        assert payload["summary_constraints"]["max_units"] == 2
        assert "sources" not in payload
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_local_summary_patch_counts_unmodified_units_too(case, tmp_path):
    gateway = SummaryGateway(case, patch_count=2)
    owner, session = await setup(case, tmp_path, gateway)
    summary = copy.deepcopy(case.summaries)
    summary["abstract"][0]["sentences"].append(
        {"text": "A protected summary finding.", "body_unit_refs": ["u3"]}
    )
    before = copy.deepcopy(summary)
    target = {
        "a": {
            "path": ["abstract", 0, 0],
            "original": summary["abstract"][0]["sentences"][0],
            "explanation": "Repair only this finding.",
        }
    }
    try:
        with pytest.raises(
            WriterError, match="writer_summary_capacity_exceeded"
        ) as caught:
            await session.patch(
                summary, target, cause="evaluation", summary_body=case.body
            )
        assert caught.value.diagnostics[0]["actual"] == 3
        assert caught.value.diagnostics[0]["maximum"] == 2
        assert summary == before
        assert len(session.state["attempts"]) == 1
        assert len(gateway.sent) == 1
        assert gateway.sent[0]["payload"]["summary_constraints"]["max_units"] == 2
        assert not any(c.get("committed") for c in session.journal["candidates"])
    finally:
        await owner.close()


def test_inconsistent_provider_projection_is_rejected_before_generation(
    case, monkeypatch
):
    import litagent.writing_v2 as writer

    original = writer.provider_response_format

    def conflicting(stage, schema, **kwargs):
        result = original(stage, schema, **kwargs)
        if stage == "summary":
            result["json_schema"]["schema"]["properties"]["conclusion"][
                "description"
            ] = "Host validation requires maxItems=6."
        return result

    monkeypatch.setattr(writer, "provider_response_format", conflicting)
    with pytest.raises(
        WriterError, match="writer_summary_provider_constraints_conflict"
    ):
        plan_for(case)


def test_run_fingerprint_binds_summary_revision_only_for_v7(monkeypatch):
    import litagent.writing_v2 as writer
    from litagent.workbench import RunOptions

    current = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        review_structure="academic_v2",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    previous = current.model_copy(update={"budget_policy": "planned-budget-v6"})
    current_id, previous_id = current.fingerprint(), previous.fingerprint()
    monkeypatch.setattr(
        writer, "SUMMARY_CONSTRAINTS_CONTRACT", "changed-summary-revision"
    )
    assert current.fingerprint() != current_id
    assert previous.fingerprint() == previous_id
