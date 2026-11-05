"""Keep complete summary premises within their locked request allowance."""

import copy
import json

import pytest
from test_evaluation_dimensions_identity import dimension_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.results import CONTRACT
from litagent.output_capacity import configured_output_tokens, envelope_output_tokens
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import check_writer_envelope, nested_json_size
from litagent.writer_source_view import (
    selected_summary_body_view,
    summary_body_view,
    validate_summary_body_view,
)
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    compile_document,
    preflight_summary_contract,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    summary_request_input,
    summary_request_upper_bound,
    validate_body,
)


def current_scope():
    """Select the actual independent writer stack with expanded output capacity."""
    return dimension_envelope(
        writer_source_view="source-only-writer-v1",
        evaluation_result_contract=CONTRACT,
        output_tokens=configured_output_tokens(
            load_config("config/expanded-capacity.yaml")
        ),
    )


def actual_request(case, units, scope):
    """Construct exactly the final messages and native schema sent by the writer."""
    system, payload, schema = summary_request_input(case.packet, units, scope)
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    native = provider_response_format("summary", schema, wire_contract="flat-writer-v1")
    return (
        system,
        payload,
        strict_request_reservation(
            messages, envelope_output_tokens(scope)["summary"], native
        ),
    )


def test_prose_and_paragraphs_are_lossless_and_host_citations_stay_local(case):
    scope = current_scope()
    body = copy.deepcopy(case.body)
    unit = copy.deepcopy(body[BODY[0]][0]["sentences"][0])
    body[BODY[0]][0]["sentences"].append(unit)
    body[BODY[0]].append({"sentences": [copy.deepcopy(unit)]})
    units = body_units(body)
    before = copy.deepcopy(units)
    view = summary_body_view(units, scope)
    assert view["body_sections"][BODY[0]] == [["u1", "u2"], ["u3"]]
    assert view["body_units"] == {uid: u["text"] for uid, u in units.items()}
    assert units == before
    system, payload, reservation = actual_request(case, units, scope)
    assert payload["body_units"] == view["body_units"]
    assert "writing_plan" not in payload
    assert "HOST WRITING PLAN" not in system
    assert "cards" not in payload and "sources" not in payload
    assert reservation <= scope["request_caps"]["summary"]
    # Rendering still uses the complete original catalog, not the model view.
    summaries = copy.deepcopy(case.summaries)
    for section in summaries:
        summaries[section][0]["sentences"][0]["body_unit_refs"] = ["u1", "u3"]
    report, binding = compile_document(
        body,
        summaries,
        case.cards,
        case.aliases,
        "en",
        {},
        writing_plan=RANGE_WRITING_PLAN,
        envelope=scope,
    )
    assert "[E:b]" in report
    assert binding["report_sha256"]


def test_unselected_summary_retains_original_catalog_and_old_writing_plan(case):
    scope = dimension_envelope(writer_source_view="source-only-writer-v1")
    units = body_units(case.body)
    assert not selected_summary_body_view(scope)
    assert summary_body_view(units, scope) is None
    system, payload, _ = actual_request(case, units, scope)
    assert payload["body_units"] == units
    assert "body_sections" not in payload and "writing_plan" in payload
    assert "HOST WRITING PLAN" in system


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "unknown", "order", "text", "type"]
)
def test_summary_partition_cannot_hide_change_or_borrow_a_body_unit(case, fault):
    scope = current_scope()
    view = summary_body_view(body_units(case.body), scope)
    groups = view["body_sections"][BODY[0]]
    if fault == "missing":
        groups[0].clear()
    elif fault == "duplicate":
        groups[0].append("u1")
    elif fault == "unknown":
        groups[0][0] = "u999"
    elif fault == "order":
        view["body_units"] = dict(reversed(view["body_units"].items()))
    elif fault == "text":
        view["body_units"]["u1"] = ""
    else:
        view["body_units"]["u1"] = {"text": "wrong shape"}
    with pytest.raises(ValueError, match="writer_source_view_invalid"):
        validate_summary_body_view(view, scope)


@pytest.mark.parametrize("paragraphs", ["one", "many"])
@pytest.mark.parametrize("padding", ["x", '"', "中"])
def test_complete_escaping_and_sparse_references_fit_the_proven_bound(
    case, paragraphs, padding
):
    scope = current_scope()
    proof = summary_request_upper_bound(
        case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    assert len(proof["rows"]) == scope["body_max_units"] - len(BODY) + 1
    assert proof["reservation"] <= 70000
    for count in (16, 48, 66):
        body = {"title": "x", **{key: [{"sentences": []}] for key in BODY}}
        limits = dict(zip(BODY, (8, 10, 24, 16, 14)))
        counts = dict.fromkeys(BODY, 1)
        for _ in range(count - len(BODY)):
            key = next(key for key in BODY if counts[key] < limits[key])
            counts[key] += 1
        for section in BODY:
            for _ in range(counts[section]):
                unit = {
                    "text": 'A quoted "condition" and a backslash \\ remain. ',
                    "fact_card_refs": [],
                }
                if paragraphs == "many" and body[section][0]["sentences"]:
                    body[section].append({"sentences": [unit]})
                else:
                    body[section][0]["sentences"].append(unit)
        # Exact body byte ceiling, spread across units without exceeding chars.
        remaining = scope["body_json_bytes"] - nested_json_size(body)
        cost = nested_json_size("x" + padding) - nested_json_size("x")
        for key in BODY:
            for paragraph in body[key]:
                for unit in paragraph["sentences"]:
                    size = min(remaining // cost, 2000 - len(unit["text"]))
                    unit["text"] += padding * size
                    remaining -= size * cost
        if remaining:
            unit = body[BODY[-1]][-1]["sentences"][-1]
            unit["text"] += "x" * remaining
            remaining = 0
        assert remaining == 0
        check_writer_envelope(body, scope, stage="body")
        validate_body(
            body, case.aliases, writing_plan=RANGE_WRITING_PLAN, envelope=scope
        )
        assert nested_json_size(body) == 32000
        _, _, measured = actual_request(case, body_units(body), scope)
        assert measured <= proof["reservation"]


def test_summary_capacity_conflict_is_rejected_before_a_model_call(case, monkeypatch):
    scope = current_scope()
    # A typed host configuration conflict cannot be silently accepted.
    scope["request_caps"]["summary"] = 100
    from litagent.quality import digest

    scope["identity"] = digest({k: v for k, v in scope.items() if k != "identity"})
    with pytest.raises((ValueError, WriterError)):
        preflight_summary_contract(case.packet, case.aliases, RANGE_WRITING_PLAN, scope)
