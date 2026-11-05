"""Keep real global-review occurrences, kinds and ownership together losslessly."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_evaluation_definition_contract import prompt_for
from test_evaluation_dimensions_identity import dimension_envelope
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_global_permissions import global_payload  # noqa: F401
from test_global_permissions import (
    actual_schema,
    historical_parent,
)

from litagent.eval import batched, decision_output, global_input, scoped
from litagent.eval.responsibilities import parse_global_response
from litagent.eval.results import CONTRACT
from litagent.safety.budget import strict_request_reservation


def flatten(wire):
    """Read full explicit kinds while preserving every distinct occurrence."""
    records = []
    for part in (p for group in wire["report_sections"] for p in group["layout"]):
        if "fragment" not in part:
            continue
        for kind, rows in (
            (kind, part[kind]) for kind in global_input.KINDS if kind in part
        ):
            for row in rows:
                records.append(
                    {
                        **dict(zip(wire["assertion_columns"], row, strict=True)),
                        "sequence": int(row[0][1:]),
                        "kind": kind,
                        "source_fragment_id": part["fragment"],
                        "source_unit_id": part["unit"],
                    }
                )
    return sorted(records, key=lambda row: row["sequence"])


def old_wire(payload):
    """Run the real pre-join projections, without weakening the current owner."""
    from litagent.writing_flat import evaluation_projection, global_identity_projection

    wire = evaluation_projection(payload)
    wire = batched.global_input_payload(wire, explicit_kinds=True)
    wire, _ = global_identity_projection(wire, None, payload, explicit_kinds=True)
    wire.pop("expected_source_unit_ids")
    wire.pop("expected_requirement_ids")
    return wire


def current_wire(payload):
    messages, native, _ = actual_schema(payload)
    return json.loads(messages[1]["content"]), messages, native


def test_actual_builder_keeps_all_kinds_and_identical_text_at_distinct_locations(
    global_payload,
):
    before = copy.deepcopy(global_payload)
    previous = old_wire(global_payload)
    wire, messages, native = current_wire(global_payload)
    assert wire["global_input_contract"] == global_input.CONTRACT
    assert "source_units" not in wire and "source_units_by_section" not in wire
    assert "assertions_by_kind" not in wire
    rows = flatten(wire)
    assert len(rows) == len(global_payload["assertions"]) == 6
    assert [r["claim_id"] for r in rows] == [f"a{i}" for i in range(1, 7)]
    assert len({r["claim_text"] for r in rows}) == 1
    assert {r["source_unit_id"] for r in rows} == {"v1", "v2"}
    assert {r["kind"] for r in rows} == set(global_input.KINDS)
    for row, original in zip(rows, global_payload["assertions"], strict=True):
        assert row["claim_text"] == original["claim_text"]
        assert row["kind"] == original["kind"]
        assert row["evidence_aliases"] == (None if original["evidence"] else [])
    for part in (p for group in wire["report_sections"] for p in group["layout"]):
        old_fragment = previous["source_units"][part["unit"]][0]
        assert part["text"] == old_fragment["text"]
        assert [
            wire["citation_aliases"][ref - 1] for ref in part["citations"]
        ] == old_fragment["citation_aliases"]
    assert global_payload == before
    assert (
        "report_sections preserves the complete report order" in messages[0]["content"]
    )
    assert "source_units maps unit ID" not in messages[0]["content"]
    assert "Do not reassess atomicity" in messages[0]["content"]
    assert strict_request_reservation(messages, 32768, native) > 32768


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_unit",
        "wrong_section",
        "unknown_fragment",
        "swapped_fragment",
        "changed_prose",
        "missing_claim",
        "changed_claim",
        "changed_kind",
        "changed_refs",
        "bool_sequence",
        "duplicate_claim",
        "unknown_exclusion",
        "duplicate_fragment",
        "unknown_fragment_field",
        "partial_kind_group",
    ],
)
def test_corrupt_join_fails_before_sending_instead_of_discarding_content(
    global_payload, fault
):
    wire = old_wire(global_payload)
    if fault == "unknown_unit":
        wire["source_units"]["v999"] = wire["source_units"].pop("v2")
    elif fault == "wrong_section":
        wire["source_units_by_section"]["methods"].pop()
    elif fault == "unknown_fragment":
        wire["source_units"]["v1"][0]["source_fragment_id"] = "f999"
    elif fault == "swapped_fragment":
        wire["source_units"]["v1"], wire["source_units"]["v2"] = (
            wire["source_units"]["v2"],
            wire["source_units"]["v1"],
        )
    elif fault == "changed_prose":
        wire["source_units"]["v1"][0]["text"] = "Altered scientific qualifier."
    elif fault == "missing_claim":
        wire["assertions_by_kind"]["scientific"].pop()
    elif fault == "changed_claim":
        wire["assertions_by_kind"]["scientific"][0][3] = "Different result."
    elif fault == "changed_kind":
        wire["assertions_by_kind"]["scientific"].append(
            wire["assertions_by_kind"]["report_structure"].pop()
        )
    elif fault == "changed_refs":
        wire["assertions_by_kind"]["scientific"][0][4] = []
    elif fault == "bool_sequence":
        wire["assertions_by_kind"]["scientific"][0][0] = True
    elif fault == "duplicate_claim":
        wire["assertions_by_kind"]["scientific"].append(
            copy.deepcopy(wire["assertions_by_kind"]["scientific"][0])
        )
    elif fault == "unknown_exclusion":
        wire["fragment_inventory"].append("f999")
    elif fault == "duplicate_fragment":
        wire["source_units"]["v1"].append(copy.deepcopy(wire["source_units"]["v1"][0]))
    elif fault == "partial_kind_group":
        wire["assertions_by_kind"].pop("runtime_method")
    else:
        wire["source_units"]["v1"][0]["unexpected"] = "Do not silently drop."
    frozen = copy.deepcopy(wire)
    with pytest.raises(ValueError, match="global_local_input_invalid"):
        global_input.project_input(wire, global_payload)
    assert wire == frozen


def test_selected_view_does_not_rewrite_old_requests_or_accept_double_projection(
    global_payload,
):
    marker = {"historical_wire": True}
    assert (
        global_input.project_input(marker, historical_parent(global_payload)) is marker
    )
    system = "Unchanged historical instructions."
    assert (
        global_input.project_prompt(system, historical_parent(global_payload)) == system
    )
    wire, _, _ = current_wire(global_payload)
    with pytest.raises(ValueError, match="global_local_input_invalid"):
        global_input.project_input(wire, global_payload)


def real_061_payload():
    """Reconstruct exact public report-side inputs, without recovering a live task."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/run061-global-completeness.json").read_text()
    )
    assert fixture["scientific_gold_standard"] is False
    wire = fixture["wire"]
    unit_ids = {uid: "host:" + uid for uid in wire["source_units"]}
    sections = {
        uid: section
        for section, ids in wire["source_units_by_section"].items()
        for uid in ids
    }
    prose = {
        part["source_fragment_id"]: part["prose"]
        for part in wire["report_layout"]
        if "source_fragment_id" in part
    }
    fragments = {}
    units = []
    for uid, entries in wire["source_units"].items():
        values = []
        for fragment in entries:
            fid = fragment["source_fragment_id"]
            refs = fragment["citation_aliases"]
            value = {
                "source_fragment_id": "host:" + fid,
                "text": prose[fid]
                + (" " + " ".join(f"[E:{ref}]" for ref in refs) if refs else ""),
                "citation_aliases": refs,
            }
            fragments[fid] = (unit_ids[uid], value)
            values.append(value)
        units.append({"source_unit_id": unit_ids[uid], "source_fragments": values})
    claims = []
    for kind, rows in wire["assertions_by_kind"].items():
        for row in rows:
            record = dict(zip(wire["assertion_columns"], row, strict=True))
            uid, fragment = fragments[record["source_fragment_id"]]
            refs = record["evidence_aliases"]
            claims.append(
                {
                    "sequence": record["sequence"],
                    "claim_id": "host:" + record["claim_id"],
                    "kind": kind,
                    "claim_text": record["claim_text"],
                    "source_unit_id": uid,
                    "source_fragment_id": fragment["source_fragment_id"],
                    "evidence": [
                        {"evidence_alias": ref, "material_ref": ref}
                        for ref in (
                            fragment["citation_aliases"] if refs is None else refs
                        )
                    ],
                }
            )
    claims.sort(key=lambda c: c.pop("sequence"))
    layout = [
        (
            {
                "source_fragment_id": fragments[p["source_fragment_id"]][1][
                    "source_fragment_id"
                ],
                "text": fragments[p["source_fragment_id"]][1]["text"],
            }
            if "source_fragment_id" in p
            else copy.deepcopy(p)
        )
        for p in wire["report_layout"]
    ]
    base = {
        **{
            key: wire[key]
            for key in (
                "global_findings_contract",
                "report_input_contract",
                "writer_content_contract",
                "summary_constraints_contract",
                "body_analysis_contract",
                "evaluation_input_contract",
                "coverage_requirements",
            )
        },
        "writer_envelope": dimension_envelope(evaluation_result_contract=CONTRACT),
        "source_units": units,
        "report_layout": layout,
        "report_structure": [
            {"source_unit_id": unit_ids[uid], "section": sections[uid]}
            for uid in unit_ids
        ],
        "assertions": claims,
        "unit_inventory": [],
    }
    inventory = [
        {
            "source_unit_id": uid,
            "source_fragment_id": f["source_fragment_id"],
            "assertion_count": sum(
                c["source_fragment_id"] == f["source_fragment_id"] for c in claims
            ),
            "exclusion_reason": "",
        }
        for uid, f in fragments.values()
    ]
    return (
        batched.build_global_payload(base, {"fragment_inventory": inventory}),
        fixture,
    )


def test_real_061_complete_request_has_all_152_assertions_beside_actual_originals():
    payload, fixture = real_061_payload()
    original_sha = hashlib.sha256(
        json.dumps(fixture, ensure_ascii=False).encode()
    ).hexdigest()
    wire, messages, native = current_wire(payload)
    records = flatten(wire)
    assert len(records) == 152
    assert len({r["source_unit_id"] for r in records}) == 38
    assert (
        len(
            [
                p
                for group in wire["report_sections"]
                for p in group["layout"]
                if "fragment" in p
            ]
        )
        == 41
    )
    previous = fixture["wire"]
    by_id = {r["claim_id"]: r for r in records}
    for kind, rows in previous["assertions_by_kind"].items():
        for row in rows:
            old = dict(zip(previous["assertion_columns"], row, strict=True))
            new = by_id[old["claim_id"]]
            assert {k: new[k] for k in old} == old
            assert new["kind"] == kind
    for number in range(139, 146):
        row = by_id[f"a{number}"]
        assert row["source_unit_id"] == ("v36" if number < 143 else "v37")
        assert row["source_fragment_id"] == ("f39" if number < 143 else "f40")
    assert strict_request_reservation(messages, 32768, native) < 120000
    # The original seven negative findings remain negative if returned again.
    progress = {}
    result = parse_global_response(fixture["raw_response"], payload, progress)
    assert len(result.issues) == 7
    assert all(issue.code == "missing_assertion" for issue in result.issues)
    assert all(issue.anchor_ids == ["unit:host:v38"] for issue in result.issues)
    assert (
        hashlib.sha256(json.dumps(fixture, ensure_ascii=False).encode()).hexdigest()
        == original_sha
    )


@pytest.mark.parametrize("fault", ["heading", "gap", "fragment", "order"])
def test_full_original_layout_cannot_be_rebound_or_silently_changed(fault):
    payload, _ = real_061_payload()
    wire = old_wire(payload)
    if fault == "heading":
        wire["report_layout"][0]["text"] = "Forged report heading."
    elif fault == "gap":
        wire["report_layout"].pop(0)
    elif fault == "fragment":
        next(part for part in wire["report_layout"] if "prose" in part)[
            "prose"
        ] += " Added premise."
    else:
        wire["report_layout"] = list(reversed(wire["report_layout"]))
    with pytest.raises(ValueError, match="global_local_input_invalid"):
        global_input.project_input(wire, payload)
