"""Locate capacity candidate ownership errors without accepting or rewriting them."""

import copy
import json
from pathlib import Path

import pytest
from test_source_map_wire import source_case  # noqa: F401
from test_source_owner_transport import owner_case, parse_owner  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent import inline_citations, writing_flat
from litagent.capacity_repair import original_unit_directory, source_edit_schema
from litagent.config import load_config
from litagent.run_limits import (
    ARCHIVED_STAGE_REPAIR_VIT12,
    configured_run_limits,
    validate_run_limits,
)
from litagent.stage_repair import SCOPE_CONTRACT
from litagent.writing_v2 import WriterError, schema_for


def new_schema(schema):
    """Select only the new inline range policy on the existing repair schema."""
    schema = copy.deepcopy(schema)
    schema["x-citation-contract"] = inline_citations.CONTRACT
    schema["x-citation-scope-contract"] = SCOPE_CONTRACT
    return schema


@pytest.mark.parametrize("fault", ["mixed", "unknown", "duplicate"])
def test_candidate_source_conflicts_are_located_without_mutation(owner_case, fault):
    targets, _, schema, canonical = owner_case
    schema = new_schema(schema)
    wire = writing_flat.encode(canonical, schema)
    ids = original_unit_directory(targets)
    if fault == "mixed":
        wire["patches"][0]["source_unit_ids"].append(ids["r2"][0])
    elif fault == "unknown":
        wire["patches"][0]["source_unit_ids"] = ["writer:unknown:0:0"]
    else:
        wire["patches"][0]["source_unit_ids"] *= 2
    before = copy.deepcopy(wire)
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid") as exc:
        parse_owner(wire, schema)
    finding = exc.value.diagnostics[0]
    assert finding["wire_path"] == ["patches", 0, "source_unit_ids"]
    assert (
        finding["reason"]
        == {
            "mixed": "multiple_authorized_targets",
            "unknown": "unknown_sources",
            "duplicate": "duplicate_sources",
        }[fault]
    )
    assert finding["source_unit_ids"] == wire["patches"][0]["source_unit_ids"]
    assert wire == before


def test_all_source_conflicts_preserve_wire_order(owner_case):
    targets, _, schema, canonical = owner_case
    schema = new_schema(schema)
    wire = writing_flat.encode(canonical, schema)
    ids = original_unit_directory(targets)
    wire["patches"][0]["source_unit_ids"] = [ids["r1"][0], ids["r2"][0]]
    wire["patches"][1]["source_unit_ids"] = ["writer:unknown:0:0"]
    with pytest.raises(WriterError) as exc:
        parse_owner(wire, schema)
    assert [d["wire_path"][1] for d in exc.value.diagnostics] == [0, 1]


def test_legal_single_owner_mapping_still_uses_real_decoder(owner_case):
    _, _, schema, canonical = owner_case
    schema = new_schema(schema)
    wire = writing_flat.encode(canonical, schema)
    _, parsed = parse_owner(wire, schema)
    assert parsed["canonical"]["source_relations"] == canonical["source_relations"]


def test_old_protocol_retains_empty_conflict_diagnostic(owner_case):
    targets, _, schema, canonical = owner_case
    wire = writing_flat.encode(canonical, schema)
    wire["patches"][0]["source_unit_ids"].append(
        original_unit_directory(targets)["r2"][0]
    )
    with pytest.raises(WriterError) as exc:
        parse_owner(wire, schema)
    assert exc.value.diagnostics == []


def test_bad_host_directory_is_not_a_candidate_diagnostic(owner_case):
    _, _, schema, canonical = owner_case
    schema = new_schema(schema)
    wire = writing_flat.encode(canonical, schema)
    schema["x-source-owner-directory"]["r2"] = schema["x-source-owner-directory"]["r1"]
    with pytest.raises(WriterError) as exc:
        parse_owner(wire, schema)
    assert exc.value.diagnostics == []


def test_vit12_identity_is_archived_only():
    config = load_config("config/llm-stage-repair.yaml")
    limits = configured_run_limits(config)
    keys = ("candidate_recovery_identity", "citation_identity", "workflow_identity")
    limits.update(zip(keys, ARCHIVED_STAGE_REPAIR_VIT12))
    assert validate_run_limits(limits, archived=True) == limits
    with pytest.raises(ValueError, match="stage_repair_identity_invalid"):
        validate_run_limits(limits)
    for key in keys:
        with pytest.raises(ValueError):
            validate_run_limits({**limits, key: "0" * 64}, archived=True)


def test_vit12_settled_ledger_remains_read_only():
    from litagent.safety.planned_accounting import validate_accounting_snapshot
    from litagent.safety.planned_budget import PlannedBudget

    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-12/budget-ledger.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT 12")
    original = path.read_bytes()
    snapshot = json.loads(original)
    validate_accounting_snapshot(snapshot, expected_plan=snapshot["plan"])
    assert snapshot["used_tokens"] == 223366
    assert snapshot["requests_sent"] == 14
    with pytest.raises(ValueError, match="stage_repair_identity_invalid"):
        PlannedBudget(snapshot["plan"])
    assert path.read_bytes() == original


def test_capacity_rejection_exports_share_one_projection(tmp_path):
    import csv

    from test_benchmark_evaluation_execution import export_bundle, result_fixture

    from litagent import review_policy
    from litagent.eval.results import CONTRACT
    from litagent.stage_repair import public_rejection

    rejection = public_rejection(
        {
            "contract": "llm-stage-repair-v1",
            "category": "capacity",
            "stage": "repair",
            "action": "repair_response",
            "reason_code": "writer_capacity_source_map_invalid",
            "call": "repair_0",
            "remaining": 2,
            "actual_tokens": 120,
            "retained_report_sha256": "a" * 64,
            "body_sha256": None,
            "raw": "PRIVATE CANDIDATE",
        }
    )
    assert rejection is not None
    value = result_fixture()
    value["observations"][0]["evaluation_summary"] = {
        "contract": CONTRACT,
        "report_sha256": "a" * 64,
        "metrics": {"faithfulness": {"status": "unverified", "score": None}},
        "review_workflow": {
            "contract": review_policy.CONTRACT,
            "report_sha256": "a" * 64,
            "reviews": [],
            "allowances": {
                key: {
                    "limit": 3,
                    "used": int(key == "capacity"),
                    "remaining": 2 if key == "capacity" else 3,
                }
                for key in review_policy.ALLOWANCES
            },
            "publishable": False,
            "evaluations": [
                {
                    "round": 1,
                    "kind": "repair_incomplete",
                    "status": "unverified",
                    "reason": "writer_capacity_source_map_invalid",
                    "patch_rejection": rejection,
                }
            ],
        },
    }
    original = copy.deepcopy(value)
    export_bundle(tmp_path, value)
    data = json.loads((tmp_path / "benchmark.json").read_text())
    projected = data["observations"][0]["evaluation_summary"]
    with (tmp_path / "benchmark.csv").open() as stream:
        assert (
            json.loads(next(csv.DictReader(stream))["evaluation_summary"]) == projected
        )
    for suffix in ("md", "html"):
        text = (tmp_path / f"benchmark.{suffix}").read_text()
        assert "重做授权响应" in text and "writer_capacity_source_map_invalid" in text
        assert "2 remaining" in text and "120 settled tokens" in text
        assert "PRIVATE CANDIDATE" not in text
    assert value == original


def test_vit12_original_candidate_still_rejects_exact_cross_target_row():
    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-12/repairs.json.writer-v2.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT 12")
    original = path.read_bytes()
    journal = json.loads(original)
    call = journal["calls"]["repair_3"]
    targets = call["guidance_wire_record"]["authorized_targets"]
    schema = source_edit_schema(
        schema_for("repair", {f"c{i}": {} for i in range(1, 30)}, targets),
        targets,
        contract="capacity-source-map-v4",
    )
    with pytest.raises(WriterError) as old:
        writing_flat.parse(
            call["raw"], new_schema(schema) | {"x-citation-scope-contract": None}
        )
    assert old.value.diagnostics == []
    with pytest.raises(WriterError) as new:
        writing_flat.parse(call["raw"], new_schema(schema))
    finding = new.value.diagnostics[0]
    assert finding["wire_path"] == ["patches", 18, "source_unit_ids"]
    assert finding["source_unit_ids"] == ["writer:methods:5:1", "writer:methods:6:0"]
    assert finding["authorized_target_ids"] == ["r15", "r16"]
    assert path.read_bytes() == original
