"""Bind the complete server certificate, including types and prepared inputs."""

import copy

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.frozen_writing import plan_frozen_budget, validate_frozen_writer_plan
from litagent.quality import digest
from litagent.writing_v2 import WriterError


@pytest.fixture
def frozen_bundle(case):
    """Keep all records needed by the real frozen preflight producer."""
    return {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "methodology": {},
        "extractions": [],
    }


@pytest.mark.parametrize("field", ["input", "allowance", "stage", "scope", "extra"])
def test_plan_comparison_rejects_every_drift_without_mutation(frozen_bundle, field):
    plan = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v4")
    original = copy.deepcopy(plan)
    changed = copy.deepcopy(plan)
    if field == "input":
        changed["envelope"]["source_identity"] = "f" * 64
    elif field == "allowance":
        changed["planned_tokens"] += 10000
    elif field == "stage":
        changed["stages"]["global"]["request_cap"] -= 1
    elif field == "scope":
        changed["envelope"]["body_max_units"] -= 1
    else:
        changed["ignored_by_python_consumer"] = True
    changed_before = copy.deepcopy(changed)
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(plan, changed)
    assert plan == original and changed == changed_before
    validate_frozen_writer_plan(plan, copy.deepcopy(plan))


@pytest.mark.parametrize("wrong", [True, 1.0, None, float("nan")])
def test_complete_plan_equality_preserves_json_types(wrong):
    """Python considers True == 1; a signed JSON input certificate must not."""
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan({"count": 1}, {"count": wrong})


def test_preflight_prepares_the_same_records_without_rewriting_bundle(frozen_bundle):
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.quality import build_fact_cards
    from litagent.review_structure import prepare_writer_methodology
    from litagent.writing_v2 import source_packet, writer_budget_plan

    bundle = frozen_bundle
    bundle["requested_methods"] = ["EViT"]
    bundle["extractions"] = [
        {"paper_id": "evit", "fulltext_acquisition": {"status": "failed"}}
    ]
    before = copy.deepcopy(bundle)
    actual = plan_frozen_budget(bundle, budget_policy="planned-budget-v4")
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    prepared = prepare_writer_methodology(
        bundle["methodology"],
        selection=selection,
        cards=cards,
        extractions=bundle["extractions"],
        requested_methods=bundle["requested_methods"],
        upstream_results={"extract": bundle["extractions"]},
    )
    packet, aliases = source_packet(
        cards, selection, bundle["query"], bundle["requirements"], "zh-CN"
    )
    expected = writer_budget_plan(
        packet, aliases, methodology=prepared, budget_policy="planned-budget-v4"
    )
    assert actual == expected and bundle == before
    assert prepared["evidence_counts"]["selected"] == len(cards)
    assert prepared["fulltext_acquisition"]["evit"]["status"] == "failed"


@pytest.mark.parametrize("policy", ["planned-budget-v2", "planned-budget-v3"])
def test_historical_planning_ignores_optional_methodology(frozen_bundle, policy):
    before = plan_frozen_budget(frozen_bundle, budget_policy=policy)
    frozen_bundle.pop("methodology")
    assert plan_frozen_budget(frozen_bundle, budget_policy=policy) == before


@pytest.mark.parametrize("change", ["source", "copy", "manifest_type", "config"])
def test_server_seal_detects_changes_at_request_boundaries(
    tmp_path, monkeypatch, change
):
    import hashlib
    import json

    from litagent.config import load_config
    from litagent.contracts import build_config_summary
    from scripts.run_frozen_writing import frozen_runtime_guard

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("# fixed source\n")
    root = tmp_path / "sealed"
    (root / "source").mkdir(parents=True)
    frozen_copy = root / "source/source.py"
    frozen_copy.write_bytes(source.read_bytes())
    config = load_config()
    value = {
        "source_hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "config": build_config_summary(config, "strict_auto_v2"),
        "count": 1,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(value))
    guard = frozen_runtime_guard(root, value, config)
    guard()
    if change == "source":
        source.write_text("# changed\n")
    elif change == "copy":
        frozen_copy.write_text("# changed copy\n")
    elif change == "config":
        config.llm.max_tokens += 1
    else:
        value["count"] = True
        manifest_path.write_text(json.dumps(value))
    with pytest.raises(WriterError, match="frozen_campaign_source_changed"):
        guard()
