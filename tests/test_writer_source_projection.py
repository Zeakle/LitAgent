"""Keep frozen source snapshots readable without inventing extraction or metadata."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.demo import extract_report_evidence, project_flow_demo_view
from litagent.frozen_writing import FrozenSelector
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.quality import digest
from litagent.writing_v2 import compile_document


@pytest.fixture
def frozen_archive(case):
    """Store only the real selected pool inside a report worker, as run 003 did."""
    selection = copy.deepcopy(case.selection.to_dict())
    for eid, item in selection["selected_items"].items():
        item["paper_title"] = "Source " + eid
    survey, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    metadata = {
        "generation_strategy": "body_summary_v2",
        "report_language": "en",
        "writing_bindings": bindings,
        "fact_cards": copy.deepcopy(case.cards),
        "evidence_snapshot_sha256": digest(case.cards),
    }
    return {
        "run_id": "bound",
        "status": "completed",
        "report": {
            "survey": survey,
            "metadata": metadata,
            "quality": {"status": "unverified"},
            "delivery": {"publishable": False},
        },
        "nodes": {
            "worker:adversarial_review": {
                "kind": "worker",
                "output": {
                    "generation_strategy": "body_summary_v2",
                    "evidence_selection": selection,
                    "fact_cards": copy.deepcopy(case.cards),
                    "evidence_snapshot_sha256": digest(case.cards),
                },
            }
        },
    }


def test_reader_restores_only_bound_selected_sources_and_retains_originals(
    frozen_archive,
):
    before = json.dumps(frozen_archive, sort_keys=True)
    assert extract_report_evidence(frozen_archive["nodes"]) == []
    evidence = extract_report_evidence(
        frozen_archive["nodes"], report=frozen_archive["report"]
    )
    assert {e["evidence_id"] for e in evidence} == {"a", "b"}
    assert (
        next(e for e in evidence if e["evidence_id"] == "b")["supporting_text"]
        == "Relation networks learn a comparison function."
    )
    view = project_flow_demo_view(frozen_archive, source="archive")
    assert len(view["evidence"]) == 2
    assert "Source b" in view["rendered_report"]["html"]
    assert not any(
        str(w).startswith("unresolved_evidence:")
        for w in view["rendered_report"]["warnings"]
    )
    assert json.dumps(frozen_archive, sort_keys=True) == before
    assert not view["delivery"]["publishable"]


@pytest.mark.parametrize(
    "fault",
    [
        "report_sha",
        "source",
        "paper",
        "snapshot",
        "outside",
        "ambiguous",
        "invalid_item",
        "invalid_selection",
    ],
)
def test_reader_does_not_borrow_unbound_or_corrupt_sources(frozen_archive, fault):
    metadata = frozen_archive["report"]["metadata"]
    out = frozen_archive["nodes"]["worker:adversarial_review"]["output"]
    pool = out["evidence_selection"]["selected_items"]
    if fault == "report_sha":
        metadata["writing_bindings"]["report_sha256"] = "wrong"
    elif fault == "source":
        pool["b"]["supporting_text"] = "Different original."
    elif fault == "paper":
        pool["b"]["paper_id"] = "another-paper"
    elif fault == "snapshot":
        out["evidence_snapshot_sha256"] = "another-snapshot"
    elif fault == "outside":
        pool["outside"] = {**copy.deepcopy(pool["b"]), "evidence_id": "outside"}
    elif fault == "invalid_item":
        pool["b"] = "not an evidence record"
    elif fault == "invalid_selection":
        out["evidence_selection"]["valid"] = False
    else:
        other = copy.deepcopy(out)
        other["evidence_selection"]["selected_items"]["b"][
            "text"
        ] = "Different unverified candidate."
        frozen_archive["nodes"]["worker:other"] = {"kind": "worker", "output": other}
    assert (
        extract_report_evidence(
            frozen_archive["nodes"], report=frozen_archive["report"]
        )
        == []
    )


def test_same_native_and_worker_snapshot_is_not_ambiguous(frozen_archive):
    pool = frozen_archive["nodes"]["worker:adversarial_review"]["output"][
        "evidence_selection"
    ]["selected_items"]
    frozen_archive["nodes"]["evidence.retrieve:reused"] = {
        "kind": "evidence.retrieve",
        "output": {"selected_items": copy.deepcopy(pool)},
    }
    assert (
        len(
            extract_report_evidence(
                frozen_archive["nodes"], report=frozen_archive["report"]
            )
        )
        == 2
    )
    frozen_archive["report"]["metadata"]["writing_bindings"][
        "report_sha256"
    ] = "old-body"
    assert (
        extract_report_evidence(
            frozen_archive["nodes"], report=frozen_archive["report"]
        )
        == []
    )


@pytest.mark.asyncio
async def test_frozen_selector_records_reuse_without_model_or_retrieval_start(
    case, tmp_path
):
    recorder = RunRecorder("replay", case.packet["query"], ArchiveRepository(tmp_path))
    bundle = {
        "query": case.packet["query"],
        "source_run_id": "prior",
        "archive_sha256": "source-sha",
        "selection": case.selection.to_dict(),
    }
    selector = FrozenSelector(bundle, trace_hook=recorder)
    await selector.select(bundle["query"], case.ledger)
    artifact = recorder.snapshot()
    assert artifact["usage"]["total_tokens"] == 0
    assert [e["event"] for e in artifact["events"]] == ["evidence.retrieve.complete"]
    node = next(iter(artifact["nodes"].values()))
    assert node["metadata"]["execution"] == "reused_frozen_input"
    assert (
        node["metadata"]["source_run_id"] == "prior"
        and node["metadata"]["source_artifact_sha256"] == "source-sha"
    )
    assert node["started_at"] is None and node["elapsed_ms"] == 0
    assert {x["evidence_id"] for x in extract_report_evidence(artifact["nodes"])} == {
        "a",
        "b",
    }


@pytest.mark.asyncio
async def test_formatting_and_audit_consumers_receive_same_complete_originals(
    frozen_archive, monkeypatch
):
    from litagent import api

    service = SimpleNamespace(get_version=AsyncMock(side_effect=KeyError("bound")))
    monkeypatch.setattr(api, "_interactive_service", AsyncMock(return_value=service))
    monkeypatch.setattr(
        api, "_flow_artifact", lambda run: frozen_archive if run == "bound" else None
    )
    version, report, ledger, catalog = await api._resolve_formatting_source("bound", 1)
    assert (
        version.content_sha256
        == frozen_archive["report"]["metadata"]["writing_bindings"]["report_sha256"]
    )
    assert (
        ledger["b"]["supporting_text"]
        == "Relation networks learn a comparison function."
    )
    snapshot = api._audit_snapshot_from_source(version, report, ledger, frozen_archive)
    assert snapshot["ledger"]["b"]["supporting_text"] == ledger["b"]["supporting_text"]
    assert catalog == {}
