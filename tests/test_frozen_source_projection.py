"""Retain reused originals and source metadata without inventing new execution."""

import copy
from unittest.mock import AsyncMock

import pytest
from test_frozen_generation_jev import trial

from litagent.demo import project_flow_demo_view
from litagent.runner import LitAgent


@pytest.mark.asyncio
async def test_actual_frozen_executor_records_locatable_reused_originals(
    tmp_path, monkeypatch
):
    prepare = AsyncMock(side_effect=ValueError("offline_after_freeze"))
    monkeypatch.setattr(LitAgent, "_prepare_report", prepare)
    _, _, _, _, state, artifact, _, service, _ = await trial(
        tmp_path,
        monkeypatch,
        valid_writer=True,
        source_metadata={
            "title": "Verified source title",
            "authors": ["Alice Example", "张三"],
            "year": 2024,
        },
    )
    prepare.assert_awaited_once()
    imported = artifact["nodes"]["worker:extract"]
    assert imported["metadata"]["execution"] == "reused_frozen_input"
    assert imported["metadata"]["source_run_id"] == "previous"
    assert imported["started_at"] is None and imported["elapsed_ms"] == 0
    assert (
        imported["output"][0]["evidence_items"][0]["supporting_text"]
        == "EViT improves speed by 50% on DeiT-S."
    )
    catalog = artifact["nodes"]["worker:dedup"]["output"][0]
    assert catalog["authors"] == ["Alice Example", "张三"] and catalog["year"] == 2024
    assert not any(
        e["event"] == "worker.start"
        and e["data"].get("task_id") in {"extract", "dedup"}
        for e in artifact["events"]
    )
    assert state["costs"]["survey"]["requests"] == len(
        [e for e in artifact["events"] if e["event"] == "llm.start"]
    )
    view_fixture = copy.deepcopy(artifact)
    view_fixture["report"] = {
        "survey": "EViT improves speed. [E:a]",
        "metadata": {"report_language": "en"},
        "quality": {"status": "unverified"},
    }
    view = project_flow_demo_view(view_fixture, source="archive")
    assert view["evidence"][0]["locator"] == "paper=evit;chunk=abstract"
    rendered = str(view["rendered_report"])
    assert "Alice Example" in rendered and "张三" in rendered and "2024" in rendered
    assert "Authors unavailable" not in rendered
    assert not state.get("report") and not service.submit.called
