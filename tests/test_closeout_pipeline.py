"""Exercise one inventory recheck followed by transactional scientific repair."""

import json

import pytest
import test_structured_repair_pipeline as structured
import test_writer_source_view_pipeline as writer_view
from test_evaluation_dimensions_pipeline import separated_peer
from test_evaluation_stability_pipeline import read_inventory_catalog_at_provider
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import PROFILE, profile_flags


class RecheckGateway(structured.LongScientificGateway):
    """Change only external atomicity decisions, preserving the actual report."""

    def __init__(self, case, *, persistent=False):
        super().__init__(case, 0)
        self.inventories = 0
        self.persistent = persistent
        original = self.evaluator.chat

        async def chat(messages, **options):
            reply = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" in payload and (self.inventories == 1 or persistent):
                value = json.loads(reply.content)
                value["decisions"][0]["atomic"] = False
                reply.content = json.dumps(value)
            return reply

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)


def select(monkeypatch):
    """Select only host configuration while retaining all real consumers."""
    monkeypatch.setattr(writer_view, "source_flags", lambda *_: profile_flags(PROFILE))
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_same_inventory_recheck_preserves_a_slot_for_scientific_repair(
    source_case, tmp_path, monkeypatch, persistent
):
    select(monkeypatch)
    gateway = RecheckGateway(source_case, persistent=persistent)
    monkeypatch.setattr(
        structured, "LongScientificGateway", lambda *_args, **_kw: gateway
    )
    result = await writer_view.run_scientific_case(source_case, tmp_path, monkeypatch)
    writer_view.assert_selected_requests(result, source_case.packet)
    requests = [row for row in gateway.sent if "inventory_recovery" in row["payload"]]
    assert len(requests) == 1
    request = requests[0]["request"]
    messages = request.get("messages", request.get("input"))
    assert "single inventory-recovery slot" in messages[0]["content"]
    assert "cycling inventories is no progress" not in messages[0]["content"]
    attempts = result.report["metadata"]["targeted_repair_state"]["attempts"]
    recovery = [
        a for a in attempts if a.get("purpose") == "bounded-inventory-recovery-v2"
    ]
    assert len(recovery) == 1
    review = recovery[0]["inventory_review"]
    assert review["operation"] == "independent_recheck"
    assert review["content_changed"] is False
    assert review["status"] == ("unresolved" if persistent else "resolved")
    assert review["fragments"]
    assert len(attempts) == (1 if persistent else 2)
    assert result.report["delivery"]["publishable"] is not persistent
    if persistent:
        assert not result.journal["candidates"]
        assert "FLOPs by 44%" in result.report["survey"]
    else:
        final = result.report["evaluation"]["faithfulness"]["details"]
        assert final["inventory_check"]["status"] == "passed"
        assert final["coverage"]["complete"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert result.journal["candidates"][-1]["committed"]


@pytest.mark.asyncio
async def test_valid_inventory_can_use_both_slots_for_prose_without_publishing_failure(
    source_case, tmp_path, monkeypatch
):
    select(monkeypatch)
    result = await writer_view.run_scientific_case(
        source_case, tmp_path, monkeypatch, failures=2
    )
    attempts = result.report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    assert not any(
        a.get("purpose") == "bounded-inventory-recovery-v2" for a in attempts
    )
    assert not result.report["delivery"]["publishable"]
    assert not any(row["committed"] for row in result.journal["candidates"])
