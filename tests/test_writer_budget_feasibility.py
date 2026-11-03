"""Keep necessary reservation diagnostics distinct from complete admission."""

import csv
import json

import pytest
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import PlannedBudget
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterSession,
)
from scripts.review_writer_budget import export_review, reservation_floor


def test_maximum_legal_path_counts_fixed_outputs_and_real_counter(case):
    result = reservation_floor(case.packet)
    assert result["path_requests"] == 5 + 3 + 1 + 2 + 3 * (1 + 8 + 1) == 41
    assert result["output_reservations"] == 10 * 20000 + 4096 + 30 * 16384
    assert result["padding_per_request"] == strict_request_reservation([], 0) - 2
    assert result["packet_copies"] == 7
    assert result["minimum_computed_allowance"] >= (
        result["necessary_reservations"] * 1.1
    )


@pytest.mark.parametrize("text", ["普通原文", '\\"\n\t', "😀é漢字", "x" * 2000])
def test_packet_members_are_only_a_lower_bound_on_nested_wire(case, text):
    packet = {**case.packet, "untrusted_example": text}
    result = reservation_floor(packet)
    user = json.dumps({**packet, "current_body": {"text": text}}, ensure_ascii=False)
    actual = strict_request_reservation([{"role": "user", "content": user}], 20000)
    assert actual >= result["packet_member_bytes"] + 1024 + 20000
    assert not result["can_admit"]


@pytest.mark.parametrize("large", [False, True])
def test_floor_neither_admits_small_packets_nor_hides_ceiling_conflicts(case, large):
    packet = {**case.packet, "extra": "x" * (60000 if large else 0)}
    result = reservation_floor(packet)
    assert result["ceiling_conflict_proven"] is large
    assert result["can_admit"] is False
    with pytest.raises((KeyError, ValueError)):
        PlannedBudget(result)


@pytest.mark.asyncio
async def test_actual_chapters_and_two_patches_retain_seven_complete_packets(
    case, tmp_path
):
    """Exercise actual writer consumers; replace only the external transport."""
    owner = Provider(case)
    session = WriterSession(
        owner,
        tmp_path / "repair.json",
        "offline-feasibility",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    body = await session.initial_body()
    await session.summaries(body, "summary_initial")
    for section in ("methods", "open_problems"):
        targets = {
            section: {
                "path": [section, 0, 0],
                "original": body[section][0]["sentences"][0],
                "explanation": "A controlled source-bound repair target.",
            }
        }
        body = await session.patch(body, targets, cause="evaluation")
    assert len(session.state["attempts"]) == 2
    complete = [p for p, _ in owner.calls if "sources" in p]
    assert len(complete) == reservation_floor(case.packet)["packet_copies"] == 7
    for payload in complete:
        assert {k: payload[k] for k in case.packet} == case.packet
    summaries = [p for p, _ in owner.calls if "body_units" in p]
    assert len(summaries) == 1 and "sources" not in summaries[0]


def test_four_formats_share_scalars_without_exposing_frozen_packet(case, tmp_path):
    row = {"case_id": "controlled", **reservation_floor(case.packet)}
    review = {"cases": [row], "new_paid_requests": 0}
    export_review(review, tmp_path)
    assert json.loads((tmp_path / "review.json").read_text()) == review
    page = (tmp_path / "review.html").read_text()
    embedded = page.split('<script id="data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == review
    rows = list(csv.DictReader((tmp_path / "review.csv").open()))
    assert int(rows[0]["necessary_reservations"]) == row["necessary_reservations"]
    assert str(row["necessary_reservations"]) in (tmp_path / "review.md").read_text()
    for path in tmp_path.iterdir():
        value = path.read_text()
        assert "Relation networks learn a comparison function." not in value
        assert "完整" in value or path.suffix in {".json", ".csv"}


@pytest.mark.parametrize("value", [None, [], {}, {"sources": {}, "cards": {}}])
def test_invalid_packet_is_not_used_for_feasibility(value):
    with pytest.raises(ValueError, match="packet_invalid"):
        reservation_floor(value)


def test_public_provenance_uses_relative_paths_and_validated_sources(case, monkeypatch):
    """Bind the real source constructor without exposing local machine paths."""
    from litagent.quality import digest
    from scripts import review_writer_budget as review

    bundle = {
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "archive_sha256": "controlled-archive-identity",
    }
    monkeypatch.setattr(review, "read_bundle", lambda _: bundle)
    result = review.build_review()
    assert len(result["cases"]) == 3
    assert all(not p.startswith("/") for p in result["source_hashes"])
    bundle["fact_cards_sha256"] = "corrupted"
    with pytest.raises(ValueError, match="identity_conflict"):
        review.build_review()
