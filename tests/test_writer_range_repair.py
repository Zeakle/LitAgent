"""Check exact range authorization without treating rewriting as scientific proof."""

import asyncio
import copy
import json

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.quality import digest
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import (
    BODY,
    EOF_CONTAINER_POLICY,
    FOCUSED_WRITING_PLAN,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    capacity_repair_targets,
    check_writing_plan,
    repair_capacity_snapshot,
    validate_targets,
    writing_plan_spec,
)


def overlong(case):
    """Mirror the actual twelve-by-three shape with distinct controlled premises."""
    body = copy.deepcopy(case.body)
    template = body["methods"][0]["sentences"][0]
    body["methods"] = [
        {
            "sentences": [
                {**template, "text": f"Premise {p}-{i}: {template['text']}"}
                for i in range(3)
            ]
        }
        for p in range(12)
    ]
    return body


def session(case, tmp_path, provider=None, plan=RANGE_WRITING_PLAN):
    """Use the actual persistent controller with an explicit server-only plan."""
    return WriterSession(
        provider or Provider(case),
        tmp_path / "repair.json",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=plan,
    )


class RangeProvider(Provider):
    """Replace only transport with bounded original-range selections."""

    async def chat(self, messages, **kwargs):
        response = await super().chat(messages, **kwargs)
        payload = json.loads(messages[1]["content"])
        if "targets" in payload:
            result = json.loads(response.content)
            if result["outcome"] == "patched":
                for patch in result["patches"]:
                    target = payload["targets"][patch["target_id"]]
                    if "end" in target:
                        patch["sentences"] = copy.deepcopy(
                            target["original"][: target["maximum"]]
                        )
                response.content = json.dumps(result)
        return response


def test_range_policy_is_explicit_and_does_not_rewrite_the_old_spec(case, tmp_path):
    old = writing_plan_spec(FOCUSED_WRITING_PLAN)
    assert "repair_scope_contract" not in old
    new = writing_plan_spec(RANGE_WRITING_PLAN)
    assert new["body_max_units"] == old["body_max_units"] == 66
    assert new["atomic_claim_capacity"] == 128
    assert (
        session(case, tmp_path / "a").identity
        != session(case, tmp_path / "b", plan=FOCUSED_WRITING_PLAN).identity
    )
    with pytest.raises(WriterError) as caught:
        check_writing_plan(overlong(case), RANGE_WRITING_PLAN)
    assert caught.value.family == "quality" and caught.value.candidate == overlong(case)
    with pytest.raises(WriterError) as caught:
        check_writing_plan(overlong(case), FOCUSED_WRITING_PLAN)
    assert caught.value.family == "budget"


@pytest.mark.asyncio
async def test_36_to_24_uses_one_shared_attempt_and_preserves_other_prose(
    case, tmp_path
):
    original = overlong(case)
    owner = RangeProvider(case)
    owner._cost_budget = CostBudget()
    owner._cost_budget.configure_strict()
    controller = session(case, tmp_path, owner)
    targets = capacity_repair_targets(original, RANGE_WRITING_PLAN)
    assert len(targets) == 12 and all(t["maximum"] == 2 for t in targets.values())
    candidate = await repair_capacity_snapshot(controller, original)
    assert sum(len(p["sentences"]) for p in candidate["methods"]) == 24
    assert len(controller.state["attempts"]) == len(owner.calls) == 1
    for key in ("title", *[k for k in BODY if k != "methods"]):
        assert candidate[key] == original[key]
    assert original == overlong(case)
    record = controller.journal["capacity_candidates"][0]
    assert record["original"] == original
    assert record["status"] == "locally_valid_unverified"
    assert record["scientific_assessment"] == "not_verified"


@pytest.mark.parametrize(
    "mutation",
    [
        "snapshot",
        "original_hash",
        "original",
        "end",
        "boolean",
        "overlap",
        "empty",
        "legacy",
    ],
)
@pytest.mark.asyncio
async def test_range_identity_failures_are_rejected_before_any_request(
    case, tmp_path, mutation
):
    body = overlong(case)
    targets = capacity_repair_targets(body, RANGE_WRITING_PLAN)
    target = next(iter(targets.values()))
    if mutation == "snapshot":
        target["snapshot_sha256"] = "wrong"
    elif mutation == "original_hash":
        target["original_sha256"] = "wrong"
    elif mutation == "original":
        target["original"][0]["text"] = "changed"
    elif mutation == "end":
        target["end"] = 100
    elif mutation == "boolean":
        target["end"] = True
    elif mutation == "overlap":
        targets["single"] = {
            "path": ["methods", 0, 1],
            "original": copy.deepcopy(body["methods"][0]["sentences"][1]),
        }
    elif mutation == "empty":
        target["end"] = 0
    controller = session(
        case,
        tmp_path,
        plan=FOCUSED_WRITING_PLAN if mutation == "legacy" else RANGE_WRITING_PLAN,
    )
    with pytest.raises(WriterError, match="identity_invalid"):
        await controller.patch(body, targets, cause="local")
    assert not controller.llm.calls and not controller.state["attempts"]


def test_incomplete_snapshot_cannot_be_used_to_project_full_reassessment(case):
    body = overlong(case)
    del body["open_problems"]
    with pytest.raises(WriterError, match="snapshot_incomplete"):
        capacity_repair_targets(body, RANGE_WRITING_PLAN)


def test_paragraphs_cannot_be_deleted_to_satisfy_a_numeric_ceiling(case):
    body = overlong(case)
    unit = copy.deepcopy(body["methods"][0]["sentences"][0])
    body["methods"] = [{"sentences": [copy.deepcopy(unit)]} for _ in range(25)]
    with pytest.raises(WriterError, match="repair_unavailable") as caught:
        capacity_repair_targets(body, RANGE_WRITING_PLAN)
    assert caught.value.family == "quality" and len(body["methods"]) == 25


@pytest.mark.asyncio
async def test_disjoint_ranges_are_applied_backwards_without_moving_protected_units(
    case, tmp_path
):
    body = copy.deepcopy(case.body)
    unit = body["methods"][0]["sentences"][0]
    originals = [
        {**unit, "text": f"Finding {chr(65+i)}: {unit['text']}"} for i in range(5)
    ]
    body["methods"][0]["sentences"] = copy.deepcopy(originals)
    targets = {}
    for start, end in ((0, 2), (3, 5)):
        original = copy.deepcopy(originals[start:end])
        targets[str(start)] = {
            "path": ["methods", 0, start],
            "end": end,
            "original": original,
            "original_sha256": digest(original),
            "snapshot_sha256": digest(body),
            "maximum": 1,
        }
    controller = session(case, tmp_path, RangeProvider(case))
    candidate = await controller.patch(body, targets, cause="evaluation")
    assert candidate["methods"][0]["sentences"] == [
        originals[0],
        originals[2],
        originals[3],
    ]
    assert body["methods"][0]["sentences"] == originals


@pytest.mark.asyncio
async def test_body_change_during_transport_invalidates_the_exact_snapshot(
    case, tmp_path
):
    body = overlong(case)

    class ConcurrentEdit(RangeProvider):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            body["title"] = "A different current version"
            return response

    controller = session(case, tmp_path, ConcurrentEdit(case))
    with pytest.raises(WriterError, match="patch_identity_invalid"):
        await repair_capacity_snapshot(controller, body)
    assert len(controller.state["attempts"]) == 1
    assert controller.journal["capacity_candidates"][0]["status"] == "rejected"


@pytest.mark.asyncio
async def test_caller_cannot_redirect_an_inflight_authorized_range(case, tmp_path):
    body = overlong(case)
    targets = capacity_repair_targets(body, RANGE_WRITING_PLAN)
    original = copy.deepcopy(targets)

    class Redirect(RangeProvider):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            targets.clear()
            return response

    controller = session(case, tmp_path, Redirect(case))
    candidate = await controller.patch(body, targets, cause="evaluation")
    assert candidate["open_problems"] == body["open_problems"]
    assert sum(len(p["sentences"]) for p in candidate["methods"]) == 24
    sent = controller.llm.calls[0][0]
    assert sent["targets"] == original


@pytest.mark.asyncio
async def test_summary_source_body_cannot_change_while_a_range_patch_is_inflight(
    case, tmp_path
):
    body = copy.deepcopy(case.body)
    summary = copy.deepcopy(case.summaries)
    unit = summary["abstract"][0]["sentences"][0]
    summary["abstract"] = [
        {"sentences": [{**unit, "text": unit["text"] + str(i)} for i in range(4)]}
    ]

    class ChangedPremise(RangeProvider):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            body["title"] = "Changed source version"
            return response

    controller = session(case, tmp_path, ChangedPremise(case))
    with pytest.raises(WriterError, match="patch_identity_invalid"):
        await repair_capacity_snapshot(controller, summary, summary_body=body)
    assert len(controller.state["attempts"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["cancelled", "unknown_usage", "bad_json", "declined"]
)
async def test_terminal_failures_and_duplicate_submissions_never_trigger_another_attempt(
    case, tmp_path, fault
):
    class Failing(RangeProvider):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            if "targets" in json.loads(messages[1]["content"]):
                if fault == "cancelled":
                    raise asyncio.CancelledError()
                if fault == "unknown_usage":
                    response.usage = {}
                if fault == "bad_json":
                    response.content = "{"
            return response

    owner = Failing(case, decline=fault == "declined")
    controller = session(case, tmp_path, owner)
    with pytest.raises((WriterError, asyncio.CancelledError)):
        await repair_capacity_snapshot(controller, overlong(case))
    assert len(owner.calls) == len(controller.state["attempts"]) == 1
    assert controller.journal["capacity_candidates"][0]["status"] == "rejected"
    resumed = session(case, tmp_path, owner)
    with pytest.raises(WriterError, match="snapshot_no_replay"):
        await repair_capacity_snapshot(resumed, overlong(case))
    assert len(owner.calls) == 1


@pytest.mark.asyncio
async def test_summary_preflight_counts_complete_body_and_pending_reviewer(
    case, tmp_path
):
    owner = RangeProvider(case)
    owner._cost_budget = CostBudget(max_tokens=250000)
    owner._cost_budget.configure_strict()
    controller = session(case, tmp_path, owner)
    summary = copy.deepcopy(case.summaries)
    unit = summary["abstract"][0]["sentences"][0]
    summary["abstract"] = [
        {"sentences": [{**unit, "text": unit["text"] + str(i)} for i in range(4)]}
    ]
    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"][0]["text"] += "x" * 1900
    before = owner._cost_budget.snapshot()
    with pytest.raises(WriterError, match="transaction_budget") as caught:
        await repair_capacity_snapshot(controller, summary, summary_body=body)
    detail = caught.value.diagnostics[0]
    assert detail["projected_model_assertions"] == 128
    assert detail["reviewer_reservation"] == int(131072 * 0.9)
    assert (
        detail["reassessment_reservation"]
        > 10 * len(json.dumps(summary).encode()) + 10 * 16384
    )
    assert owner._cost_budget.snapshot() == before
    assert not owner.calls and not controller.state["attempts"]


def test_priority_recommendations_never_change_the_frozen_pool(case, tmp_path):
    controller = session(case, tmp_path)
    packet = copy.deepcopy(case.packet)
    system, payload = controller.planned_input("system", packet)
    assert payload["sources"] == case.packet["sources"]
    assert payload["cards"] == case.packet["cards"]
    assert payload["requirements"] == case.packet["requirements"]
    assert payload["chapter_recommendations"] == case.packet["chapter_recommendations"]
    assert (
        payload["host_priority_card_refs"]["methods"]
        == packet["chapter_recommendations"]["methods"][:6]
    )
    assert packet == case.packet and "citation whitelist" in system


def test_six_priority_cards_keep_all_twenty_sources_and_allow_other_citations(tmp_path):
    from test_complete_native_repair import source_selection

    from litagent.quality import build_fact_cards
    from litagent.writing_v2 import SCOPE, compile_document, source_packet

    ledger = {
        f"e{i}": {
            "evidence_id": f"e{i}",
            "paper_id": f"p{i}",
            "text": f"Method {chr(65+i)} supports a documented comparison.",
            "supporting_text": f"Method {chr(65+i)} supports a documented comparison.",
            "chunk_key": "abstract",
        }
        for i in range(20)
    }
    cards = build_fact_cards(ledger)
    selection = source_selection(cards)
    selection.writing_scope_policy = SCOPE
    query = "Compare all explicitly requested methods, preserving their conditions."
    packet, aliases = source_packet(
        cards,
        selection,
        query,
        [{"requirement_id": "question", "kind": "question", "text": query}],
        "en",
    )
    original = copy.deepcopy(packet)
    controller = WriterSession(
        Provider(None),
        tmp_path / "twenty",
        "r",
        packet,
        aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=RANGE_WRITING_PLAN,
    )
    _, planned = controller.planned_input("system", packet)
    assert len(planned["host_priority_card_refs"]["methods"]) == 6
    assert len(planned["cards"]) == len(planned["sources"]) == 20
    assert packet == original and planned["requirements"] == original["requirements"]
    assert planned["chapter_recommendations"] == original["chapter_recommendations"]
    assert (
        planned["cards"] == original["cards"]
        and planned["sources"] == original["sources"]
    )
    ref = next(
        r for r in aliases if r not in planned["host_priority_card_refs"]["methods"]
    )
    body = {
        "title": "Comparison",
        **{
            k: [
                {
                    "sentences": [
                        {
                            "text": "An independently unverified finding.",
                            "fact_card_refs": [ref],
                        }
                    ]
                }
            ]
            for k in BODY
        },
    }
    summaries = {
        k: [
            {
                "sentences": [
                    {
                        "text": "A separately verified summary is required.",
                        "body_unit_refs": ["u3"],
                    }
                ]
            }
        ]
        for k in ("abstract", "conclusion")
    }
    draft, _ = compile_document(
        body, summaries, cards, aliases, "en", writing_plan=RANGE_WRITING_PLAN
    )
    assert "[E:" + cards[aliases[ref]]["evidence_id"] + "]" in draft


def test_frozen_manifest_binds_range_authorization_and_priority_limits(manifest):
    from scripts import run_frozen_writing as campaign

    root, config = manifest
    target = root.with_name("range-policy")
    config = campaign.profile_config(config, campaign.EOF_PROFILE)
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.EOF_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
    )
    frozen = campaign.validate(target, config)
    assert frozen["writing_parameters"]["native_writing_plan"] == RANGE_WRITING_PLAN
    assert (
        frozen["writing_plan_spec"]["repair_scope_contract"]
        == "host-paragraph-ranges-v1"
    )
    assert frozen["writing_plan_spec"]["priority_card_limits"]["methods"] == 6
    assert (
        campaign.profile_description(
            campaign.EOF_PROFILE, frozen["config"], writing_plan=RANGE_WRITING_PLAN
        )["writing_plan"]
        == frozen["writing_plan_spec"]
    )


@pytest.mark.asyncio
async def test_repeated_overlong_replacement_spends_two_attempts_and_keeps_original(
    case, tmp_path
):
    class TooLong(RangeProvider):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "targets" in payload:
                result = json.loads(response.content)
                for patch in result["patches"]:
                    patch["sentences"] = payload["targets"][patch["target_id"]][
                        "original"
                    ]
                response.content = json.dumps(result)
            return response

    body = overlong(case)
    owner = TooLong(case)
    controller = session(case, tmp_path, owner)
    with pytest.raises(WriterError, match="patch_capacity_exceeded"):
        await repair_capacity_snapshot(controller, body)
    assert len(owner.calls) == len(controller.state["attempts"]) == 2
    assert (
        body == overlong(case)
        and controller.journal["capacity_candidates"][0]["original"] == body
    )
    resumed = session(case, tmp_path, owner)
    with pytest.raises(WriterError, match="repair_limit"):
        await repair_capacity_snapshot(resumed, body)
    assert len(owner.calls) == 2
