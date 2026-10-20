"""Exercise bounded atomic-group repair against actual failed native reports."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_complete_native_repair import source_selection
from test_evaluation_wire_contract import AnalysisTransport
from test_writing_passages import facts

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    GroundedWritingError,
    _native_unit,
    apply_native_repairs,
    digest,
    grounded_messages,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
    strict_acceptance,
)
from litagent.safety.budget import CostBudget, strict_request_reservation


def render(candidate, cards, *, language="en", selection=None, bindings=None):
    """Retain every production source and sentence-scope compiler constraint."""
    return render_grounded_report_traced(
        candidate if isinstance(candidate, str) else json.dumps(candidate),
        cards,
        "academic_v2",
        language,
        evidence_selection=selection,
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=True,
        bindings=bindings,
    )


def request(candidate, cards, *, language="en", selection=None):
    """Derive authorized targets from actual compiler diagnostics."""
    candidate = candidate if isinstance(candidate, str) else json.dumps(candidate)
    with pytest.raises(GroundedWritingError) as failed:
        render(candidate, cards, language=language, selection=selection)
    messages, _ = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            language,
            "Explain the selected methods and their limitations",
            evidence_selection=selection,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        )
    )
    messages, schema, binding = prepare_native_repair_request(
        messages,
        candidate,
        failed.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language=language,
        evidence_selection=selection,
    )
    return candidate, messages, schema, binding, failed.value.diagnostics


@pytest.fixture
def received():
    """Keep the complete 007 failure as protocol evidence, not scientific gold."""
    return json.loads(Path("tests/fixtures/native-group-repair-real.json").read_text())


def recorded_groups(received):
    """Represent two recorded clauses explicitly; never repair provider JSON."""
    response = json.loads(received["received_repairs"][0]["response"])
    splits = 0
    for patch in response["patches"].values():
        if "；" not in patch["text"]:
            continue
        first, second = patch["text"].split("；")
        extra = {**patch, "text": second}
        patch.update(text=first + "。", additional_units=[extra])
        splits += 1
    assert splits == 2
    return response


def test_received_groups_keep_protected_prose_and_relocate_summary(received):
    candidate, messages, schema, binding, diagnostics = request(
        received["candidate"],
        received["cards"],
        language="zh-CN",
        selection=received["selection"],
    )
    assert diagnostics == received["initial_diagnostics"]
    assert binding["version"] == "complete-native-unit-repair-v5"
    response = recorded_groups(received)
    Draft202012Validator(schema["json_schema"]["schema"]).validate(response)
    revised, proof = apply_native_repairs(
        candidate, json.dumps(response), binding, cards=received["cards"]
    )
    render(
        revised,
        received["cards"],
        language="zh-CN",
        selection=received["selection"],
    )
    original, updated = json.loads(candidate), json.loads(revised)
    targets = {t["path"] for t in binding["targets"].values()}
    assert len(targets) == 6 and len(proof["derived_summary_changes"]) == 1
    relocation = {r["original_path"]: r for r in proof["relocations"]}
    for section_key, section in original["sections"].items():
        for bi, block in enumerate(
            [section["first_block"], *section["additional_blocks"]]
        ):
            for si, unit in enumerate(
                [block["first_sentence"], *block["additional_sentences"]]
            ):
                path = (
                    f"sections.{section_key}."
                    + ("first_block" if bi == 0 else f"additional_blocks.{bi - 1}")
                    + "."
                    + (
                        "first_sentence"
                        if si == 0
                        else f"additional_sentences.{si - 1}"
                    )
                )
                if path in targets:
                    assert relocation[path]["authored_target"]
                    continue
                new_path = relocation.get(path, {"new_paths": [path]})["new_paths"][0]
                actual = copy.deepcopy(_native_unit(updated, new_path))
                if section_key in {"abstract", "conclusion"}:
                    actual["body_references"] = unit["body_references"]
                assert actual == unit
    assert strict_request_reservation(messages, 20000, schema) < 117964
    assert proof["input_sha256"] == binding["candidate_sha256"]
    assert proof["output_sha256"] == hashlib.sha256(revised.encode()).hexdigest()
    assert received["candidate"] == candidate


def test_received_v4_response_keeps_its_original_semicolon_failure(received):
    candidate, _ = apply_native_repairs(
        received["candidate"],
        received["legacy_v4_response"],
        received["legacy_v4_binding"],
        cards=received["cards"],
    )
    assert candidate == received["legacy_v4_applied_candidate"]
    with pytest.raises(GroundedWritingError, match="citation_scope"):
        render(
            candidate,
            received["cards"],
            language="zh-CN",
            selection=received["selection"],
        )
    response = json.loads(received["legacy_v4_response"])
    patch = next(iter(response["patches"].values()))
    patch["additional_units"] = []
    with pytest.raises(ValueError, match="strict_writing_scope_notice_invalid"):
        apply_native_repairs(
            received["candidate"],
            json.dumps(response),
            received["legacy_v4_binding"],
            cards=received["cards"],
        )


@pytest.mark.parametrize(
    "fault",
    ["null", "oversized", "nested", "type", "card", "notice", "target", "sha"],
)
def test_malformed_groups_never_change_original_or_permissions(received, fault):
    candidate, _, _, binding, _ = request(
        received["candidate"],
        received["cards"],
        language="zh-CN",
        selection=received["selection"],
    )
    response = recorded_groups(received)
    patch = next(p for p in response["patches"].values() if p.get("additional_units"))
    extra = patch["additional_units"][0]
    if fault == "null":
        patch["additional_units"] = None
    elif fault == "oversized":
        patch["additional_units"] *= 4
    elif fault == "nested":
        extra["additional_units"] = []
    elif fault == "type":
        patch["additional_units"] = ["not a unit"]
    elif fault == "card":
        extra["fact_card_ids"] = ["f:unknown"]
    elif fault == "notice":
        extra.update(
            text="",
            kind="organization",
            fact_card_ids=[],
            scope_notice_id="review_scope",
        )
    elif fault == "target":
        response["patches"]["u999"] = extra
    else:
        candidate += " "
    before = copy.deepcopy(received)
    with pytest.raises(ValueError):
        apply_native_repairs(
            candidate, json.dumps(response), binding, cards=received["cards"]
        )
    assert received == before


def test_multiple_targets_stay_in_place_and_shift_untouched_dependency():
    cards = facts()
    document = analysis_document(cards)
    block = document["sections"]["methods"]["first_block"]
    base = copy.deepcopy(block["first_sentence"])
    block["first_sentence"]["text"] = "First fact; second fact."
    block["additional_sentences"] = [
        {**base, "text": "Unchanged intermediate condition."},
        {**base, "text": "Third fact; fourth fact."},
        {**base, "text": "Unchanged final condition."},
    ]
    for section, index in (("abstract", 1), ("conclusion", 3)):
        document["sections"][section]["first_block"]["first_sentence"][
            "body_references"
        ][0]["sentence_index"] = index
    candidate, _, _, binding, _ = request(document, cards)
    assert len(binding["targets"]) == 2
    patches = {}
    for key, target in binding["targets"].items():
        ordinal = "First" if target["path"].endswith("first_sentence") else "Third"
        primary = {**base, "text": f"{ordinal} result under condition A."}
        primary["additional_units"] = [
            {**base, "text": f"{ordinal} result under condition B."}
        ]
        patches[key] = primary
    revised, proof = apply_native_repairs(
        candidate,
        json.dumps({"can_repair": True, "patches": patches}),
        binding,
        cards=cards,
    )
    revised = json.loads(revised)
    updated = revised["sections"]["methods"]["first_block"]
    units = [updated["first_sentence"], *updated["additional_sentences"]]
    assert [u["text"] for u in units] == [
        "First result under condition A.",
        "First result under condition B.",
        "Unchanged intermediate condition.",
        "Third result under condition A.",
        "Third result under condition B.",
        "Unchanged final condition.",
    ]
    assert [
        revised["sections"][s]["first_block"]["first_sentence"]["body_references"][0][
            "sentence_index"
        ]
        for s in ("abstract", "conclusion")
    ] == [2, 5]
    assert len(proof["derived_summary_changes"]) == 2
    render(revised, cards)


def test_summary_groups_bind_to_original_body_catalog_and_expand_without_trimming():
    cards = facts()
    document = analysis_document(cards)
    document["sections"]["methods"]["first_block"]["first_sentence"][
        "text"
    ] = "First fact; second fact."
    document["sections"]["abstract"]["first_block"]["first_sentence"][
        "text"
    ] = "First summary; second summary."
    document["sections"]["abstract"]["first_block"]["first_sentence"][
        "body_references"
    ][0]["section_key"] = "experiments"
    candidate, _, schema, binding, _ = request(document, cards)
    patches = {}
    for key, target in binding["targets"].items():
        if target.get("unit_type") == "summary":
            uid = next(
                uid
                for uid, ref in binding["body_unit_catalog"].items()
                if ref["section_key"] == "methods"
            )
            patches[key] = {
                "text": "First summary.",
                "body_unit_ids": [uid],
                "additional_units": [
                    {"text": "Second summary.", "body_unit_ids": [uid]}
                ],
            }
        else:
            primary = copy.deepcopy(_native_unit(document, target["path"]))
            primary["text"] = "First fact."
            primary["additional_units"] = [{**primary, "text": "Second fact."}]
            patches[key] = primary
    response = {"can_repair": True, "patches": patches}
    Draft202012Validator(schema["json_schema"]["schema"]).validate(response)
    revised, _ = apply_native_repairs(
        candidate, json.dumps(response), binding, cards=cards
    )
    abstract = json.loads(revised)["sections"]["abstract"]["first_block"]
    assert len(abstract["additional_sentences"]) == 1
    assert all(
        len(u["body_references"]) == 2
        for u in [abstract["first_sentence"], *abstract["additional_sentences"]]
    )
    render(revised, cards)


def test_group_expansion_never_trims_summary_to_fit_existing_capacity():
    """A larger dependency group must fail rather than lose scientific findings."""
    cards = facts()
    document = analysis_document(cards)
    section = document["sections"]["methods"]
    section["first_block"]["first_sentence"]["text"] = "First fact; second fact."
    section["additional_blocks"].append(copy.deepcopy(section["first_block"]))
    for key in ("abstract", "conclusion"):
        refs = document["sections"][key]["first_block"]["first_sentence"][
            "body_references"
        ]
        refs.append({"section_key": "methods", "block_index": 1, "sentence_index": 0})
    candidate, _, _, binding, _ = request(document, cards)
    patches = {}
    for key, target in binding["targets"].items():
        assert target.get("unit_type") != "summary"
        primary = copy.deepcopy(_native_unit(document, target["path"]))
        primary["text"] = "One independently cited fact."
        primary["additional_units"] = [
            {**primary, "text": f"Another fact under condition {index}."}
            for index in range(3)
        ]
        patches[key] = primary
    revised, _ = apply_native_repairs(
        candidate,
        json.dumps({"can_repair": True, "patches": patches}),
        binding,
        cards=cards,
    )
    for key in ("abstract", "conclusion"):
        assert (
            len(
                json.loads(revised)["sections"][key]["first_block"]["first_sentence"][
                    "body_references"
                ]
            )
            == 8
        )
    with pytest.raises(GroundedWritingError):
        render(revised, cards)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_number", [False, True])
@pytest.mark.parametrize("rejected_first", [False, True])
async def test_sdk_synthesis_reviewer_and_full_evaluation_visit_every_new_atom(
    tmp_path, wrong_number, rejected_first
):
    """Mock only HTTP; fabricated semantic decisions cannot bless a wrong number."""
    source = "EViT improves speed by 50% on DeiT-S. EViT is evaluated on DeiT-S."
    cards = facts(source)
    document = analysis_document(cards)
    document["sections"]["methods"]["first_block"]["first_sentence"][
        "text"
    ] = "EViT improves speed by 50% on DeiT-S; EViT is evaluated on DeiT-S."
    _, _, _, binding, _ = request(document, cards)
    atom = (
        "EViT improves speed by 90% on DeiT-S."
        if wrong_number
        else "EViT is evaluated on DeiT-S."
    )
    patch = {
        "can_repair": True,
        "patches": {
            key: {
                "text": "EViT improves speed by 50% on DeiT-S.",
                "kind": "scientific",
                "scope_notice_id": "none",
                "fact_card_ids": list(cards),
                "additional_units": [
                    {
                        "text": atom,
                        "kind": "scientific",
                        "scope_notice_id": "none",
                        "fact_card_ids": list(cards),
                    }
                ],
            }
            for key in binding["targets"]
        },
    }
    wire, snapshots = [], []
    evaluator = AnalysisTransport()

    reviewer = {
        "score": 0.9,
        "verdict": "accept",
        "evidence_compliance": {
            "passed": True,
            "unsupported_claims": [],
            "unknown_evidence_ids": [],
        },
    }
    from litagent.native_wire import array_from_native

    outputs = [array_from_native(document)]
    if rejected_first:
        rejected = copy.deepcopy(patch)
        next(iter(rejected["patches"].values()))["additional_units"][0][
            "text"
        ] += "; extra claim."
        outputs.append(rejected)
    outputs.extend([patch, reviewer])

    async def transport(request):
        payload = json.loads(request.content)
        wire.append(payload)
        if len(wire) <= len(outputs):
            content = json.dumps(outputs[len(wire) - 1])
        else:
            controlled = await evaluator.chat(
                payload["messages"], response_format=payload["response_format"]
            )
            content = controlled.content
        return httpx.Response(
            200,
            json={
                "id": f"offline-{len(wire)}",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "lite", api_key="offline", cost_budget=budget
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    pro, lite = ModelBoundClient(owner, "pro", "generation"), ModelBoundClient(
        owner, "lite", "evaluation"
    )
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=source_selection(cards))
        ),
        run_id_provider=lambda: "atomic-groups",
        repair_state_path_provider=lambda: tmp_path / "repairs.json",
        thinking_mode="disabled",
    )
    worker = AdversarialReviewWorker(
        pro,
        synthesis,
        ReviewerWorker(lite),
        max_rounds=1,
    )
    task = SubTask(
        "adversarial_review",
        "write",
        "adversarial_review",
        input_data={
            "query": "Explain EViT speed on DeiT-S",
            "quality_policy": "strict_auto_v2",
            "review_structure": "academic_v2",
            "body_quality": "analysis_v1",
            "report_language": "en",
            "verification": {"judge": False, "jev": False},
        },
    )
    token = None
    try:
        result = await worker.execute(task)
        assert [p["model"] for p in wire] == ["pro"] * (2 + int(rejected_first)) + [
            "lite"
        ]
        if rejected_first:
            previous = json.loads(wire[2]["messages"][1]["content"])[
                "previous_local_failure"
            ]
            assert previous["repair_relocations"] and previous["migration_proof_sha256"]
            assert (
                json.loads(wire[1]["messages"][1]["content"])["current_candidate"]
                == json.loads(wire[2]["messages"][1]["content"])["current_candidate"]
            )
        token = set_task_id("evaluation.initial")
        evaluated = await evaluate_scoped(
            llm=lite,
            survey=result["final_draft"],
            context={
                "query": "Explain EViT speed on DeiT-S",
                "quality_policy": "strict_auto_v2",
                "body_quality": "analysis_v1",
                "review_structure": "academic_v2",
                "report_language": "en",
                "scope_notice_contract": "scope-notice-v1",
                "writing_bindings": result["writing_bindings"],
                "review_methodology": result["review_methodology"],
                "fact_cards": cards,
                "evidence_snapshot_sha256": digest(cards),
                "evidence": {
                    "a": {
                        "evidence_id": "a",
                        "paper_id": "evit",
                        "text": source,
                        "supporting_text": source,
                        "chunk_key": "abstract",
                    }
                },
                "coverage_requirements": [
                    {
                        "requirement_id": "question",
                        "kind": "question",
                        "text": "Explain EViT speed on DeiT-S",
                    }
                ],
                "evaluation_checkpoint": lambda value: snapshots.append(
                    copy.deepcopy(value)
                ),
            },
            config=EvalConfig(),
        )
        atoms = [
            c for c in evaluated["assertions"] if atom.rstrip(".") in c["claim_text"]
        ]
        assert atoms
        if wrong_number:
            assert not evaluated["coverage"]["complete"] and evaluated["score"] is None
            assert all(
                c["status"] == "unverifiable"
                and c["local_error"] == "support_number_unverified"
                for c in atoms
            ), atoms
        else:
            assert evaluated["coverage"]["complete"] and evaluated["score"] == 1
            assert all(c["status"] == "supported" for c in atoms)
        # A plumbing fixture deliberately fails academic analysis; it is not gold.
        assert strict_acceptance(evaluated)["status"] == "failed"
        assert evaluated["analysis_check"]["status"] == "failed"
        assert len(wire) == len(outputs) + 3 and budget.used == len(wire) * 20
        assert all(p["model"] == "lite" for p in wire[len(outputs) :])
        assert snapshots[-1]["global_check"] == "completed"
        saved = json.loads((tmp_path / "repairs.json").read_text())
        assert saved["native_attempt_0"]["repair_proof"]["relocations"]
        assert len(saved["state"]["attempts"]) == 1 + int(rejected_first)
    finally:
        if token is not None:
            reset_task_id(token)
        await owner.close()
