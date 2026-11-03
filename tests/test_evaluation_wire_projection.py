"""Verify lossless citation input compaction without granting scientific support."""

import copy
import hashlib
import json

import httpx
import pytest
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.body_quality import ADAPTIVE_ASSESSMENT_INPUT_CONTRACT
from litagent.eval.batched import (
    BATCH_ALIAS_INSTRUCTION,
    batch_alias_input,
    batch_payload,
    batch_prompt,
)
from litagent.eval.scoped import build_evaluation_request
from litagent.safety.budget import strict_request_reservation


@pytest.fixture
def canonical():
    """Keep original passages distinct from unsupported extracted candidates."""
    claims = [
        {
            "claim_id": f"claim_{i}",
            "kind": "scientific",
            "claim_text": "A method has a source-bound limitation.",
            "source_unit_id": "u1",
            "source_fragment_id": "f1",
            "evidence": [{"evidence_alias": "e1", "material_ref": "e1"}],
            "required_support_aspects": ["predicate", "conditions"],
        }
        for i in range(16)
    ]
    return batch_payload(
        {
            "body_analysis_contract": "body-analysis-v1",
            "assessment_input_contract": ADAPTIVE_ASSESSMENT_INPUT_CONTRACT,
            "source_units": [
                {
                    "source_unit_id": "u1",
                    "scope_context": "Earlier report conditions remain untrusted.",
                    "citation_aliases": ["e1"],
                    "source_fragments": [
                        {
                            "source_fragment_id": "f1",
                            "text": "A method has a source-bound limitation.",
                            "citation_aliases": ["e1"],
                        }
                    ],
                }
            ],
            "report_structure": [{"source_unit_id": "u1", "section": "methods"}],
            "unit_inventory": [],
            "assertions": claims,
            "cited_materials": {"e1": {"original_support_ref": "s1"}},
            "original_supports": {
                "s1": {"paper_id": "paper", "text": "Complete original conditions."}
            },
            "support_passages": {"p1": ["e1", "s1", 0, 29]},
            "host_runtime_record": {"queries": []},
        },
        claims,
    )


def test_projection_is_reversible_and_preserves_every_other_field(canonical):
    before = copy.deepcopy(canonical)
    projected = batch_alias_input(canonical)
    for claim, original in zip(projected["assertions"], before["assertions"]):
        assert claim["evidence_aliases"] == [
            r["evidence_alias"] for r in original["evidence"]
        ]
        restored = copy.deepcopy(claim)
        restored["evidence"] = [
            {"evidence_alias": a, "material_ref": a}
            for a in restored.pop("evidence_aliases")
        ]
        assert restored == original
    assert canonical == before
    for key in canonical.keys() - {"assertions"}:
        assert projected[key] == canonical[key]


@pytest.mark.parametrize(
    "fault",
    [
        "rebind",
        "unknown",
        "duplicate",
        "unit",
        "fragment",
        "kind",
        "extra",
        "ambiguous",
        "type",
    ],
)
def test_bad_ownership_or_shape_is_rejected_without_repair(canonical, fault):
    invalid = copy.deepcopy(canonical)
    claim = invalid["assertions"][0]
    if fault == "rebind":
        claim["evidence"][0]["material_ref"] = "different"
    elif fault == "unknown":
        claim["evidence"] = [{"evidence_alias": "e404", "material_ref": "e404"}]
    elif fault == "duplicate":
        claim["evidence"].append(copy.deepcopy(claim["evidence"][0]))
    elif fault == "unit":
        claim["source_unit_id"] = "other-unit"
    elif fault == "fragment":
        invalid["source_units"][0]["source_fragments"][0]["citation_aliases"] = []
    elif fault == "kind":
        claim["kind"] = "runtime_method"
    elif fault == "extra":
        claim["evidence"][0]["unrecognized"] = "must not disappear"
    elif fault == "ambiguous":
        claim["evidence_aliases"] = ["e1"]
    else:
        claim["evidence"] = None
    frozen = copy.deepcopy(invalid)
    with pytest.raises(ValueError, match="assessment_input"):
        batch_alias_input(invalid)
    assert invalid == frozen


def test_unknown_fragment_citations_and_uncited_science_are_not_erased(canonical):
    canonical["source_units"][0]["source_fragments"][0]["citation_aliases"].append(
        "unknown"
    )
    canonical["assertions"][0]["evidence"] = []
    projected = batch_alias_input(canonical)
    assert projected["assertions"][0]["evidence_aliases"] == []
    assert (
        "unknown"
        in projected["source_units"][0]["source_fragments"][0]["citation_aliases"]
    )
    assert len(projected["assertions"]) == len(canonical["assertions"])


def test_shared_builder_preserves_schema_originals_and_legacy_wire(canonical):
    native = copy.deepcopy(canonical)
    legacy = copy.deepcopy(canonical)
    legacy.pop("assessment_input_contract")
    old_messages, old_schema = build_evaluation_request(batch_prompt(legacy), legacy)
    messages, schema = build_evaluation_request(batch_prompt(native), native)
    assert schema == old_schema
    assert BATCH_ALIAS_INSTRUCTION in messages[0]["content"]
    assert BATCH_ALIAS_INSTRUCTION not in old_messages[0]["content"]
    old_input, new_input = [
        json.loads(m[1]["content"]) for m in (old_messages, messages)
    ]
    for key in ("original_supports", "support_passages", "source_units"):
        assert old_input[key] == new_input[key]
    old_sha = hashlib.sha256(
        json.dumps(old_input["original_supports"]).encode()
    ).hexdigest()
    assert (
        hashlib.sha256(json.dumps(new_input["original_supports"]).encode()).hexdigest()
        == old_sha
    )
    # This one-reference fixture proves equivalence; larger citation sets produce
    # a net saving after the fixed alias-resolution instructions are included.
    assert strict_request_reservation(messages, 16384, schema) > 0
    assert build_evaluation_request(batch_prompt(legacy), legacy) == (
        old_messages,
        old_schema,
    )


def test_single_claim_uses_exact_legacy_wire_when_explanation_costs_more(canonical):
    single = batch_payload(canonical, canonical["assertions"][:1])
    legacy = copy.deepcopy(single)
    legacy.pop("assessment_input_contract")
    assert build_evaluation_request(
        batch_prompt(single), single
    ) == build_evaluation_request(batch_prompt(legacy), legacy)


def test_many_citations_save_bytes_including_the_explanatory_prompt(canonical):
    aliases = [f"e{i}" for i in range(1, 21)]
    canonical["cited_materials"] = {a: {"original_support_ref": "s1"} for a in aliases}
    canonical["source_units"][0]["source_fragments"][0]["citation_aliases"] = aliases
    for claim in canonical["assertions"]:
        claim["evidence"] = [{"evidence_alias": a, "material_ref": a} for a in aliases]
    legacy = copy.deepcopy(canonical)
    legacy.pop("assessment_input_contract")
    old, schema = build_evaluation_request(batch_prompt(legacy), legacy)
    new, new_schema = build_evaluation_request(batch_prompt(canonical), canonical)
    assert schema == new_schema
    assert (
        strict_request_reservation(old, 16384, schema)
        - strict_request_reservation(new, 16384, new_schema)
        > 10000
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported,wire_fault", [(False, False), (True, False), (False, True)]
)
async def test_sdk_evaluation_uses_alias_lists_without_changing_scientific_gate(
    case, unsupported, wire_fault
):
    """Replace only model transport, retaining inventory, batching and strict gates."""
    from test_adaptive_budget import certified_bounds
    from test_writer_v2 import EvaluatorTransport

    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
    from litagent.llm.routing import ModelBoundClient
    from litagent.quality import digest, strict_acceptance
    from litagent.safety.planned_budget import adaptive_writer_plan
    from litagent.writing_v2 import compile_document

    scripted_model = EvaluatorTransport(unsupported=unsupported)
    seen = []
    raw_assessments = []

    async def transport(request):
        received = json.loads(request.content)
        actual = json.loads(received["messages"][1]["content"])
        messages = copy.deepcopy(received["messages"])
        if actual.get("assessment_output_contract") == "host-batch-assessment-v1":
            assert (
                actual["assessment_input_contract"]
                == ADAPTIVE_ASSESSMENT_INPUT_CONTRACT
            )
            assert all("evidence" not in c for c in actual["assertions"])
            seen.append(copy.deepcopy(actual))
            # The scripted model predates alias-list inputs. Interpret the new
            # wire binding inside the fake model, never in a production consumer.
            for claim in actual["assertions"]:
                claim["evidence"] = [
                    {"evidence_alias": a, "material_ref": a}
                    for a in claim.pop("evidence_aliases")
                ]
            messages[1]["content"] = json.dumps(actual)
        answer = await scripted_model.chat(
            messages,
            **{k: v for k, v in received.items() if k not in {"model", "messages"}},
        )
        response_content = answer.content
        if actual.get("assessment_output_contract") == "host-batch-assessment-v1":
            if wire_fault:
                invalid = json.loads(response_content)
                invalid["decisions"][0]["support_basis"][0]["support_id"] = "p999"
                response_content = json.dumps(invalid)
            raw_assessments.append(response_content)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": received["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": response_content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, budget, _ = await owner_for(transport)
    limits = certified_bounds(60000)
    limits["inventory"]["request_cap"] = limits["global"]["request_cap"] = 80000
    budget.activate_plan(
        adaptive_writer_plan(identity="complete-native-evaluation", stage_bounds=limits)
    )
    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    context = {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "writing_bindings": bindings,
        "scope_notice_contract": "scope-notice-v1",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }
    try:
        evaluation = evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=draft,
            context=context,
            config=EvalConfig(),
        )
        if wire_fault:
            with pytest.raises(
                ScopedEvaluationError, match="response_support_identity_invalid"
            ):
                await evaluation
            assert budget.used == 240 and budget.snapshot()["requests_sent"] == 2
            assert (
                json.loads(raw_assessments[0])["decisions"][0]["support_basis"][0][
                    "support_id"
                ]
                == "p999"
            )
            return
        details = await evaluation
        assert seen and details["coverage"]["complete"]
        assert (
            details["stages"]["assessment_input_contract"]
            == ADAPTIVE_ASSESSMENT_INPUT_CONTRACT
        )
        assert strict_acceptance(details)["status"] == (
            "failed" if unsupported else "passed"
        )
        assert budget.used == 360 and budget.snapshot()["requests_sent"] == 3
        assert set(seen[0]["original_supports"]) and all(
            c["evidence_aliases"] for c in seen[0]["assertions"]
        )
        assert all(
            value["text"] == case.ledger["b"]["supporting_text"]
            for value in seen[0]["original_supports"].values()
        )
        if not unsupported:
            provider_ids = {
                s["support_id"]
                for d in json.loads(raw_assessments[0])["decisions"]
                for s in d["support_basis"]
            }
            canonical_ids = {
                s["support_id"]
                for d in details["stages"]["batches"][0]["decisions"]
                for s in d["support_basis"]
            }
            assert provider_ids and provider_ids.issubset(seen[0]["support_passages"])
            assert canonical_ids and canonical_ids != provider_ids
    finally:
        await owner.close()
