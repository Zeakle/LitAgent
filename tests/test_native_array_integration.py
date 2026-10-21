"""Exercise explicit array generation while preserving canonical scientific gates."""

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator
from test_body_analysis import analysis_document, render_analysis
from test_writing_passages import facts

from litagent.body_quality import NATIVE_ARRAY_WIRE_CONTRACT
from litagent.native_wire import (
    array_from_native,
    array_wire_schema,
    canonical_from_array,
)
from litagent.quality import (
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
    project_grounded_response,
)
from litagent.workbench import RunOptions
from scripts import probe_native_array_wire as experimental


def request(cards, language="en", **kwargs):
    """Use actual preparation and schema producers, including section permissions."""
    return prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            language,
            "Explain EViT on DeiT-S",
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_production_codec_retains_values_and_canonical_schema(language):
    cards = facts()
    original = analysis_document(cards, language)
    canonical_schema = request(cards, language)[1]["json_schema"]["schema"]
    messages, rf = request(cards, language, array_wire=True)
    wire = array_from_native(original)
    before = copy.deepcopy(wire)
    raw = json.dumps(wire, ensure_ascii=False)
    events = []
    result, proof = project_grounded_response(
        raw, rf, trace_hook=lambda *a: events.append(a)
    )
    assert json.loads(result) == original and wire == before
    Draft202012Validator(canonical_schema).validate(json.loads(result))
    Draft202012Validator(rf["json_schema"]["schema"]).validate(wire)
    assert proof["contract"] == NATIVE_ARRAY_WIRE_CONTRACT
    assert proof["scientific_verdict"] == "not_evaluated"
    assert proof["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert (
        proof["canonical_candidate_sha256"]
        == hashlib.sha256(result.encode()).hexdigest()
    )
    assert len(proof["positions"]) == 7
    assert events[0][0] == "writing.wire.projected"
    assert render_analysis(json.loads(result), cards, language) == render_analysis(
        original, cards, language
    )
    packet = json.loads(messages[1]["content"])
    assert packet["native_wire_contract"] == NATIVE_ARRAY_WIRE_CONTRACT
    system = messages[0]["content"]
    for old in (
        "first_block",
        "additional_blocks",
        "first_sentence",
        "additional_sentences",
    ):
        assert old not in system
    assert "body_references" in system and "fact_card_ids" in system
    assert "sources[source_ref].source_quote in full" in system
    assert all("source_quote" in s for s in packet["sources"].values())


def test_candidate_codec_matches_frozen_experimental_value_projection():
    original = analysis_document(facts())
    expected_wire = experimental.array_from_native(original)
    assert array_from_native(original) == expected_wire
    decoded, proof = canonical_from_array(expected_wire)
    assert decoded == original and proof["scientific_verdict"] == "not_evaluated"
    schema = request(facts())[1]["json_schema"]["schema"]
    assert array_wire_schema(schema) == experimental.array_wire_schema(schema)


@pytest.mark.parametrize(
    "fault",
    [
        "legacy",
        "mixed",
        "empty",
        "missing",
        "unknown",
        "duplicate",
        "bad_reference",
        "first_notice",
    ],
)
def test_new_provider_wire_is_explicit_and_never_silently_repaired(fault):
    cards = facts()
    document = array_from_native(analysis_document(cards))
    if fault == "legacy":
        document = analysis_document(cards)
    elif fault == "mixed":
        document["sections"]["methods"]["first_block"] = {}
    elif fault == "empty":
        document["sections"]["methods"]["blocks"] = []
    elif fault == "missing":
        del document["sections"]["abstract"]
    elif fault == "unknown":
        document["sections"]["methods"]["blocks"][0]["sentences"][0]["invented"] = 1
    elif fault == "bad_reference":
        document["sections"]["abstract"]["blocks"][0]["sentences"][0][
            "body_references"
        ][0]["sentence_index"] = True
    elif fault == "first_notice":
        block = document["sections"]["open_problems"]["blocks"][0]
        block["sentences"].insert(
            0,
            {
                "kind": "gap",
                "text": "",
                "fact_card_ids": [],
                "scope_notice_id": "evidence_limit",
            },
        )
    elif fault == "wrong_card":
        document["sections"]["methods"]["blocks"][0]["sentences"][0][
            "fact_card_ids"
        ] = ["unknown"]
    raw = json.dumps(document)
    if fault == "duplicate":
        raw = raw.replace('"title":', '"title":"ignored", "title":', 1)
    before = copy.deepcopy(document)
    with pytest.raises(ValueError, match="array_wire_contract_invalid"):
        project_grounded_response(raw, request(cards, array_wire=True)[1])
    assert document == before


def test_valid_transport_does_not_waive_summary_limits_or_semantic_checks():
    cards = facts()
    document = analysis_document(cards)
    section = document["sections"]["abstract"]
    section["additional_blocks"] = [
        copy.deepcopy(section["first_block"]) for _ in range(2)
    ]
    raw = json.dumps(array_from_native(document))
    canonical, _ = project_grounded_response(raw, request(cards, array_wire=True)[1])
    with pytest.raises(GroundedWritingError):
        render_analysis(json.loads(canonical), cards)
    wrong = analysis_document(cards)
    wrong["sections"]["methods"]["first_block"]["first_sentence"][
        "text"
    ] = "EViT improves speed by 90% on DeiT-S."
    canonical, proof = project_grounded_response(
        json.dumps(array_from_native(wrong)), request(cards, array_wire=True)[1]
    )
    assert "90%" in canonical and proof["scientific_verdict"] == "not_evaluated"


def test_json_failure_preserves_offsets_and_original_bytes():
    raw = '{"title":"broken"'
    events = []
    with pytest.raises(ValueError, match="array_wire_contract_invalid") as failed:
        project_grounded_response(
            raw,
            request(facts(), array_wire=True)[1],
            trace_hook=lambda *a: events.append(a),
        )
    assert isinstance(failed.value.__cause__, json.JSONDecodeError)
    assert events[0][1]["json_position"] == len(raw)
    assert (
        events[0][1]["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    )
    assert raw == '{"title":"broken"'


def test_array_identity_is_limited_to_analysis_and_legacy_reads_stay_exact(monkeypatch):
    import litagent.body_quality as body

    old = RunOptions()
    current = RunOptions(
        quality_policy="strict_auto_v2",
        review_structure="academic_v2",
        body_quality="analysis_v1",
    )
    old_id, current_id = old.fingerprint(), current.fingerprint()
    monkeypatch.setattr(body, "NATIVE_ARRAY_WIRE_CONTRACT", "native-array-wire-future")
    assert old.fingerprint() == old_id and current.fingerprint() != current_id
    raw = json.dumps(analysis_document(facts()))
    assert project_grounded_response(raw, request(facts())[1]) == (raw, None)


def test_source_enum_errors_remain_located_and_repairable_in_canonical_compiler():
    cards = facts()
    document = array_from_native(analysis_document(cards))
    document["sections"]["methods"]["blocks"][0]["sentences"][0]["fact_card_ids"] = [
        "unknown"
    ]
    candidate, proof = project_grounded_response(
        json.dumps(document), request(cards, array_wire=True)[1]
    )
    assert proof["schema_status"] == "requires_host_diagnostics"
    assert proof["deferred_source_paths"]
    with pytest.raises(GroundedWritingError) as failed:
        render_analysis(json.loads(candidate), cards)
    assert (
        failed.value.candidate_sha256 == hashlib.sha256(candidate.encode()).hexdigest()
    )
    assert any(d["section_key"] == "methods" for d in failed.value.diagnostics)
    assert proof["scientific_verdict"] == "not_evaluated"


def test_invalid_summary_targets_reach_existing_located_repair_contract():
    cards = facts()
    document = array_from_native(analysis_document(cards))
    document["sections"]["abstract"]["blocks"][0]["sentences"][0]["body_references"][0][
        "sentence_index"
    ] = 500
    candidate, _ = project_grounded_response(
        json.dumps(document), request(cards, array_wire=True)[1]
    )
    with pytest.raises(GroundedWritingError) as failed:
        render_analysis(json.loads(candidate), cards)
    assert any(
        d["section_key"] == "abstract" and "summary" in d["reason_code"]
        for d in failed.value.diagnostics
    )


@pytest.mark.parametrize("fault", ["empty", "negative", "duplicate", "not_scientific"])
def test_summary_identity_errors_are_still_rejected_by_host_with_locations(fault):
    cards = facts()
    document = array_from_native(analysis_document(cards))
    refs = document["sections"]["abstract"]["blocks"][0]["sentences"][0][
        "body_references"
    ]
    if fault == "empty":
        refs.clear()
    elif fault == "negative":
        refs[0]["sentence_index"] = -1
    elif fault == "duplicate":
        refs.append(copy.deepcopy(refs[0]))
    else:
        document["sections"][refs[0]["section_key"]]["blocks"][0]["sentences"][
            0
        ].update(
            kind="organization",
            text="",
            fact_card_ids=[],
            scope_notice_id="review_scope",
        )
    candidate, _ = project_grounded_response(
        json.dumps(document), request(cards, array_wire=True)[1]
    )
    with pytest.raises(GroundedWritingError) as failed:
        render_analysis(json.loads(candidate), cards)
    assert any(d["section_key"] == "abstract" for d in failed.value.diagnostics)


def test_observability_failure_cannot_change_projection_or_mask_json_error():
    def failed_hook(*_):
        raise RuntimeError("optional trace unavailable")

    raw = json.dumps(array_from_native(analysis_document(facts())))
    candidate, proof = project_grounded_response(
        raw, request(facts(), array_wire=True)[1], trace_hook=failed_hook
    )
    assert json.loads(candidate) == analysis_document(facts()) and proof
    with pytest.raises(ValueError, match="array_wire_contract_invalid") as failed:
        project_grounded_response(
            "{", request(facts(), array_wire=True)[1], trace_hook=failed_hook
        )
    assert isinstance(failed.value.__cause__, json.JSONDecodeError)


def test_trace_projection_exports_only_metadata_and_hashes():
    from test_frozen_trial_tracing import SDK

    from litagent.observability.recorder import RedactingTraceHook
    from litagent.observability.tracing import LangFuseTracer

    sdk = SDK()
    tracer = LangFuseTracer("", "", "")
    tracer._client = sdk
    tracer._root = sdk.root
    hook = RedactingTraceHook(tracer, payload_mode="metadata_only")
    raw = json.dumps(array_from_native(analysis_document(facts())))
    _, proof = project_grounded_response(
        raw, request(facts(), array_wire=True)[1], trace_hook=hook
    )
    data = sdk.root.updates[-1]["metadata"]["writing_wire"]
    assert (
        data["count"] == 7 and data["candidate_sha256"] == proof["raw_response_sha256"]
    )
    assert data["input_sha256"] == proof["canonical_candidate_sha256"]
    assert "EViT" not in json.dumps(sdk.root.updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("staged", [False, True])
async def test_real_reviewer_to_whole_revision_uses_explicit_array_wire(
    tmp_path, staged
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx
    from openai import AsyncOpenAI
    from test_complete_native_repair import source_selection

    from litagent.agents.adversarial import AdversarialReviewWorker
    from litagent.agents.reviewer import ReviewerWorker
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.llm.client import OpenAICompatibleClient
    from litagent.llm.routing import ModelBoundClient
    from litagent.orchestrator.task_graph import SubTask
    from litagent.safety.budget import CostBudget

    cards = facts()
    doc = analysis_document(cards)
    doc["sections"]["introduction"]["first_block"]["first_sentence"][
        "text"
    ] = "On DeiT-S, EViT improves speed by 50%."
    revised = copy.deepcopy(doc)
    revised["sections"]["introduction"]["first_block"]["first_sentence"][
        "text"
    ] = "For DeiT-S, EViT improves speed by 50%."
    requests = []

    def transport(request):
        packet = json.loads(request.content)
        requests.append(packet)
        index = len(requests) - 1
        if index in ({0, 1, 3, 4} if staged else {0, 2}):
            name = packet["response_format"]["json_schema"]["name"]
            value = array_from_native(doc if index < (2 if staged else 1) else revised)
            if staged:
                if index in {0, 3}:
                    assert name == "grounded_body_v4_typed"
                    if index == 3:
                        writing_packet = json.loads(packet["messages"][1]["content"])
                        assert "On DeiT-S" in writing_packet["current_draft"]
                        assert (
                            "Clarify this sentence"
                            in writing_packet["located_feedback"]
                        )
                        assert (
                            "address those bound criticisms"
                            in packet["messages"][0]["content"]
                        )
                    for key in ("abstract", "conclusion"):
                        del value["sections"][key]
                    from litagent.quality import _writing_scopes
                    from litagent.staged_generation import (
                        card_reference_binding,
                        encode_typed_body,
                    )

                    value = encode_typed_body(
                        value,
                        card_reference_binding(
                            cards,
                            _writing_scopes(
                                cards, "academic_v2", "en", source_selection(cards)
                            ),
                        ),
                    )
                else:
                    assert name == "grounded_summaries_v2_refs"
                    value = {
                        "sections": {
                            key: value["sections"][key]
                            for key in ("abstract", "conclusion")
                        }
                    }
                    from test_staged_native_generation import provider_summaries

                    value = provider_summaries(value, packet)
            else:
                assert name == "grounded_report_v23_arrays"
            content = json.dumps(value)
        else:
            assert index in ({2, 5} if staged else {1, 3})
            first_review = index == (2 if staged else 1)
            review_packet = json.loads(packet["messages"][1]["content"])
            content = {
                "score": 0.6 if first_review else 0.9,
                "verdict": "revise" if first_review else "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
                "issues": [],
            }
            if first_review:
                quote = next(
                    line
                    for line in review_packet["current_draft"].splitlines()
                    if line.startswith("On DeiT-S")
                )
                content["issues"] = [
                    {
                        "type": "clarity",
                        "source_quote": quote,
                        "existing_evidence_ids": ["a"],
                        "explanation": "Clarify this sentence while retaining its condition and value.",
                    }
                ]
            content = json.dumps(content)
        return httpx.Response(
            200,
            json={
                "id": str(index),
                "object": "chat.completion",
                "created": 0,
                "model": packet["model"],
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
    pro = ModelBoundClient(owner, "pro", "generation")
    lite = ModelBoundClient(owner, "lite", "evaluation")
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=source_selection(cards))
        ),
        **(
            {
                "run_id_provider": lambda: "staged-review",
                "repair_state_path_provider": lambda: tmp_path / "repair.json",
            }
            if staged
            else {}
        ),
    )
    worker = AdversarialReviewWorker(pro, synthesis, ReviewerWorker(lite), max_rounds=2)
    task = SubTask(
        "adversarial_review",
        "write",
        "adversarial_review",
        input_data={
            "query": "Explain EViT on DeiT-S",
            "quality_policy": "strict_auto_v2",
            "review_structure": "academic_v2",
            "body_quality": "analysis_v1",
            "report_language": "en",
            "verification": {"judge": False, "jev": False},
            **({"generation_strategy": "body_summary_v1"} if staged else {}),
        },
    )
    try:
        result = await worker.execute(task)
        assert [r["model"] for r in requests] == (
            ["pro", "pro", "lite", "pro", "pro", "lite"]
            if staged
            else ["pro", "lite", "pro", "lite"]
        )
        assert "For DeiT-S, EViT" in result["final_draft"]
        proof = result["writing_bindings"]["native_wire_input_projection"]
        if staged:
            assert proof["transport"] == "host_assembled_stages"
            assert set(proof["stage_response_sha256"]) == {"body", "summaries"}
        else:
            assert (
                proof["raw_response_sha256"]
                == hashlib.sha256(
                    json.dumps(array_from_native(revised)).encode()
                ).hexdigest()
            )
        assert proof["scientific_verdict"] == "not_evaluated"
        assert budget.snapshot()["requests_sent"] == (6 if staged else 4)
    finally:
        await owner.close()
