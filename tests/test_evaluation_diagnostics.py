"""Preserve original inventory failures across real SDK and host partitions."""

import copy
import hashlib
import json

import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    _incomplete_fragment_diagnostic,
    build_source_fragments,
    evaluate_scoped,
)
from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest
from litagent.report_render import build_report_source_units
from litagent.writing_v2 import compile_document


class DiagnosticGateway(FlatGateway):
    """Change provider inventory only; keep locations and all consumers real."""

    def __init__(self, case, fault):
        super().__init__(case)
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "allowed_citation_aliases" not in payload:
                raise AssertionError("incomplete_inventory_must_stop_before_assessment")
            value = json.loads(response.content)
            for fragment in value["fragments"].values():
                if fault == "excluded":
                    fragment.update(assertions=[], exclusion_reason="No assertion.")
                else:
                    for claim in fragment["assertions"]:
                        claim["kind"] = "report_structure"
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


def context_for(case, bindings=None):
    """Keep the frozen evidence and actual writer bindings in evaluator input."""
    return {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        **({"writing_bindings": bindings} if bindings is not None else {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fault", ["excluded", "classification"])
async def test_public_consumer_preserves_exact_primary_error_and_locations(
    case, native, fault
):
    case.body["methods"][0]["sentences"][0].update(
        text="EViT reduces computation.", fact_card_refs=[case.by_eid["a"]]
    )
    case.body["taxonomy"] = copy.deepcopy(case.body["methods"])
    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    gateway = DiagnosticGateway(case, fault)
    owner, budget, events = await owner_for(gateway)
    before = copy.deepcopy(bindings)
    checkpoints = []
    context = context_for(case, bindings if native else None)
    context["evaluation_checkpoint"] = lambda p: checkpoints.append(copy.deepcopy(p))
    try:
        with pytest.raises(ScopedEvaluationError) as caught:
            await evaluate_scoped(
                llm=ModelBoundClient(owner, "lite", "evaluation"),
                survey=draft,
                context=context,
                config=EvalConfig(thinking_mode="disabled"),
            )
        assert str(caught.value) == "evaluation_extraction_incomplete"
        details = caught.value.details
        assert (
            details["source_report_sha256"]
            == hashlib.sha256(draft.encode()).hexdigest()
        )
        assert not details["coverage"]["complete"]
        assert len(details["fact_card_map"]) == len(case.cards)
        issues = details["unsupported_claims"]
        assert issues and len(issues) == len(details["incomplete_fragments"])
        for issue in issues:
            assert issue["diagnostic_location_status"] == "located"
            assert (
                draft[issue["source_start"] : issue["source_end"]]
                == issue["source_quote"]
            )
            assert issue["reason_code"] == "evaluation_extraction_incomplete"
            assert bool(":native:" in issue["source_fragment_id"]) is native
        repeated = [i for i in issues if i["source_quote"].startswith("EViT")]
        assert len(repeated) == 2
        assert repeated[0]["source_quote"] == repeated[1]["source_quote"]
        assert repeated[0]["source_start"] != repeated[1]["source_start"]
        assert len(gateway.sent) == 1 and budget.snapshot()["requests_sent"] == 1
        assert not budget.snapshot()["usage_uncertain"]
        assert any(e == "llm.complete" and d["usage_available"] for e, d in events)
        assert bindings == before
        assert "_host_source_partition" not in context
        assert all("_host_source_partition" not in p for p in checkpoints)
        assert all(
            "_host_source_partition" not in str(r["payload"]) for r in gateway.sent
        )
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "fault",
    ["missing", "sha", "unit", "fragment", "quote", "start", "citations", "shape"],
)
def test_unavailable_location_never_repartitions_or_borrows_citations(fault):
    draft = "## Methods\nSame result. [E:a]\n\nSame result. [E:b]\n"
    units = build_report_source_units(draft)
    fragments = {
        u.unit_id: {f["source_fragment_id"]: f for f in build_source_fragments(u)}
        for u in units
    }
    unit = units[0]
    fid = next(iter(fragments[unit.unit_id]))
    partition = {
        "report_sha256": hashlib.sha256(draft.encode()).hexdigest(),
        "units": {u.unit_id: u for u in units},
        "fragments": fragments,
    }
    issue = {
        "source_unit_id": unit.unit_id,
        "source_fragment_id": fid,
        "explanation": "Original extraction failure.",
        "source_quote": "Forged quote",
        "evidence_refs": ["b"],
    }
    context = {"_host_source_partition": partition}
    if fault == "missing":
        context.clear()
    elif fault == "sha":
        partition["report_sha256"] = "0" * 64
    elif fault == "unit":
        issue["source_unit_id"] = "unknown"
    elif fault == "fragment":
        issue["source_fragment_id"] = "unknown"
    elif fault == "shape":
        partition.pop("units")
    else:
        fragment = fragments[unit.unit_id][fid]
        fragment[
            {
                "quote": "source_quote",
                "start": "start",
                "citations": "citation_aliases",
            }[fault]
        ] = {"quote": "Forged quote", "start": True, "citations": ["b"]}[fault]
    result = _incomplete_fragment_diagnostic(issue, draft, context)
    assert result["explanation"] == issue["explanation"]
    assert result["diagnostic_location_status"] == "unavailable"
    assert "source_quote" not in result and "source_start" not in result
    assert "evidence_refs" not in result


@pytest.mark.parametrize("issue", [None, "invalid", {"source_unit_id": []}])
def test_invalid_diagnostic_identity_cannot_mask_primary_failure(issue):
    result = _incomplete_fragment_diagnostic(issue, "report", {})
    assert result["diagnostic_location_status"] == "unavailable"
    assert result["reason_code"] == "evaluation_extraction_incomplete"
