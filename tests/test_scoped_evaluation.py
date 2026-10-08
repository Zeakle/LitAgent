"""Exercise versioned evaluation through actual prompts, budgets and quality gates."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from litagent.config import EvalConfig, load_config
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import (
    CLASSIFICATION_CONTRACT,
    CLASSIFICATION_RULES,
    EVALUATION_VERSION,
    EXTRACTION_CONTRACT,
    build_source_fragments,
    evaluate_scoped,
)
from litagent.llm.client import LLMResponse
from litagent.report_render import build_report_source_units
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget


class ScriptedLLM:
    """Return controlled transport outputs while capturing actual model inputs."""

    def __init__(self, inventory=None, decisions=None, *, mutate=None, budget=None):
        self.calls = []
        self.inventory = inventory
        self.decisions = decisions
        self.mutate = mutate
        self._cost_budget = budget

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[1]["content"])
        if len(self.calls) == 1:
            units = self.inventory or [
                {
                    "source_unit_id": unit["source_unit_id"],
                    "assertions": [
                        {
                            "kind": "scientific",
                            "claim_text": fragment["text"],
                            "source_fragment_id": fragment["source_fragment_id"],
                            "evidence_aliases": fragment["citation_aliases"],
                        }
                        for fragment in unit["source_fragments"]
                    ],
                    "exclusion_reason": "",
                }
                for i, unit in enumerate(payload["source_units"])
            ]
            output = {"units": units}
        else:
            payload["support_passages"] = {
                pid: {
                    "evidence_alias": value[0],
                    "source": (
                        "evidence_text" if value[1] == value[0] else "original_support"
                    ),
                    "material_ref": value[1],
                    "source_start": value[2],
                    "source_end": value[3],
                }
                for pid, value in payload["support_passages"].items()
            }
            for claim in payload["assertions"]:
                aliases = {e["evidence_alias"] for e in claim["evidence"]}
                claim["allowed_support_ids"] = [
                    pid
                    for pid, passage in payload["support_passages"].items()
                    if passage["evidence_alias"] in aliases
                ]
            output = {
                "units": [
                    {
                        "source_unit_id": u["source_unit_id"],
                        "complete": True,
                        "explanation": "All assertions extracted.",
                    }
                    for u in payload["source_units"]
                ],
                "decisions": self.decisions
                or [
                    {
                        "claim_id": c["claim_id"],
                        "correct_kind": c["kind"],
                        "atomic": True,
                        "support_basis": [
                            {"support_id": pid}
                            for pid in c["allowed_support_ids"]
                            if payload["support_passages"][pid]["source_start"] == 0
                            and payload["support_passages"][pid]["source_end"]
                            == len(
                                payload["original_supports"][
                                    payload["support_passages"][pid]["material_ref"]
                                ]["text"]
                                if payload["support_passages"][pid]["source"]
                                == "original_support"
                                else payload["cited_materials"][
                                    payload["support_passages"][pid]["material_ref"]
                                ]["text"]
                            )
                            and (
                                payload["support_passages"][pid]["source"]
                                == "original_support"
                                or not payload["cited_materials"][
                                    payload["support_passages"][pid]["evidence_alias"]
                                ].get("original_support_ref")
                            )
                        ],
                        "aspect_checks": [
                            {
                                "aspect": aspect,
                                "supported": True,
                                "support_ids": [],
                                "explanation": "The controlled source entails this aspect.",
                            }
                            for aspect in c["required_support_aspects"]
                        ],
                        "classification_valid": True,
                        "source_faithful": True,
                        "status": "supported",
                        "reason_code": "supported",
                        "explanation": "The actual cited material entails the assertion.",
                    }
                    for c in payload["assertions"]
                ],
                "conflicts": [],
            }
        if len(self.calls) == 2:
            for decision in output["decisions"]:
                for check in decision.get("aspect_checks", []):
                    check["support_ids"] = [
                        x["support_id"]
                        for x in decision.get("support_basis", [])
                        if "support_id" in x
                    ]
        if self.mutate:
            self.mutate(len(self.calls), output, payload)
        if self._cost_budget:
            self._cost_budget.record({"prompt_tokens": 100, "completion_tokens": 100})
        return LLMResponse(
            content=json.dumps(output),
            model="offline",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 100,
                "usage_available": True,
            },
        )


def materials_for_claim(payload, claim):
    """Resolve only the host allowlist, never the other table entries."""
    return [
        payload["cited_materials"][entry["material_ref"]] for entry in claim["evidence"]
    ]


def fragment_for_claim(payload, claim):
    """Resolve a claimed host fragment only within the corresponding unit."""
    unit = next(
        u
        for u in payload["source_units"]
        if u["source_unit_id"] == claim["source_unit_id"]
    )
    return next(
        f
        for f in unit["source_fragments"]
        if f["source_fragment_id"] == claim["source_fragment_id"]
    )


def ctx(**extra):
    """Supply two different papers so unrelated materials cannot silently rescue a citation."""
    return {
        "evaluation_version": EVALUATION_VERSION,
        "query": "Compare EViT and Evo-ViT",
        "evidence": {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "paper_title": "EViT",
                "text": "EViT accelerates DeiT-S by 50% with 0.3% accuracy loss.",
            },
            "b": {
                "evidence_id": "b",
                "paper_id": "evovit",
                "paper_title": "Evo-ViT",
                "text": "Evo-ViT uses slow-fast token evolution.",
            },
        },
        "structure_checks": {"status": "not_applicable"},
        "method_checks": {"status": "not_applicable"},
        **extra,
    }


SURVEY = "# Review\n\n## Results\n\nEvo-ViT accelerates DeiT-S by 50%. [E:a]\n"


@pytest.mark.asyncio
async def test_actual_citations_and_attribution_error_block_even_above_threshold():
    def incorrect(stage, output, payload):
        if stage == 2:
            assert (
                materials_for_claim(payload, payload["assertions"][0])[0]["paper_id"]
                == "evit"
            )
            assert all(
                e["paper_id"] != "evovit"
                for c in payload["assertions"]
                for e in materials_for_claim(payload, c)
            )
            output["decisions"][0].update(
                status="unsupported",
                reason_code="attribution_mismatch",
                explanation="The result belongs to EViT, not Evo-ViT.",
            )

    llm = ScriptedLLM(mutate=incorrect)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        SURVEY, ctx()
    )
    assert result.score == 0 and not result.passed
    diagnostic = result.details["unsupported_claims"][0]
    assert (
        SURVEY[diagnostic["source_start"] : diagnostic["source_end"]]
        == diagnostic["source_quote"]
    )
    assert diagnostic["evidence_ids"] == ["a"]
    assert diagnostic["section"] == "Results"
    assert len(llm.calls) == 2
    assert 'status="supported"' in llm.calls[1][0][0]["content"]
    assert "supported=true" not in llm.calls[1][0][0]["content"]
    assert all(call[1]["transport_max_attempts"] == 1 for call in llm.calls)
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": True, "details": result.details},
    }
    quality = LitAgent._derive_quality(evaluation)
    assert quality["status"] == "failed"
    assert "critical_attribution_errors" in quality["failed_metrics"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,kind",
    [
        ("This review focuses on ViT, CNN and data efficiency.", "report_structure"),
        ("本综述聚焦ViT、CNN及数据效率。", "report_structure"),
        (
            "Content includes data efficiency improvements and open challenges.",
            "report_structure",
        ),
        ("内容涵盖数据效率改进与开放挑战。", "report_structure"),
        (
            "This review compares original ViT, DeiT and hierarchical variants.",
            "report_structure",
        ),
        ("本综述比较原始ViT、DeiT和分层变体。", "report_structure"),
        (
            "For this review we group selected methods into three categories.",
            "report_structure",
        ),
        ("本综述将选定方法划为三类。", "report_structure"),
        ("This review finds that ViT outperforms CNN on large datasets.", "scientific"),
        ("本综述发现ViT在大规模数据集上优于CNN。", "scientific"),
        ("ViT requires more pretraining data than CNN.", "scientific"),
        ("ViT比CNN需要更多预训练数据。", "scientific"),
        ("How can ViT's poor small-data performance be improved?", "scientific"),
        ("如何改善ViT在小数据上的性能不足？", "scientific"),
    ],
)
async def test_classified_scope_does_not_exempt_uncited_science(text, kind):
    """Exercise curated kinds through both real request contracts, not an LLM proxy."""
    survey = (
        f"# Review\n\n## Abstract\n\n{text}\n\nEViT accelerates DeiT-S by 50%. [E:a]\n"
    )
    units = build_report_source_units(survey)
    inventory = [
        {
            "source_unit_id": u.unit_id,
            "assertions": [
                {
                    "kind": k,
                    "claim_text": t,
                    "source_fragment_id": build_source_fragments(u)[0][
                        "source_fragment_id"
                    ],
                    "evidence_aliases": [] if i == 0 else ["e1"],
                }
            ],
            "exclusion_reason": "",
        }
        for i, (u, k, t) in enumerate(
            zip(units, [kind, "scientific"], [text, "EViT accelerates DeiT-S by 50%."])
        )
    ]
    llm = ScriptedLLM(inventory=inventory)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["factual_claim_count"] == (1 if kind == "report_structure" else 2)
    assert result["score"] == (1 if kind == "report_structure" else 0.5)
    assert result["assertions"][0]["kind"] == kind
    assert result["assertions"][0]["source_quote"] == text
    payload = json.loads(llm.calls[1][0][1]["content"])
    assert payload["assertions"][0]["evidence"] == []
    assert result["classification_contract"] == CLASSIFICATION_CONTRACT
    for messages, kwargs in llm.calls:
        assert CLASSIFICATION_RULES in messages[0]["content"]
        assert (
            json.loads(messages[1]["content"])["classification_contract"]
            == CLASSIFICATION_CONTRACT
        )
        assert kwargs["transport_max_attempts"] == 1
    if kind == "scientific":
        assert result["unsupported_claims"][0]["reason_code"] == "uncited_claim"


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_mixed_organization_and_scientific_premise_remain_separate(language):
    text = (
        "This review compares ViT and CNN, finding that ViT is more accurate."
        if language == "en"
        else "本综述比较ViT与CNN，并发现ViT准确率更高。"
    )
    survey = f"# Review\n\n## Conclusion\n\n{text}\n"
    u = build_report_source_units(survey)[0]
    fragment = build_source_fragments(u)[0]
    llm = ScriptedLLM(
        inventory=[
            {
                "source_unit_id": u.unit_id,
                "assertions": [
                    {
                        "kind": k,
                        "claim_text": t,
                        "source_fragment_id": fragment["source_fragment_id"],
                        "evidence_aliases": [],
                    }
                    for k, t in [
                        (
                            "report_structure",
                            (
                                "This review compares ViT and CNN"
                                if language == "en"
                                else "本综述比较ViT与CNN"
                            ),
                        ),
                        (
                            "scientific",
                            (
                                "ViT is more accurate than CNN"
                                if language == "en"
                                else "ViT比CNN准确率更高"
                            ),
                        ),
                    ]
                ],
                "exclusion_reason": "",
            }
        ]
    )
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["factual_claim_count"] == 1 and result["score"] == 0
    assert result["unsupported_count"] == 1
    assert len(result["assertions"]) == 2
    assert all(c["source_quote"] == text for c in result["assertions"])


@pytest.mark.asyncio
async def test_wrong_scope_kind_is_rejected_instead_of_rescored():
    survey = "# Review\n\n## Abstract\n\n本综述比较ViT与CNN。\n"

    def reject(stage, output, payload):
        if stage == 2:
            assert (
                "Independently determine the correct kind"
                in llm.calls[-1][0][0]["content"]
            )
            output["decisions"][0].update(
                classification_valid=False,
                explanation="Scope is report_structure, not a paper finding.",
            )

    u = build_report_source_units(survey)[0]
    llm = ScriptedLLM(
        inventory=[
            {
                "source_unit_id": u.unit_id,
                "assertions": [
                    {
                        "kind": "scientific",
                        "claim_text": "本综述比较ViT与CNN",
                        "source_fragment_id": build_source_fragments(u)[0][
                            "source_fragment_id"
                        ],
                        "evidence_aliases": [],
                    }
                ],
                "exclusion_reason": "",
            }
        ],
        mutate=reject,
    )
    with pytest.raises(ValueError, match="classification_or_source"):
        await evaluate_scoped(
            llm=llm, survey=survey, context=ctx(), config=EvalConfig()
        )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_method_scientific_assertion_and_report_organization_are_split():
    survey = "# Review\n\n## Methods\n\nRetrieved 77 papers.\n\nEViT accelerates DeiT-S by 50%. [E:a]\n\nThis review groups methods into two sections.\n"
    units = build_report_source_units(survey)
    inventory = [
        {
            "source_unit_id": u.unit_id,
            "assertions": [
                {
                    "kind": kind,
                    "claim_text": text,
                    "source_fragment_id": build_source_fragments(u)[0][
                        "source_fragment_id"
                    ],
                    "evidence_aliases": ["e1"] if kind == "scientific" else [],
                }
            ],
            "exclusion_reason": "",
        }
        for u, kind, text in zip(
            units,
            ["runtime_method", "scientific", "report_structure"],
            [
                "Retrieved 77 papers.",
                "EViT accelerates DeiT-S by 50%.",
                "This review has two groups.",
            ],
        )
    ]
    llm = ScriptedLLM(inventory=inventory)
    result = await evaluate_scoped(
        llm=llm,
        survey=survey,
        context=ctx(review_methodology={"dedup": 77}),
        config=EvalConfig(),
    )
    assert result["score"] == 1
    assert result["factual_claim_count"] == 1
    payload = json.loads(llm.calls[1][0][1]["content"])
    assert payload["host_runtime_record"] == {"dedup": 77}
    assert payload["assertions"][0]["evidence"] == []
    assert payload["assertions"][1]["evidence"]
    assert result["coverage"]["processed_units"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation,reason,calls",
    [
        ("omit_unit", "unit_coverage", 1),
        ("duplicate_unit", "unit_coverage", 1),
        ("bad_fragment", "source_fragment", 1),
        ("cross_citation", "citation_ownership", 1),
        ("empty_exclusion", "exclusion_unexplained", 1),
        ("duplicate_claim", "duplicate_assertion", 1),
        ("omit_decision", "decision_coverage", 2),
        ("duplicate_decision", "decision_coverage", 2),
        ("bad_classification", "classification_or_source", 2),
        ("bad_faithfulness", "classification_or_source", 2),
        ("incomplete", "extraction_incomplete", 2),
        ("invalid_conflict", "conflict_identity", 2),
    ],
)
async def test_invalid_contract_never_retries_or_passes(mutation, reason, calls):
    def mutate(stage, output, payload):
        if stage == 1:
            if mutation == "omit_unit":
                output["units"] = []
            if mutation == "duplicate_unit":
                output["units"] *= 2
            if mutation == "bad_fragment":
                output["units"][0]["assertions"][0]["source_fragment_id"] = "invented"
            if mutation == "cross_citation":
                output["units"][0]["assertions"][0]["evidence_aliases"] = ["e99"]
            if mutation == "empty_exclusion":
                output["units"][0]["assertions"] = []
            if mutation == "duplicate_claim":
                output["units"][0]["assertions"] *= 2
        else:
            if mutation == "omit_decision":
                output["decisions"] = []
            if mutation == "duplicate_decision":
                output["decisions"] *= 2
            if mutation == "bad_classification":
                output["decisions"][0]["classification_valid"] = False
            if mutation == "bad_faithfulness":
                output["decisions"][0]["source_faithful"] = False
            if mutation == "incomplete":
                output["units"][0]["complete"] = False
            if mutation == "invalid_conflict":
                output["conflicts"] = [
                    {"claim_a": "invented", "claim_b": "claim_0", "reason": "invalid"}
                ]

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        SURVEY, ctx()
    )
    assert result.skipped
    assert reason in result.details["skipped_reason"]
    assert len(llm.calls) == calls
    assert (
        LitAgent._derive_quality(
            {
                "citation_accuracy": {"passed": True},
                "faithfulness": {"skipped": True, "details": result.details},
            }
        )["status"]
        == "unverified"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "prefixed",
        "bracketed",
        "omit_combined_alias",
        "borrow_alias",
    ],
)
async def test_real_extraction_failure_shapes_stop_before_second_call(mutation):
    """Reject real transport mistakes without normalizing or expanding the excerpt."""
    survey = "# Review\n\nFinding A [E:a] [E:b]. Finding B [E:b].\n"

    def mutate(stage, output, payload):
        if stage != 1:
            pytest.fail("An invalid inventory must not reach scientific verification.")
        assertion = output["units"][0]["assertions"][0]
        output["units"][0]["assertions"] = [assertion]
        assertion.update(evidence_aliases=["e1", "e2"])
        if mutation == "prefixed":
            assertion["evidence_aliases"] = ["E:e1", "E:e2"]
        elif mutation == "bracketed":
            assertion["evidence_aliases"] = ["[E:e1]", "[E:e2]"]
        elif mutation == "omit_combined_alias":
            assertion["evidence_aliases"] = ["e1"]
        else:
            assertion.update(
                source_fragment_id=payload["source_units"][0]["source_fragments"][1][
                    "source_fragment_id"
                ],
                evidence_aliases=["e1"],
            )

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped
    assert result.details["skipped_reason"] == "evaluation_citation_ownership_invalid"
    assert len(llm.calls) == 1
    assert (
        LitAgent._derive_quality(
            {"faithfulness": {"skipped": True, "details": result.details}}
        )["status"]
        == "unverified"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [" [E:a] [E:b].", ". [E:a] [E:b]"])
async def test_inventory_contract_exposes_allowed_aliases_and_preserves_attached_quotes(
    suffix,
):
    """Use actual host IDs in the request schema and restore exact quoted spans."""
    survey = "# Review\n\nFinding A" + suffix + "\n"

    def mutate(stage, output, payload):
        if stage == 1:
            assert payload["allowed_citation_aliases"] == ["e1", "e2"]
            unit = payload["source_units"][0]
            assert unit["citation_aliases"] == ["e1", "e2"]
            item_schema = payload["schema"]["$defs"]["Assertion"]["properties"][
                "evidence_aliases"
            ]["items"]
            assert item_schema["enum"] == ["e1", "e2"]
            assert item_schema["pattern"] == r"^e[1-9][0-9]*$"
            output["units"][0]["assertions"][0].update(
                claim_text="Finding A",
                source_fragment_id=unit["source_fragments"][0]["source_fragment_id"],
                evidence_aliases=unit["citation_aliases"],
            )

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    prompt = llm.calls[0][0][0]["content"]
    assert 'never "E:e1" or "[E:e1]"' in prompt
    example = json.loads(
        prompt.split("Example for source unit", 1)[1]
        .split("\n", 1)[1]
        .split("\nThis is a format example", 1)[0]
    )
    from litagent.eval.scoped import Inventory

    Inventory.model_validate(example)
    assert len(llm.calls) == 2 and result["coverage"]["complete"]
    assertion = result["assertions"][0]
    assert assertion["evidence_ids"] == ["a", "b"]
    assert (
        survey[assertion["source_start"] : assertion["source_end"]]
        == assertion["source_quote"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference,expected", [("missing", "unknown_evidence_ref"), (None, "uncited_claim")]
)
async def test_missing_or_uncited_assertion_remains_in_denominator(reference, expected):
    survey = (
        f"# Review\n\nA scientific finding.{f' [E:{reference}]' if reference else ''}\n"
    )

    def mutate(stage, output, payload):
        if stage == 1 and reference is None:
            output["units"][0]["assertions"][0]["evidence_aliases"] = []

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["score"] == 0 and result["factual_claim_count"] == 1
    assert result["unsupported_claims"][0]["reason_code"] == expected


@pytest.mark.asyncio
async def test_sixty_items_are_complete_and_material_overflow_calls_zero():
    survey = "# Review\n\n" + "\n\n".join(f"Finding {i}. [E:id{i}]" for i in range(60))
    ledger = {
        f"id{i}": {"evidence_id": f"id{i}", "text": f"Original finding {i}."}
        for i in range(60)
    }
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(evidence=ledger), config=EvalConfig()
    )
    assert result["contexts_count"] == 60 and result["factual_claim_count"] == 60
    assert result["coverage"]["complete"] and len(llm.calls) == 2
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="material_budget"):
        await evaluate_scoped(
            llm=llm,
            survey=survey,
            context=ctx(evidence=ledger),
            config=EvalConfig(faithfulness_max_evidence_items=40),
        )
    assert llm.calls == []


@pytest.mark.asyncio
async def test_model_and_run_budget_checked_before_requests_and_released():
    budget = CostBudget(max_tokens=10)
    llm = ScriptedLLM(budget=budget)
    with pytest.raises(ValueError, match="run_budget"):
        await evaluate_scoped(
            llm=llm, survey=SURVEY, context=ctx(), config=EvalConfig()
        )
    assert llm.calls == [] and budget.remaining() == 10
    budget = CostBudget(max_tokens=100000)
    llm = ScriptedLLM(budget=budget)
    await evaluate_scoped(llm=llm, survey=SURVEY, context=ctx(), config=EvalConfig())
    assert budget.used == 400 and budget.remaining() == 99600
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="context_budget"):
        await evaluate_scoped(
            llm=llm,
            survey=SURVEY,
            context=ctx(),
            config=EvalConfig(max_tokens=32768, context_tokens=32768),
        )
    assert llm.calls == []


@pytest.mark.asyncio
async def test_cancellation_propagates_without_second_call():
    class Cancelled(ScriptedLLM):
        async def chat(self, *args, **kwargs):
            self.calls.append(args)
            raise asyncio.CancelledError

    llm = Cancelled(budget=CostBudget())
    with pytest.raises(asyncio.CancelledError):
        await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(SURVEY, ctx())
    assert len(llm.calls) == 1 and llm._cost_budget.remaining() == 500000


@pytest.mark.asyncio
async def test_runner_shares_two_calls_with_consistency_instead_of_third_call():
    config = load_config()
    llm = ScriptedLLM()
    agent = LitAgent(config)
    agent._evaluators = [
        CitationEvaluator(llm),
        ConsistencyEvaluator(llm),
        RagasFaithfulnessEvaluator(config, llm=llm),
    ]
    results = {
        "extract": [{"claims": [], "evidence_items": list(ctx()["evidence"].values())}],
        "adversarial_review": {"review_structure": "legacy"},
    }
    evaluation = await agent._evaluate(
        query="ViT",
        survey=SURVEY.replace("Evo-ViT", "EViT"),
        results=results,
        phase="initial",
    )
    assert len(llm.calls) == 2
    assert evaluation["internal_consistency"]["score"] == 1
    assert (
        evaluation["faithfulness"]["details"]["evaluation_version"]
        == EVALUATION_VERSION
    )


@pytest.mark.parametrize(
    "check,status",
    [
        ("structure_checks", "failed"),
        ("method_checks", "failed"),
        ("method_checks", "unverified"),
    ],
)
def test_local_checks_participate_in_gate_independently_of_audits(check, status):
    details = {
        "evaluation_version": EVALUATION_VERSION,
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "passed"},
        check: {"status": status},
        "judge": {"status": "completed"},
    }
    quality = LitAgent._derive_quality(
        {
            "citation_accuracy": {"passed": True},
            "faithfulness": {"passed": True, "details": details},
        }
    )
    assert quality["status"] == ("failed" if status == "failed" else "unverified")


@pytest.mark.asyncio
async def test_v2_empty_ledger_keeps_citation_check_local_and_two_call_cap():
    """An empty ledger must not activate the legacy paper-title model fallback."""
    llm = ScriptedLLM()
    config = load_config()
    agent = LitAgent(config)
    agent._evaluators = [
        CitationEvaluator(llm),
        ConsistencyEvaluator(llm),
        RagasFaithfulnessEvaluator(config, llm=llm),
    ]
    evaluation = await agent._evaluate(
        query="ViT", survey=SURVEY, results={}, phase="initial"
    )
    assert len(llm.calls) == 2
    assert evaluation["citation_accuracy"]["score"] == 0
    assert evaluation["citation_accuracy"]["details"]["threshold"] == 0.8
    assert evaluation["faithfulness"]["score"] == 0


@pytest.mark.asyncio
async def test_unverifiable_decision_preserves_counts_locations_without_passing():
    """Incomplete evidence stays visible rather than leaving a smaller passing denominator."""

    def mutate(stage, output, payload):
        if stage == 2:
            output["decisions"][0].update(
                status="unverifiable",
                reason_code="unverifiable",
                explanation="Source does not provide the necessary condition.",
            )

    result = await RagasFaithfulnessEvaluator(
        load_config(), llm=ScriptedLLM(mutate=mutate)
    ).evaluate(SURVEY, ctx())
    assert result.skipped
    assert result.details["skipped_reason"] == "evaluation_assertions_unverifiable"
    assert result.details["unverified_count"] == 1
    assert result.details["unsupported_count"] == 0
    assert not result.details["coverage"]["complete"]
    assert result.details["unsupported_claims"][0]["source_quote"] in SURVEY


@pytest.mark.asyncio
async def test_table_combined_citations_and_repeated_paper_are_scoped_per_claim():
    """Table units retain their own combined references and exact source positions."""
    survey = "# Review\n\n## Comparison\n\n| Method | Finding |\n| --- | --- |\n| EViT and Evo-ViT | Different token mechanisms [E:a] [E:b] |\n\nEViT accelerates DeiT-S by 50%. [E:a]\n"
    units = build_report_source_units(survey)

    def mutate(stage, output, payload):
        if stage == 1:
            output["units"] = [
                {
                    "source_unit_id": unit["source_unit_id"],
                    "assertions": (
                        [
                            {
                                "kind": "scientific",
                                "claim_text": "Quoted finding",
                                "source_fragment_id": unit["source_fragments"][0][
                                    "source_fragment_id"
                                ],
                                "evidence_aliases": [
                                    alias
                                    for alias in ["e1", "e2"]
                                    if alias in unit["citation_aliases"]
                                ],
                            }
                        ]
                        if unit["citation_aliases"]
                        else []
                    ),
                    "exclusion_reason": (
                        "" if unit["citation_aliases"] else "Table header only."
                    ),
                }
                for unit in payload["source_units"]
            ]

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["coverage"]["processed_units"] == len(units)
    assert any(c["evidence_ids"] == ["a", "b"] for c in result["assertions"])
    assert all(
        survey[c["source_start"] : c["source_end"]] == c["source_quote"]
        for c in result["assertions"]
    )


@pytest.mark.asyncio
async def test_forged_method_is_separate_failed_gate_and_no_science_is_unverified():
    """Correct classification does not authorize fabricated procedures or empty reports."""
    survey = "# Review\n\n## Methods\n\nTwo reviewers screened all full texts.\n"
    unit = build_report_source_units(survey)[0]
    inventory = [
        {
            "source_unit_id": unit.unit_id,
            "assertions": [
                {
                    "kind": "runtime_method",
                    "claim_text": unit.text,
                    "source_fragment_id": build_source_fragments(unit)[0][
                        "source_fragment_id"
                    ],
                    "evidence_aliases": [],
                }
            ],
            "exclusion_reason": "",
        }
    ]

    def mutate(stage, output, payload):
        if stage == 2:
            output["decisions"][0].update(
                status="unsupported",
                reason_code="record_mismatch",
                explanation="The trusted record says not performed.",
            )

    result = await RagasFaithfulnessEvaluator(
        load_config(), llm=ScriptedLLM(inventory=inventory, mutate=mutate)
    ).evaluate(
        survey,
        ctx(review_methodology={"independent_duplicate_screening": "not performed"}),
    )
    assert result.skipped and result.details["method_assertion_errors"] == ["claim_0"]
    assert result.details["factual_claim_count"] == 0
    assert result.details["skipped_reason"] == "evaluation_no_scientific_assertions"


def test_source_ordinals_survive_styles_and_distinguish_repeated_text():
    """Map exact source units to rendered blocks independently of citation lengths."""
    import re

    from litagent.report_render import build_reader_report, reader_report_projection
    from litagent.workbench import ReportVersion

    survey = "# Review\n\nSame result [E:a].\n\nSame result [E:a].\n\n| Method | Result |\n| --- | --- |\n| EViT | Result [E:a] |\n"
    units = build_report_source_units(survey)
    report = ReportVersion(
        run_id="test", revision=1, report_language="en", survey=survey
    )
    for style in ["evidence", "numbered", "apa7", "ieee"]:
        reader = build_reader_report(
            report,
            evidence_ledger=ctx()["evidence"],
            paper_catalog={
                "evit": {"title": "EViT", "authors": ["Test Author"], "year": 2022}
            },
            citation_style=style,
        )
        html = reader_report_projection(reader)["html"]
        ordinals = re.findall(r'data-source-unit-ordinal="(\d+)"', html)
        assert ordinals[: len(units)] == [str(i) for i in range(len(units))]
        assert (
            'data-source-unit-ordinal="0"' in html
            and 'data-source-unit-ordinal="1"' in html
        )


def test_managed_delimiters_hidden_without_changing_original_positions():
    """Host ownership markers are not report content or extracted assertions."""
    from litagent.report_render import render_report_html
    from litagent.review_structure import assemble_methodology

    survey = assemble_methodology(
        "# Review\n\n## Search and Evidence Selection Methods\n\nFinding [E:a].",
        "method_transparent_v1",
        "en",
        {},
    )
    rendered = render_report_html(survey).html
    assert "litagent:run-methodology" not in rendered
    units = build_report_source_units(survey)
    assert all("litagent:run-methodology" not in u.text for u in units)
    assert all(survey[u.start : u.end] == u.text for u in units)


def test_extraction_prompt_change_invalidates_checkpoint_identity(monkeypatch):
    """An unchanged evaluation label must not allow recovery under a new prompt."""
    from pathlib import Path

    from litagent.interaction import interaction_source_fingerprint
    from litagent.workbench import ResumeIdentity, RunOptions

    original = Path.read_bytes
    before = interaction_source_fingerprint()
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: original(path)
        + (b"\nnew extraction protocol" if path.name == "scoped.py" else b""),
    )
    after = interaction_source_fingerprint()
    assert after != before
    old = ResumeIdentity(
        run_id="test",
        source_fingerprint=before,
        config_fingerprint="a" * 64,
        options=RunOptions(),
    )
    with pytest.raises(ValueError, match="incompatible_checkpoint"):
        old.require_compatible(old.model_copy(update={"source_fingerprint": after}))


@pytest.mark.asyncio
async def test_managed_disclosure_export_content_has_no_internal_markers():
    """PDF and Word keep visible disclosures and hide host delimiters."""
    import io

    import fitz
    from docx import Document

    from litagent.report_export import LocalReportExporter
    from litagent.review_structure import assemble_methodology
    from litagent.workbench import ReportVersion

    survey = assemble_methodology(
        "# Review\n\n## Search and Evidence Selection Methods\n",
        "method_transparent_v1",
        "en",
        {},
    )
    report = ReportVersion(
        run_id="test", revision=1, report_language="en", survey=survey
    )
    exporter = LocalReportExporter()
    pdf = await exporter.export(report, format="pdf", publishable=False)
    docx = await exporter.export(report, format="docx", publishable=False)
    with fitz.open(stream=pdf.content, filetype="pdf") as document:
        pdf_text = "\n".join(page.get_text() for page in document)
    docx_text = "\n".join(p.text for p in Document(io.BytesIO(docx.content)).paragraphs)
    for text in (pdf_text, docx_text):
        assert "litagent:run-methodology" not in text
        assert "Search start time" in text and "not recorded" in text
    assert pdf.report_sha256 == docx.report_sha256 == report.content_sha256


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First sentence. Intermediate method sentence. [E:a] [E:b]", [[], ["a", "b"]]),
        ("第一句。中间方法句。[E:a] [E:b]", [[], ["a", "b"]]),
        ("A and B share one result [E:a] [E:b].", [["a", "b"]]),
        ("A [E:a]. B [E:b].", [["a"], ["b"]]),
        ("Measured 3.14 percent. [E:a]", [["a"]]),
        ("A [E:doi.1]. B [E:b].", [["doi.1"], ["b"]]),
        ("Same [E:a]. Same [E:a].", [["a"], ["a"]]),
    ],
)
def test_host_fragments_are_contiguous_and_never_inherit_later_citations(
    text, expected
):
    unit = build_report_source_units("# Review\n\n" + text + "\n")[0]
    fragments = build_source_fragments(unit)
    assert [f["citation_aliases"] for f in fragments] == expected
    assert len({f["source_fragment_id"] for f in fragments}) == len(fragments)
    assert all(unit.text[f["start"] : f["end"]] == f["source_quote"] for f in fragments)
    assert "".join(
        f["source_quote"].replace(" ", "") for f in fragments
    ) == text.replace(" ", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "cross_unit",
        "missing",
        "extra_quote",
        "extra_offsets",
        "stitched_quote",
    ],
)
async def test_fragment_contract_rejects_invented_locations_without_retry(mutation):
    survey = "# Review\n\nFirst sentence. Intermediate sentence. [E:a] [E:b]\n\nOther finding [E:b].\n"

    def mutate(stage, output, payload):
        assert stage == 1
        assertion = output["units"][0]["assertions"][0]
        if mutation == "unknown":
            assertion["source_fragment_id"] = "not-provided"
        elif mutation == "cross_unit":
            assertion["source_fragment_id"] = payload["source_units"][1][
                "source_fragments"
            ][0]["source_fragment_id"]
        elif mutation == "missing":
            del assertion["source_fragment_id"]
        elif mutation == "extra_offsets":
            assertion["source_start"] = 0
        else:
            assertion["source_quote"] = (
                "First sentence. [E:e1] [E:e2]"
                if mutation == "stitched_quote"
                else "First sentence."
            )

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped and len(llm.calls) == 1
    assert result.details["extraction_contract"] == EXTRACTION_CONTRACT
    assert (
        LitAgent._derive_quality(
            {"faithfulness": {"skipped": True, "details": result.details}}
        )["status"]
        == "unverified"
    )


@pytest.mark.asyncio
async def test_abstract_fragment_selects_host_text_without_borrowing_sixty_end_refs():
    refs = " ".join(f"[E:id{i}]" for i in range(60))
    survey = (
        "# Review\n\nThis review describes ViT methods. Retrieved sixty paper evidence items. "
        + refs
        + "\n"
    )

    def mutate(stage, output, payload):
        if stage == 1:
            assertions = output["units"][0]["assertions"]
            assertions[0].update(
                kind="report_structure", claim_text="This review describes ViT methods."
            )
            assertions[1].update(
                kind="runtime_method",
                claim_text="Retrieved sixty paper evidence items.",
            )
        else:
            assert (
                fragment_for_claim(payload, payload["assertions"][0])[
                    "citation_aliases"
                ]
                == []
            )
            assert (
                len(
                    fragment_for_claim(payload, payload["assertions"][1])[
                        "citation_aliases"
                    ]
                )
                == 60
            )
            assert payload["assertions"][1]["evidence"] == []
            assert payload["cited_materials"] == payload["original_supports"] == {}
            assert (
                fragment_for_claim(payload, payload["assertions"][0])["text"]
                == "This review describes ViT methods."
            )

    ledger = {
        f"id{i}": {"text": f"Finding {i}", "paper_id": f"paper{i}"} for i in range(60)
    }
    result = await evaluate_scoped(
        llm=ScriptedLLM(mutate=mutate),
        survey=survey,
        context=ctx(
            evidence=ledger, review_methodology={"selected_evidence_count": 60}
        ),
        config=EvalConfig(),
    )
    assert (
        result["score"] is None
    )  # Honest organization/method statements are not science.
    assert result["extraction_contract"] == EXTRACTION_CONTRACT
    assert result["coverage"]["complete"] and result["factual_claim_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["classification_valid", "source_faithful"])
async def test_host_location_does_not_bypass_semantic_or_classification_gate(field):
    def mutate(stage, output, payload):
        if stage == 2:
            output["decisions"][0][field] = False

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        SURVEY, ctx()
    )
    assert (
        result.skipped
        and "classification_or_source" in result.details["skipped_reason"]
    )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_uncited_first_sentence_cannot_be_rescued_by_cited_second_sentence():
    survey = "# Review\n\nEViT accelerates DeiT-S by 50%. Evo-ViT uses token evolution [E:b].\n"
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["factual_claim_count"] == 2 and result["score"] == 0.5
    assert result["assertions"][0]["reason_code"] == "uncited_claim"
    payload = json.loads(llm.calls[1][0][1]["content"])
    assert payload["assertions"][0]["evidence"] == []
    assert (
        materials_for_claim(payload, payload["assertions"][1])[0]["paper_id"]
        == "evovit"
    )


@pytest.mark.asyncio
async def test_repeated_sentence_fragments_have_distinct_exact_offsets():
    survey = "# Review\n\nSame result [E:a]. Same result [E:a].\n"
    result = await evaluate_scoped(
        llm=ScriptedLLM(), survey=survey, context=ctx(), config=EvalConfig()
    )
    claims = result["assertions"]
    assert len(claims) == 2 and claims[0]["source_quote"] == claims[1]["source_quote"]
    assert claims[0]["source_start"] != claims[1]["source_start"]
    assert all(
        survey[c["source_start"] : c["source_end"]] == c["source_quote"] for c in claims
    )


@pytest.mark.asyncio
async def test_alias_mapping_does_not_cascade_when_original_ids_are_short_aliases():
    survey = "# Review\n\nA [E:e2]. B [E:e1].\n"
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm,
        survey=survey,
        context=ctx(evidence={"e2": {"text": "A"}, "e1": {"text": "B"}}),
        config=EvalConfig(),
    )
    request = json.loads(llm.calls[0][0][1]["content"])["source_units"][0]
    assert [f["text"] for f in request["source_fragments"]] == [
        "A [E:e1].",
        "B [E:e2].",
    ]
    for fragment in request["source_fragments"]:
        quote = fragment["text"]
        assert fragment["citation_aliases"][0] in quote
    assert [c["evidence_ids"] for c in result["assertions"]] == [["e2"], ["e1"]]


@pytest.mark.asyncio
async def test_malformed_reference_does_not_crash_host_alias_mapping():
    result = await evaluate_scoped(
        llm=ScriptedLLM(),
        survey="# Review\n\nFinding [E:a;b].\n",
        context=ctx(),
        config=EvalConfig(),
    )
    assert (
        result["score"] == 0
        and result["assertions"][0]["local_error"] == "uncited_claim"
    )


@pytest.mark.asyncio
async def test_atomic_compound_claims_share_fragment_without_repeating_positions():
    survey = "# Review\n\nEViT is faster and loses 0.3% accuracy [E:a].\n"

    def mutate(stage, output, payload):
        if stage == 1:
            first = output["units"][0]["assertions"][0]
            first["claim_text"] = "EViT is faster."
            output["units"][0]["assertions"].append(
                {**first, "claim_text": "EViT loses 0.3% accuracy."}
            )

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert len(result["assertions"]) == 2 and len(llm.calls) == 2
    a, b = result["assertions"]
    assert a["source_fragment_id"] == b["source_fragment_id"]
    assert a["source_start"] == b["source_start"] and a["evidence_ids"] == b[
        "evidence_ids"
    ] == ["a"]


@pytest.mark.asyncio
async def test_separate_clause_citations_use_distinct_host_fragments_and_materials():
    """Resolve the live mismatch without relaxing the complete citation set gate."""
    survey = "# 综述\n\nEViT加速推理[E:a]；Evo-ViT采用慢快令牌演化[E:b]。\n"

    def mutate(stage, output, payload):
        if stage == 1:
            fragments = payload["source_units"][0]["source_fragments"]
            assert [f["citation_aliases"] for f in fragments] == [["e1"], ["e2"]]
            for assertion, claim in zip(
                output["units"][0]["assertions"],
                ["EViT加速推理", "Evo-ViT采用慢快令牌演化"],
                strict=True,
            ):
                assertion["claim_text"] = claim
        else:
            assert [
                [item["paper_id"] for item in materials_for_claim(payload, claim)]
                for claim in payload["assertions"]
            ] == [["evit"], ["evovit"]]

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["coverage"]["complete"] and len(llm.calls) == 2
    assert result["extraction_contract"] == "host-source-fragments-v2"
    for assertion in result["assertions"]:
        assert (
            survey[assertion["source_start"] : assertion["source_end"]]
            == assertion["source_quote"]
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A [E:a]; B [E:b].", [["a"], ["b"]]),
        ("A[E:a]；B[E:b]。", [["a"], ["b"]]),
        ("A [E:a], B [E:b].", [["a"], ["b"]]),
        ("A[E:a]，B[E:b]。", [["a"], ["b"]]),
        ("A[E:a]、B[E:b]。", [["a"], ["b"]]),
        ("1）A[E:a]；2）B[E:b]；3）C[E:c]。", [["a"], ["b"], ["c"]]),
        ("A [E:a] [E:b]; C [E:c].", [["a", "b"], ["c"]]),
        ("A [E:a], B [E:b], only on ImageNet.", [["a"], ["b"]]),
        ("A [E:a]; only under the stated condition.", [["a"]]),
        ("A [E:a], only under the stated condition.", [["a"]]),
        ("A[E:a]；仅在指定条件下成立。", [["a"]]),
        ("Uncited A; B [E:b].", [[], ["b"]]),
        ("未引用A；B[E:b]。", [[], ["b"]]),
        ("Uncited A; uncited B; C [E:c].", [[], [], ["c"]]),
        ("A, B and C share an experiment [E:a] [E:b].", [["a", "b"]]),
        ("A、B及C使用同一实验[E:a] [E:b]。", [["a", "b"]]),
        ("At 3.14 percent, A [E:a]; B [E:b].", [["a"], ["b"]]),
        ("Using X, Y and Z, A [E:a], B [E:b].", [["a"], ["b"]]),
        ("A [E:doi:10.1234/a;b]; B [E:b].", [[], ["b"]]),
        ('"A [E:a]; B [E:b]" is quoted.', [["a", "b"]]),
        ("“A[E:a]；B[E:b]”是引用原话。", [["a", "b"]]),
        ("A (X [E:a]; Y [E:b]).", [["a", "b"]]),
        ("A（X[E:a]；Y[E:b]）。", [["a", "b"]]),
        ("A [E:a] (only with X, Y); B [E:b].", [["a"], ["b"]]),
        ("`A [E:a]; B [E:b]` is code.", [["a", "b"]]),
        ("``A `x` [E:a]; B [E:b]`` is code.", [["a", "b"]]),
        ("A [E:a]\\; B [E:b].", [["a", "b"]]),
        ("A [E:a]\\, B [E:b].", [["a", "b"]]),
        ("A (unclosed [E:a]; B [E:b].", [["a", "b"]]),
        ("A `unclosed [E:a]; B [E:b].", [["a", "b"]]),
        ("See https://example.org/a;b,c [E:a]; B [E:b].", [["a"], ["b"]]),
        ("A [E:a]. Intermediate sentence. B [E:b].", [["a"], [], ["b"]]),
    ],
)
def test_clause_boundaries_preserve_exact_text_and_conservative_scopes(text, expected):
    """Citation clusters and protected punctuation cannot create borrowed evidence."""
    unit = SimpleNamespace(unit_id="unit", text=text)
    fragments = build_source_fragments(unit)
    assert [f["citation_aliases"] for f in fragments] == expected
    assert all(text[f["start"] : f["end"]] == f["source_quote"] for f in fragments)
    assert "".join(f["source_quote"] for f in fragments).replace(
        " ", ""
    ) == text.replace(" ", "")
    assert len({f["source_fragment_id"] for f in fragments}) == len(fragments)


@pytest.mark.asyncio
async def test_eleven_live_clause_shapes_have_exact_scoped_citation_inventories():
    """Exercise the three compound-sentence shapes as self-contained offline input."""
    survey = (
        "# 综述\n\n"
        "核心发现：ViT在大规模预训练下性能优异[E:a] [E:b]，"
        "而DeiT提高数据效率[E:c]；分层ViT降低计算成本[E:d]；ConViT增加归纳偏置[E:e]。\n\n"
        "然而，ViT面临挑战：标注数据需求[E:f]、自注意力计算量[E:g]、局部归纳偏置[E:h]。\n\n"
        "1）计算量较大[E:g]；2）数据需求较高[E:f]；3）局部结构建模不足[E:h]；"
        "4）鲁棒性仍有挑战[E:i]。\n"
    )
    expected = ["ab", "c", "d", "e", "f", "g", "h", "g", "f", "h", "i"]
    ledger = {
        eid: {"evidence_id": eid, "paper_id": f"paper-{eid}", "text": f"Finding {eid}"}
        for eid in "abcdefghi"
    }

    def mutate(stage, output, payload):
        if stage == 1:
            for unit, entry in zip(
                payload["source_units"], output["units"], strict=True
            ):
                for f, a in zip(
                    unit["source_fragments"], entry["assertions"], strict=True
                ):
                    a["claim_text"] = f["text"]
        else:
            for decision in output["decisions"]:
                decision.update(
                    status="unsupported",
                    reason_code="insufficient_support",
                    explanation="Placeholder material has no scientific support.",
                )
            assert [
                [m["evidence_id"] for m in materials_for_claim(payload, c)]
                for c in payload["assertions"]
            ] == [list(ids) for ids in expected]

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(evidence=ledger), config=EvalConfig()
    )
    assert len(llm.calls) == 2 and result["factual_claim_count"] == 11
    assert result["coverage"]["complete"]
    assert [c["evidence_ids"] for c in result["assertions"]] == [
        list(e) for e in expected
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["borrow_neighbor", "omit_combined", "duplicate"])
async def test_clause_scopes_still_reject_incomplete_or_borrowed_citation_sets(
    mutation,
):
    survey = "# Review\n\nA [E:a] [E:b]; B [E:b].\n"

    def mutate(stage, output, payload):
        assert stage == 1
        assertions = output["units"][0]["assertions"]
        if mutation == "borrow_neighbor":
            assertions[1]["evidence_aliases"] = ["e1"]
        elif mutation == "omit_combined":
            assertions[0]["evidence_aliases"] = ["e1"]
        else:
            assertions[1]["evidence_aliases"] = ["e2", "e2"]

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped and len(llm.calls) == 1
    assert result.details["skipped_reason"] == "evaluation_citation_ownership_invalid"


@pytest.mark.asyncio
async def test_uncited_clause_cannot_inherit_terminal_citation():
    survey = "# Review\n\nEViT is faster; Evo-ViT uses evolution [E:b].\n"
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert result["score"] == 0.5 and result["factual_claim_count"] == 2
    assert result["assertions"][0]["local_error"] == "uncited_claim"
    payload = json.loads(llm.calls[1][0][1]["content"])
    assert payload["assertions"][0]["evidence"] == []


@pytest.mark.asyncio
async def test_table_cells_with_repeated_cited_clauses_keep_original_locations():
    survey = "# Review\n\n| Comparison |\n| --- |\n| A [E:a]; B [E:b]. |\n| A [E:a]; B [E:b]. |\n"

    def mutate(stage, output, payload):
        if stage == 1:
            for unit in output["units"]:
                if not unit["assertions"][0]["evidence_aliases"]:
                    unit["assertions"] = []
                    unit["exclusion_reason"] = "Column heading"

    llm = ScriptedLLM(mutate=mutate)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(), config=EvalConfig()
    )
    assert [c["evidence_ids"] for c in result["assertions"]] == [["a"], ["b"]] * 2
    assert len({c["source_start"] for c in result["assertions"]}) == 4
    for c in result["assertions"]:
        assert survey[c["source_start"] : c["source_end"]] == c["source_quote"]


@pytest.mark.asyncio
async def test_clause_attribution_error_blocks_even_with_high_average_support():
    survey = (
        "# Review\n\nEvo-ViT accelerates DeiT-S by 50% [E:a]; "
        + "; ".join("Evo-ViT uses slow-fast token evolution [E:b]" for _ in range(8))
        + ".\n"
    )

    def mutate(stage, output, payload):
        if stage == 1:
            for a, f in zip(
                output["units"][0]["assertions"],
                payload["source_units"][0]["source_fragments"],
                strict=True,
            ):
                a["claim_text"] = f["text"]
        else:
            first = payload["assertions"][0]
            assert [m["paper_id"] for m in materials_for_claim(payload, first)] == [
                "evit"
            ]
            output["decisions"][0].update(
                status="unsupported",
                reason_code="attribution_mismatch",
                explanation="The cited speed result belongs to EViT, not Evo-ViT.",
            )

    llm = ScriptedLLM(mutate=mutate)
    metric = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert not metric.skipped and metric.score == pytest.approx(8 / 9)
    assert metric.passed and len(metric.details["critical_attribution_errors"]) == 1
    quality = LitAgent._derive_quality(
        {"faithfulness": {"passed": metric.passed, "details": metric.details}}
    )
    assert quality["status"] == "failed"
    assert "critical_attribution_errors" in quality["failed_metrics"]
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_shared_material_table_keeps_wrong_attribution_blocked_above_threshold():
    """A paper available to another claim must not rescue the wrongly cited claim."""
    survey = SURVEY + "\n\n".join(
        "Evo-ViT uses slow-fast token evolution [E:b]." for _ in range(8)
    )

    def mutate(stage, output, payload):
        if stage == 1:
            for unit, source in zip(
                output["units"], payload["source_units"], strict=True
            ):
                unit["assertions"][0]["claim_text"] = source["source_fragments"][0][
                    "text"
                ]
        else:
            assert len(payload["cited_materials"]) == 2
            first = payload["assertions"][0]
            assert first["evidence"] == [{"evidence_alias": "e1", "material_ref": "e1"}]
            assert materials_for_claim(payload, first)[0]["paper_id"] == "evit"
            assert payload["cited_materials"]["e2"]["paper_id"] == "evovit"
            assert all(
                c["evidence"] == [{"evidence_alias": "e2", "material_ref": "e2"}]
                for c in payload["assertions"][1:]
            )
            output["decisions"][0].update(
                status="unsupported",
                reason_code="attribution_mismatch",
                explanation="The acceleration figure belongs to EViT, not Evo-ViT.",
            )

    result = await RagasFaithfulnessEvaluator(
        load_config(), llm=ScriptedLLM(mutate=mutate)
    ).evaluate(survey, ctx())
    assert result.score == pytest.approx(8 / 9) and result.passed
    assert (
        LitAgent._derive_quality(
            {
                "citation_accuracy": {"passed": True},
                "faithfulness": {"passed": result.passed, "details": result.details},
            }
        )["status"]
        == "failed"
    )


@pytest.mark.asyncio
async def test_repeated_scientific_material_is_transmitted_once_with_complete_allowlists():
    survey = "# Review\n\n" + "\n\n".join(f"Finding {i} [E:a]." for i in range(60))
    item = {
        "paper_id": "evit",
        "paper_title": "EViT",
        "text": "x" * 2000,
        "supporting_text": "y" * 4000,
    }
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=ctx(evidence={"a": item}), config=EvalConfig()
    )
    payload = json.loads(llm.calls[1][0][1]["content"])
    assert len(payload["cited_materials"]) == len(payload["original_supports"]) == 1
    assert len(payload["assertions"]) == result["factual_claim_count"] == 60
    assert all(
        c["evidence"] == [{"evidence_alias": "e1", "material_ref": "e1"}]
        for c in payload["assertions"]
    )
    assert payload["cited_materials"]["e1"]["text"] == item["text"]
    assert (
        next(iter(payload["original_supports"].values()))["text"]
        == item["supporting_text"]
    )
    assert len(llm.calls) == 2
