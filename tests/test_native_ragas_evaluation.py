"""Exercise native RAGAS, exact source ownership and independent delivery gates."""

import copy
import json

import pytest
from test_operational_limits import uplift_plan  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.body_quality import ROLES
from litagent.config import EvalConfig
from litagent.eval.ragas_global_wire import CONTRACT as GLOBAL_WIRE
from litagent.eval.ragas_global_wire import UNIT_CONTRACT, unpack_global_input
from litagent.eval.ragas_native import completed_repair, execute_native
from litagent.eval.results import acceptance
from litagent.llm.client import LLMResponse
from litagent.review_structure import structure_sections


def verdict_pairs(job, output):
    """Pair mock verdicts with the input statements for either wire format."""
    statements = job["statements"]
    verdicts = output["statements"]
    if isinstance(statements, dict):
        return ((text, verdicts[key]) for key, text in statements.items())
    return zip(statements, verdicts, strict=True)


class SemanticTransport:
    """Replace only provider responses while retaining all production consumers."""

    def __init__(self, *, reject=None, global_failure=None):
        """Select a negative verdict or independent invalid global region."""
        self.reject = reject
        self.global_failure = global_failure
        self.calls = []

    async def chat(self, messages, **kwargs):
        """Return typed RAGAS statements and a report review with actual host IDs."""
        payload = json.loads(messages[1]["content"])
        self.calls.append((payload, kwargs))
        if payload.get("global_input_contract") in {GLOBAL_WIRE, UNIT_CONTRACT}:
            payload = unpack_global_input(payload)
        if "jobs" in payload:
            if "context_pool" not in payload:
                raw = {
                    key: {"statements": [value["answer"]]}
                    for key, value in payload["jobs"].items()
                }
            else:
                raw = {}
                for key, job in payload["jobs"].items():
                    statements = job["statements"]
                    if isinstance(statements, dict):
                        decisions = {
                            cid: {
                                "reason": "source entailment fixture",
                                "verdict": int(s != self.reject),
                            }
                            for cid, s in statements.items()
                        }
                    else:
                        decisions = [
                            {
                                "statement": s,
                                "reason": "source entailment fixture",
                                "verdict": int(s != self.reject),
                            }
                            for s in statements
                        ]
                    raw[key] = {"statements": decisions}
        else:
            ids = [c["claim_id"] for c in payload["assertions"]]
            titles = dict(structure_sections("academic_v2", "en"))
            by_section = {
                role: [
                    c["claim_id"]
                    for c in payload["assertions"]
                    if c["section"] == titles[role]
                ]
                for role in ROLES
            }
            raw = {
                "units": [
                    {
                        "source_unit_id": u["source_unit_id"],
                        "complete": True,
                        "explanation": "complete",
                    }
                    for u in payload["source_units"]
                ],
                "conflicts": [],
                "issues": [],
                "coverage": [
                    {
                        "requirement_id": r["requirement_id"],
                        "answered": True,
                        "claim_ids": ids,
                        "explanation": "covered",
                    }
                    for r in payload["coverage_requirements"]
                ],
                "analyses": [
                    {
                        "section_key": role,
                        "adequate": True,
                        "claim_ids": by_section[role],
                        "explanation": "analysis",
                    }
                    for role in ROLES
                ],
                "limitations": {
                    "adequate": True,
                    "claim_ids": by_section["open_problems"],
                    "explanation": "specific limitation",
                },
            }
            if self.global_failure == "analysis":
                raw["analyses"] = "invalid"
            if self.global_failure == "omission":
                raw["units"][0]["complete"] = False
            if self.global_failure == "identity":
                raw["coverage"][0]["requirement_id"] = "invented"
            if payload.get("global_unit_response_contract"):
                raw["units"] = {
                    unit["coverage_key"]: {
                        "complete": decision["complete"],
                        "explanation": decision["explanation"],
                    }
                    for unit, decision in zip(
                        payload["source_units"], raw["units"], strict=True
                    )
                }
        return LLMResponse(
            json.dumps(raw),
            model="offline",
            finish_reason="stop",
            usage={
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )


@pytest.fixture
def native_case():
    """Keep the same evidence text at distinct original report positions."""
    titles = dict(structure_sections("academic_v2", "en"))
    sentences = {
        "abstract": "Method A requires labeled examples.",
        "introduction": "Few-shot tasks use 1 to 5 samples.",
        "taxonomy": "The benchmark includes 15 methods and measures accuracy.",
        "methods": "Method A learns an embedding.",
        "experiments": "Method A reaches 80 percent under setting X.",
        "open_problems": "Method A requires labeled examples.",
        "conclusion": "Method A reaches 80 percent under setting X.",
    }
    text = "# Survey\n\n" + "\n\n".join(
        "## " + title + "\n\n" + sentences[role] + " [E:e1]"
        for role, title in titles.items()
    )
    context = {
        "query": "Explain method A",
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "coverage_requirements": [
            {"requirement_id": "question", "description": "Explain method A"}
        ],
        "evidence": {
            "e1": {
                "evidence_id": "e1",
                "paper_id": "paper:A",
                "supporting_text": "Typically 1-5 samples. The benchmark includes 15 classic methods and reports accuracy. Method A learns an embedding, requires labeled examples and reaches 80 percent under setting X.",
            }
        },
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "passed"},
        "language_checks": {"status": "passed"},
    }
    from litagent.quality import build_fact_cards, digest

    context["evidence"]["e1"].update(
        text="Source-supported method A finding", source_locator="abstract"
    )
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    return text, context


@pytest.mark.asyncio
async def test_ragas_handles_ranges_and_method_counts_without_old_heuristics(
    native_case, monkeypatch
):
    """Official NLI, not a numeric token heuristic, owns source entailment."""
    import litagent.eval.scoped as scoped

    def forbidden(*args, **kwargs):
        raise AssertionError("custom support checker called")

    monkeypatch.setattr(scoped, "validate_assessment_decisions", forbidden)
    text, context = native_case
    llm = SemanticTransport()
    run = await execute_native(
        llm=llm, survey=text, context=context, config=EvalConfig()
    )
    metrics = run.metrics()
    assert metrics["faithfulness"]["score"] == 1.0, run.details
    assert (
        metrics["faithfulness"]["details"]["implementation"]
        == "ragas-cited-faithfulness-v1"
    )
    assert run.checks()["analysis"]["status"] == "passed"
    assert acceptance(run.details)["status"] == "passed"
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_unsupported_science_has_score_and_independent_consistency(native_case):
    """An unsupported scientific statement does not erase contradiction results."""
    text, context = native_case
    llm = SemanticTransport(reject="Method A learns an embedding.")
    run = await execute_native(
        llm=llm, survey=text, context=context, config=EvalConfig()
    )
    metrics = run.metrics()
    assert metrics["faithfulness"]["score"] == 6 / 7
    assert metrics["internal_consistency"]["score"] == 1
    assert acceptance(run.details)["status"] == "failed"
    assert completed_repair(run.details, text)
    assert not completed_repair(run.details, text + "changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["analysis", "omission", "identity"])
async def test_global_failures_cannot_become_delivery(native_case, fault):
    """Preserve independently valid science while refusing invalid global checks."""
    text, context = native_case
    run = await execute_native(
        llm=SemanticTransport(global_failure=fault),
        survey=text,
        context=context,
        config=EvalConfig(),
    )
    metrics = run.metrics()
    assert acceptance(run.details)["status"] == "failed"
    if fault == "analysis":
        assert metrics["faithfulness"]["score"] == 1
        assert run.checks()["analysis"]["status"] == "error"
    else:
        assert metrics["faithfulness"]["score"] is None


@pytest.mark.asyncio
async def test_unknown_citation_rejected_before_provider(native_case):
    """A different paper cannot silently replace a missing citation source."""
    text, context = native_case
    llm = SemanticTransport()
    run = await execute_native(
        llm=llm,
        survey=text.replace("[E:e1]", "[E:unknown]"),
        context=context,
        config=EvalConfig(),
    )
    assert run.metrics()["faithfulness"]["score"] is None
    assert not llm.calls


@pytest.mark.asyncio
async def test_partial_nli_counts_remain_visible_without_a_full_score(native_case):
    """Valid individual results survive a later malformed job without delivery."""

    class Partial(SemanticTransport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            if "context_pool" in json.loads(messages[1]["content"]):
                raw = json.loads(response.content)
                raw[list(raw)[4]]["statements"] = []
                response.content = json.dumps(raw)
            return response

    text, context = native_case
    llm = Partial()
    run = await execute_native(
        llm=llm, survey=text, context=context, config=EvalConfig()
    )
    assert run.metrics()["faithfulness"]["score"] is None
    assert len(run.assertions) == 7
    assert sum(a["status"] == "supported" for a in run.assertions) == 4
    assert sum(a["status"] == "unverifiable" for a in run.assertions) == 3
    assert len(llm.calls) == 2
    assert acceptance(run.details)["status"] != "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "identity"])
async def test_expanded_global_compacts_full_methodology_and_keeps_identity_guards(
    native_case, fault
):
    """Large global input fits without skipping source text or identity validation."""
    from litagent.config import load_config
    from litagent.operational_limits import PROFILE
    from litagent.safety.budget import CostBudget

    text, context = native_case
    locator_keys = ["abstract", "page:01:block:0", "特殊字段"] + [
        f"page:{i}:block:0" for i in range(1, 11001)
    ]
    context["review_methodology"] = {
        "fulltext_acquisition": {
            "paper:A": {"previous_chunk_keys": locator_keys, "status": "available"},
        }
    }
    cfg = load_config("config/operational-plus20.yaml").eval
    llm = SemanticTransport(global_failure=fault)
    llm._cost_budget = CostBudget(max_tokens=600000, resource_profile=PROFILE)
    run = await execute_native(llm=llm, survey=text, context=context, config=cfg)
    payload, opts = llm.calls[-1]
    assert payload["global_input_contract"] == GLOBAL_WIRE
    canonical = unpack_global_input(payload)
    assert canonical["review_methodology"] == context["review_methodology"]
    assert len(canonical["assertions"]) == 7
    assert all(a["evidence_ids"] == ["e1"] for a in canonical["assertions"])
    assert opts["transport_max_attempts"] == 1
    assert run.metrics()["faithfulness"]["score"] == (1.0 if fault is None else None)
    assert acceptance(run.details)["status"] == (
        "passed" if fault is None else "failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_stage", [None, "assessment", "global"])
async def test_native_inventory_admission_measures_complete_consumer_wires(
    native_case, uplift_plan, oversized_stage
):
    """Keep inventories over 36k and reject an oversize wire before NLI."""
    from litagent.config import load_config
    from litagent.safety.planned_budget import PlannedBudget, nested_json_size

    class LargeInventory(SemanticTransport):
        """Return more statement text than the legacy cumulative JSON gate."""

        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = self.calls[-1][0]
            if "jobs" in payload and "context_pool" not in payload:
                response.content = json.dumps(
                    {
                        key: {
                            "statements": [
                                f"{job['answer']} Fixture assertion {index} retains the original setting and experimental controls."
                                for index in range(7)
                            ]
                        }
                        for key, job in payload["jobs"].items()
                    }
                )
            return response

    text, context = native_case
    text += "".join(
        f"\n\nFixture report unit {index} provides one comparison condition. [E:e1]"
        for index in range(45)
    )
    if oversized_stage == "global":
        context["review_methodology"] = {"uncompressible_record": "x" * 220000}
    elif oversized_stage == "assessment":
        from litagent.quality import build_fact_cards, digest

        context["evidence"]["e1"]["supporting_text"] = "x" * 220000
        context["fact_cards"] = build_fact_cards(context["evidence"])
        context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    cfg = load_config("config/operational-plus20.yaml").eval
    llm = LargeInventory()
    llm._cost_budget = PlannedBudget(uplift_plan)
    run = await execute_native(llm=llm, survey=text, context=context, config=cfg)
    assertions = run.assertions
    assert len(assertions) == 364
    assert nested_json_size([a["claim_text"] for a in assertions]) > 36000
    admission = run.details["stages"]["capacity_admission"]
    assert len(admission["inventory"]["batch_reservations"]) == 3
    if oversized_stage != "assessment":
        assert len(admission["assessment"]["batch_reservations"]) == 3
    if oversized_stage == "global":
        assert admission["global"]["reservation"] > admission["global"]["context_limit"]
    if oversized_stage:
        assert admission["rejected_stage"] == oversized_stage
        assert len(llm.calls) == 3
        assert all(a["status"] == "unverifiable" for a in assertions)
        assert run.metrics()["faithfulness"]["score"] is None
    else:
        assert admission["complete_inventory"] == {
            "sample_count": 52,
            "statement_count": 364,
        }
        assert len(llm.calls) == 7
        assert (
            unpack_global_input(llm.calls[-1][0])["assertions"][0]["status"]
            == "supported"
        )
        assert run.metrics()["faithfulness"]["score"] == 1.0


@pytest.mark.asyncio
async def test_native_adapter_without_round_admission_keeps_original_inventory_gate(
    native_case, uplift_plan
):
    """A raw adapter cannot bypass the original aggregate limit on its own."""
    from litagent.config import load_config
    from litagent.eval.ragas_adapter import NativeRagasAdapter
    from litagent.eval.ragas_native import source_samples
    from litagent.safety.planned_budget import PlannedBudget

    class OversizedInventory(SemanticTransport):
        """Keep each statement within the individual bound but exceed the total."""

        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            raw = json.loads(response.content)
            for value in raw.values():
                value["statements"] *= 150
            response.content = json.dumps(raw)
            return response

    text, context = native_case
    _, _, _, samples = source_samples(text, context)
    cfg = load_config("config/operational-plus20.yaml").eval
    llm = OversizedInventory()
    llm._cost_budget = PlannedBudget(uplift_plan)
    adapter = NativeRagasAdapter(llm, cfg)
    with pytest.raises(ValueError, match="ragas_statement_capacity_exceeded"):
        await adapter.score(samples)
    assert len(llm.calls) == 1
