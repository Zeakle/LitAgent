"""Exercise the converged writer through real source, worker and evaluator consumers."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_complete_native_repair import source_selection

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.body_quality import WRITER_CONTENT_CONTRACT
from litagent.config import EvalConfig
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.scoped import build_source_fragments, evaluate_scoped
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    _writing_scopes,
    build_fact_cards,
    digest,
    load_targeted_repairs,
)
from litagent.report_render import build_report_source_units
from litagent.workbench import RunOptions
from litagent.writing_v2 import (
    BODY,
    CONTRACT,
    PROMPT_CONTRACT,
    SCOPE,
    WriterError,
    WriterSession,
    body_units,
    bound_fragments,
    classify_failure,
    compile_document,
    parse_response,
    repair_targets,
    schema_for,
    source_packet,
)


@pytest.fixture
def case():
    """Keep a valid second source outside a chapter's retrieval recommendation."""
    ledger = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "chunk_key": "abstract",
            "text": "On DeiT-S, EViT reduces computation.",
            "supporting_text": "On DeiT-S, EViT reduces computation.",
        },
        "b": {
            "evidence_id": "b",
            "paper_id": "relation",
            "chunk_key": "abstract",
            "text": "Relation networks learn a comparison function.",
            "supporting_text": "Relation networks learn a comparison function.",
        },
    }
    cards = build_fact_cards(ledger)
    selection = source_selection(cards)
    selection.section_evidence_ids["methods"] = ["a"]
    selection.writing_scope_policy = SCOPE
    packet, aliases = source_packet(
        cards,
        selection,
        "Explain mechanisms and limitations.",
        [
            {
                "requirement_id": "q1",
                "kind": "question",
                "text": "Explain mechanisms and limitations.",
            }
        ],
        "en",
    )
    by_eid = {cards[cid]["evidence_id"]: ref for ref, cid in aliases.items()}
    body = {
        "title": "Source bound analysis",
        **{
            k: [
                {
                    "sentences": [
                        {
                            "text": f"{k}: Relation networks learn a comparison function.",
                            "fact_card_refs": [by_eid["b"]],
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
                        "text": f"{k}: Relation networks learn a comparison function.",
                        "body_unit_refs": ["u3"],
                    }
                ]
            }
        ]
        for k in ("abstract", "conclusion")
    }
    return SimpleNamespace(
        ledger=ledger,
        cards=cards,
        selection=selection,
        packet=packet,
        aliases=aliases,
        by_eid=by_eid,
        body=body,
        summaries=summaries,
    )


def test_legacy_selection_bytes_and_fingerprints_remain_unchanged(case):
    """Legacy selection bytes and fingerprints remain unchanged."""
    old = copy.deepcopy(case.selection)
    old.writing_scope_policy = "section_v1"
    assert "writing_scope_policy" not in old.to_dict()
    assert EvidenceSelection.from_dict(old.to_dict()).to_dict() == old.to_dict()
    assert (
        EvidenceSelection.from_dict(case.selection.to_dict()).writing_scope_policy
        == SCOPE
    )
    assert RunOptions().generation_strategy == "single_v1"
    a = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v1",
    )
    b = a.model_copy(update={"generation_strategy": "body_summary_v2"})
    assert a.fingerprint() != b.fingerprint()


def test_global_pool_is_not_the_chapter_recommendation(case):
    """Global pool is not the chapter recommendation."""
    scopes = _writing_scopes(case.cards, "academic_v2", "en", case.selection)
    assert set(scopes["methods"]) == set(case.cards)
    assert case.packet["chapter_recommendations"]["methods"] == [case.by_eid["a"]]
    raw = json.dumps(case.body)
    assert parse_response(raw, schema_for("body", case.aliases)) == case.body
    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    assert "[E:b]" in draft and bindings["writer_contract"] == CONTRACT
    assert len(case.packet["sources"]) == 2


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "metadata", "empty", "null", "extra", "bad_json", "duplicate"],
)
def test_provider_errors_are_not_silently_repaired(case, mutation):
    """Provider errors are not silently repaired."""
    body = copy.deepcopy(case.body)
    unit = body["methods"][0]["sentences"][0]
    if mutation == "unknown":
        unit["fact_card_refs"] = ["c999"]
    if mutation == "metadata":
        unit["block_index"] = 0
    if mutation == "empty":
        unit["fact_card_refs"] = []
    if mutation == "null":
        unit["text"] = None
    if mutation == "extra":
        body["references"] = []
    raw = json.dumps(body)
    if mutation == "bad_json":
        raw = raw[:-1]
    if mutation == "duplicate":
        raw = '{"title":"a",' + raw[1:]
    with pytest.raises(WriterError):
        parse_response(raw, schema_for("body", case.aliases))


def test_source_identity_corruption_is_rejected(case):
    """Source identity corruption is rejected."""
    cards = copy.deepcopy(case.cards)
    next(iter(cards.values()))["source_quote"] = "Altered source"
    with pytest.raises(WriterError, match="source_hash"):
        source_packet(cards, case.selection, "question", [], "en")


def test_summary_dependencies_cannot_invent_body_or_paper(case):
    """Summary dependencies cannot invent body or paper."""
    summaries = copy.deepcopy(case.summaries)
    summaries["abstract"][0]["sentences"][0]["body_unit_refs"] = ["u999"]
    with pytest.raises(WriterError):
        compile_document(case.body, summaries, case.cards, case.aliases, "en")


def test_native_citation_scope_does_not_depend_on_punctuation(case):
    """Native citation scope does not depend on punctuation."""
    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"][0][
        "text"
    ] = "Relation networks learn comparisons; they use a learned comparison function."
    body["methods"][0]["sentences"].append(
        {
            "text": "On DeiT-S, EViT reduces computation.",
            "fact_card_refs": [case.by_eid["a"]],
        }
    )
    draft, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en"
    )
    fragments = bound_fragments(draft, bindings)
    own = next(
        fs
        for fs in fragments.values()
        if any("learn comparisons;" in f["source_quote"] for f in fs)
    )
    assert (
        len(own) == 2
        and own[0]["citation_aliases"] == ["b"]
        and own[1]["citation_aliases"] == ["a"]
    )
    tampered = copy.deepcopy(bindings)
    tampered["units"][0]["source_start"] += 1
    with pytest.raises(WriterError):
        bound_fragments(draft, tampered)
    with pytest.raises(WriterError):
        bound_fragments(draft + " fabricated", bindings)


def test_unbound_prose_and_omitted_bindings_are_rejected(case):
    """Unbound prose and omitted bindings are rejected."""
    draft, b = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    b["units"].pop()
    with pytest.raises(WriterError, match="coverage"):
        bound_fragments(draft, b)


class Provider:
    """Replace only model transport while retaining actual worker consumers."""

    def __init__(self, case, *, decline=False, invalid=False):
        """Initialize the controlled transport."""
        self.case = case
        self.calls = []
        self.messages = []
        self.decline = decline
        self.invalid = invalid

    async def chat(self, messages, **kwargs):
        """Return controlled provider output at the external transport boundary."""
        p = json.loads(messages[1]["content"])
        self.calls.append((p, kwargs))
        self.messages.append(copy.deepcopy(messages))
        if "targets" in p:
            result = (
                {
                    "outcome": "declined",
                    "explanation": "No safe replacement.",
                    "patches": [],
                }
                if self.decline
                else {
                    "outcome": "patched",
                    "explanation": "Corrected source-bound unit.",
                    "patches": [
                        {
                            "target_id": tid,
                            "sentences": [
                                {
                                    "text": "A corrected source-bound comparison.",
                                    "fact_card_refs": [self.case.by_eid["b"]],
                                }
                            ],
                        }
                        for tid in p["targets"]
                    ],
                }
            )
        elif "body_units" in p:
            result = self.case.summaries
        elif "draft" in p:
            result = {"issues": [], "explanation": "No controlled review defect."}
        elif "requested_body_section" in p:
            result = {
                key: copy.deepcopy(self.case.body[key])
                for key in kwargs["response_format"]["json_schema"]["schema"][
                    "properties"
                ]
            }
        else:
            result = self.case.body
        return LLMResponse(
            content="{" if self.invalid else json.dumps(result),
            model="offline",
            finish_reason="stop",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )


@pytest.mark.asyncio
async def test_actual_synthesis_reviewer_and_compiler_preserve_frozen_pool(
    case, tmp_path
):
    """Actual synthesis reviewer and compiler preserve frozen pool."""
    provider = Provider(case)
    selector = AsyncMock()
    selector.select.return_value = case.selection
    worker = SynthesisWorker(
        provider,
        evidence_selector=selector,
        run_id_provider=lambda: "writer-test",
        repair_state_path_provider=lambda: tmp_path / "repairs.json",
    )
    reviewer = ReviewerWorker(provider)
    adv = AdversarialReviewWorker(provider, worker, reviewer)
    task = SubTask(
        "adversarial_review",
        "Review controlled body",
        "adversarial_review",
        input_data={
            "query": case.packet["query"],
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v2",
            "report_language": "en",
            "review_structure": "academic_v2",
            "verification": {"judge": False, "jev": False},
            "upstream_results": {},
        },
    )
    # Frozen selection replaces retrieval only; all writing/review consumers are real.
    result = await adv.execute(task)
    assert result["evidence_selection"]["writing_scope_policy"] == SCOPE
    assert result["fact_cards"] == case.cards and result["total_rounds"] == 1
    assert len(provider.calls) == len(BODY) + 2
    assert (
        result["writing_bindings"]["report_sha256"]
        == hashlib.sha256(result["final_draft"].encode()).hexdigest()
    )
    assert bound_fragments(result["final_draft"], result["writing_bindings"])


@pytest.mark.asyncio
async def test_shared_repairs_refusal_and_restart_never_resend(case, tmp_path):
    """Shared repairs refusal and restart never resend."""
    p = Provider(case, decline=True)
    path = tmp_path / "repairs.json"
    s = WriterSession(p, path, "r", case.packet, case.aliases)
    target = {
        "x": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    with pytest.raises(WriterError, match="declined"):
        await s.patch(case.body, target, cause="evaluation", claim_count=5)
    assert len(load_targeted_repairs(path, "r")["attempts"]) == 1
    assert (
        classify_failure(WriterError("writer_repair_declined", "quality"))["family"]
        == "quality"
    )
    with pytest.raises(WriterError, match="no_replay"):
        await s.call(
            "repair_0", "system", {}, schema_for("repair", case.aliases, target)
        )
    assert len(p.calls) == 1


@pytest.mark.asyncio
async def test_two_repairs_cannot_change_unrelated_units_or_gain_third_attempt(
    case, tmp_path
):
    """Two repairs cannot change unrelated units or gain third attempt."""
    p = Provider(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    target = {
        "x": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    updated = await s.patch(case.body, target, cause="evaluation", claim_count=5)
    for k in BODY:
        if k != "methods":
            assert updated[k] == case.body[k]
    with pytest.raises(WriterError, match="no_progress"):
        await s.patch(
            updated,
            {
                "x": {
                    "path": ["methods", 0, 0],
                    "original": updated["methods"][0]["sentences"][0],
                }
            },
            cause="evaluation",
            claim_count=5,
        )
    with pytest.raises(WriterError, match="limit"):
        await s.patch(case.body, target, cause="evaluation")
    assert len(p.calls) == 2


@pytest.mark.asyncio
async def test_invalid_json_is_saved_and_not_retried(case, tmp_path):
    """Invalid json is saved and not retried."""
    p = Provider(case, invalid=True)
    path = tmp_path / "r.json"
    s = WriterSession(p, path, "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="json_invalid"):
        await s.call("body", "system", case.packet, schema_for("body", case.aliases))
    record = json.loads(s.journal_path.read_text())["calls"]["body"]
    assert record["raw"] == "{" and record["usage"]["prompt_tokens"] == 10
    s2 = WriterSession(p, path, "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="no_replay"):
        await s2.call("body", "system", case.packet, schema_for("body", case.aliases))
    assert len(p.calls) == 1


class EvaluatorTransport:
    """Script semantic decisions at the transport boundary of the actual v3 evaluator."""

    def __init__(self, *, unsupported=False, incomplete=False):
        """Initialize the controlled transport."""
        from test_batched_evaluation import BatchLLM

        self.base = BatchLLM(incomplete_global=incomplete)
        self.unsupported = unsupported

    async def chat(self, messages, **kwargs):
        """Return controlled provider output at the external transport boundary."""
        response = await self.base.chat(messages, **kwargs)
        payload = self.base.calls[-1]
        value = json.loads(response.content)
        if "cited_materials" in payload and self.unsupported:
            for d in value["decisions"]:
                d.update(
                    status="unsupported",
                    reason_code="insufficient_support",
                    explanation="Controlled mismatch in the actual cited original.",
                    support_basis=[],
                )
                for a in d["aspect_checks"]:
                    a.update(supported=False, support_ids=[])
        if "analyses" in payload.get("schema", {}).get("properties", {}):
            from litagent.body_quality import ROLES
            from litagent.review_structure import structure_sections

            titles = dict(
                structure_sections("academic_v2", payload.get("report_language", "en"))
            )
            # Evaluator payload stores language separately on some historical paths.
            if not any(
                u["section"] == titles["open_problems"]
                for u in payload["report_structure"]
            ):
                titles = dict(structure_sections("academic_v2", "zh-CN"))
            sections = {
                u["source_unit_id"]: u["section"] for u in payload["report_structure"]
            }
            grouped = {
                k: [
                    c["claim_id"]
                    for c in payload["assertions"]
                    if sections[c["source_unit_id"]] == titles[k]
                ]
                for k in ROLES
            }
            value["limitations"] = {
                "adequate": True,
                "claim_ids": grouped["open_problems"],
                "explanation": "Controlled source-backed limitation.",
            }
            value["analyses"] = [
                {
                    "section_key": k,
                    "adequate": True,
                    "claim_ids": grouped[k],
                    "explanation": "Controlled analytical coverage.",
                }
                for k in ROLES
            ]
        response.content = json.dumps(value)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("current_content", [False, True])
async def test_actual_v3_does_not_confuse_cross_chapter_permission_with_support(
    case, unsupported, current_content
):
    """Actual v3 does not confuse cross chapter permission with support."""
    from litagent.quality import strict_acceptance

    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    if not current_content:
        bindings.pop("content_contract")
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
    llm = EvaluatorTransport(unsupported=unsupported)
    details = await evaluate_scoped(
        llm=llm, survey=draft, context=context, config=EvalConfig()
    )
    assert details["coverage"]["complete"]
    assert strict_acceptance(details)["status"] == (
        "failed" if unsupported else "passed"
    )
    assert details["score"] == (0.0 if unsupported else 1.0)
    assert details.get("writer_content_contract") == (
        WRITER_CONTENT_CONTRACT if current_content else None
    )
    assert llm.base.calls[-1].get("writer_content_contract") == (
        WRITER_CONTENT_CONTRACT if current_content else None
    )
    if unsupported:
        assert details["unsupported_count"] > 0
    inventory = llm.base.calls[0]
    assert sum(len(u["source_fragments"]) for u in inventory["source_units"]) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "supported",
        "declined",
        "repaired",
        "second_repaired",
        "no_progress",
        "invalid_body",
    ],
)
async def test_frozen_runtime_uses_real_workers_report_exit_and_evaluation(
    case, tmp_path, monkeypatch, scenario
):
    """Frozen runtime uses real workers report exit and evaluation."""
    from litagent.config import load_config
    from litagent.frozen_writing import frozen_options, run_frozen
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy

    if scenario == "invalid_body":
        case.body["methods"][0]["sentences"][0]["text"] = "[E:borrowed] invalid markup"
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "public-source",
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            {
                "__search_queries__": {"search_0": "relation networks"},
                "search_0": [{"paper_id": "relation"}],
                "dedup": [{"paper_id": "relation"}],
                "relevance_gate": [{"paper_id": "relation"}],
            },
            source_outcomes=[
                {
                    "task_id": "search_0",
                    "query": "relation networks",
                    "source": "arxiv",
                    "status": "completed",
                    "result_count": 1,
                }
            ],
            started_at=1700000000,
        ),
        "requirements": case.packet["requirements"],
    }
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )

    class Transport(Provider):
        def __init__(self):
            """Initialize the controlled transport."""
            super().__init__(case)
            self.evaluator = EvaluatorTransport(unsupported=scenario != "supported")
            self.repairs = 0

        async def chat(self, messages, **kwargs):
            """Return controlled provider output at the external transport boundary."""
            p = json.loads(messages[1]["content"])
            if not any(
                key in p
                for key in (
                    "allowed_citation_aliases",
                    "cited_materials",
                    "report_structure",
                )
            ):
                assert messages[0]["content"].count(PROMPT_CONTRACT) == 1
                assert messages[0]["content"].count(WRITER_CONTENT_CONTRACT) == 1
            if "targets" in p:
                summary = "body_units" in p
                if not summary:
                    self.repairs += 1
                declined = scenario == "declined"
                self.evaluator.unsupported = self.repairs < (
                    2 if scenario == "second_repaired" else 1
                )
                return LLMResponse(
                    content=json.dumps(
                        {
                            "outcome": "declined" if declined else "patched",
                            "explanation": "Controlled terminal refusal or source-bound repair.",
                            "patches": (
                                []
                                if declined
                                else [
                                    {
                                        "target_id": tid,
                                        "sentences": [
                                            (
                                                p["targets"][tid]["original"]
                                                if scenario == "no_progress"
                                                else {
                                                    "text": (
                                                        "A source-bound summary."
                                                        if summary
                                                        else (
                                                            "Relation networks learn a comparison function."
                                                            if self.repairs == 1
                                                            else "A relation network uses a learned comparison function."
                                                        )
                                                    ),
                                                    (
                                                        "body_unit_refs"
                                                        if summary
                                                        else "fact_card_refs"
                                                    ): (
                                                        ["u3"]
                                                        if summary
                                                        else [case.by_eid["b"]]
                                                    ),
                                                }
                                            )
                                        ],
                                    }
                                    for tid in p["targets"]
                                ]
                            ),
                        }
                    ),
                    model="offline",
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "usage_available": True,
                    },
                )
            if (
                "allowed_citation_aliases" in p
                or "cited_materials" in p
                or "report_structure" in p
            ):
                return await self.evaluator.chat(messages, **kwargs)
            return await super().chat(messages, **kwargs)

    cfg = load_config()
    cfg.observability.enabled = False
    cfg.mcp_servers = {}
    agent = LitAgent(cfg, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"
    transport = Transport()
    try:
        result = await run_frozen(agent, bundle, "test-frozen-v2", transport=transport)
        if scenario == "invalid_body":
            assert not result["delivery"]["publishable"]
            assert result["metadata"]["writer_failure"]["code"] == "writer_text_invalid"
            assert result["metadata"]["writer_failure"]["body_section"] == "methods"
            assert result["evaluation"]["faithfulness"]["skipped"]
            assert len(transport.calls) == BODY.index("methods") + 1
            assert all("requested_body_section" in p for p, _ in transport.calls)
            return
        assert result["metadata"]["generation_strategy"] == "body_summary_v2"
        assert result["metadata"]["writing_bindings"]["writer_contract"] == CONTRACT
        # A controlled positive exercises automatic acceptance, never human approval.
        assert result["delivery"]["publishable"] == (
            scenario not in {"declined", "no_progress"}
        ), json.dumps(
            {
                "quality": result["quality"],
                "delivery": result["delivery"],
                "method": result["evaluation"]["faithfulness"]["details"].get(
                    "method_checks"
                ),
                "strict": result["evaluation"]["faithfulness"]["details"].get(
                    "strict_acceptance"
                ),
                "automatic": result["metadata"].get("automatic_acceptance"),
            }
        )
        assert (
            not result["metadata"]["verification"]["judge"]
            and not result["metadata"]["verification"]["jev"]
        )
        attempts = result["metadata"]["targeted_repair_state"]["attempts"]
        assert (
            len(attempts)
            == {
                "supported": 0,
                "declined": 1,
                "repaired": 1,
                "second_repaired": 2,
                "no_progress": 1,
            }[scenario]
        )
        history = result["metadata"].get("evidence_rewrite_attempts", [])
        if scenario == "second_repaired":
            assert not history[0]["committed"] and history[1]["committed"]
            assert result["survey"] != history[0]["candidate"]
        assert result["metadata"]["automatic_acceptance"]["actor"] == "system"
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
async def test_transaction_budget_rejection_does_not_consume_or_send(case, tmp_path):
    """Transaction budget rejection does not consume or send."""
    from litagent.safety.budget import CostBudget

    p = Provider(case)
    p._cost_budget = CostBudget(max_tokens=50000)
    p._cost_budget.configure_strict(halt_on_unknown_usage=True)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="budget"):
        await s.patch(
            case.body,
            {
                "x": {
                    "path": ["methods", 0, 0],
                    "original": case.body["methods"][0]["sentences"][0],
                }
            },
            cause="evaluation",
        )
    assert (
        not p.calls
        and not s.state["attempts"]
        and p._cost_budget.snapshot()["requests_sent"] == 0
    )


@pytest.mark.asyncio
async def test_summary_dependency_update_preserves_unaffected_prose(case, tmp_path):
    """Split body units without changing unrelated summary prose or premises."""
    from litagent.writing_v2 import refresh_summaries

    old = copy.deepcopy(case.body)
    new = copy.deepcopy(old)
    changed = old["introduction"][0]["sentences"][0]
    new["introduction"][0]["sentences"] = [
        {**changed, "text": "First supported premise."},
        {**changed, "text": "Second supported premise."},
    ]
    summaries = copy.deepcopy(case.summaries)
    summaries["abstract"][0]["sentences"][0]["body_unit_refs"] = ["u1"]

    class SummaryProvider(Provider):
        async def chat(self, messages, **kwargs):
            """Return only the affected abstract replacement."""
            p = json.loads(messages[1]["content"])
            self.calls.append(p)
            return LLMResponse(
                content=json.dumps(
                    {
                        "outcome": "patched",
                        "explanation": "Preserve changed premises.",
                        "patches": [
                            {
                                "target_id": next(iter(p["targets"])),
                                "sentences": [
                                    {
                                        "text": "Updated abstract.",
                                        "body_unit_refs": ["u1", "u2"],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                model="offline",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "usage_available": True,
                },
            )

    provider = SummaryProvider(case)
    s = WriterSession(provider, tmp_path / "s.json", "r", case.packet, case.aliases)
    from litagent.quality import begin_targeted_repair

    attempt = begin_targeted_repair(
        s.state, "evaluation", digest("controlled body change")
    )
    attempt.update(status="completed", replacement_lengths={"x": 2})
    result = await refresh_summaries(
        s, old, new, summaries, {"x": {"path": ["introduction", 0, 0]}}
    )
    assert (
        result["conclusion"][0]["sentences"][0]["text"]
        == summaries["conclusion"][0]["sentences"][0]["text"]
    )
    assert result["conclusion"][0]["sentences"][0]["body_unit_refs"] == ["u4"]
    assert result["abstract"][0]["sentences"][0]["body_unit_refs"] == ["u1", "u2"]
    assert len(provider.calls) == 1 and len(s.state["attempts"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_usage", "cancelled", "timeout"])
async def test_uncertain_writer_calls_are_durable_and_never_replayed(
    case, tmp_path, fault
):
    """Unknown responses and cancellation cannot be automatically resent."""
    import asyncio

    class BrokenProvider(Provider):
        async def chat(self, messages, **kwargs):
            """Return an unknown accounting response or interrupt transport."""
            self.calls.append(messages)
            if fault == "cancelled":
                raise asyncio.CancelledError()
            if fault == "timeout":
                raise TimeoutError("controlled timeout")
            return LLMResponse(
                content=json.dumps(case.body),
                model="offline",
                finish_reason="stop",
                usage={"usage_available": False},
            )

    p = BrokenProvider(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises((ValueError, TimeoutError, asyncio.CancelledError)):
        await s.call("body", "system", case.packet, schema_for("body", case.aliases))
    restored = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="no_replay"):
        await restored.call(
            "body", "system", case.packet, schema_for("body", case.aliases)
        )
    assert len(p.calls) == 1


def test_mathematical_less_than_is_not_layout_and_cannot_inject_html(case):
    """A mathematical bound is valid scientific prose, while HTML is rejected."""
    body = copy.deepcopy(case.body)
    body["methods"][0]["sentences"][0][
        "text"
    ] = "The measured value is <50 under the recorded conditions."
    draft, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en"
    )
    assert "<50" in draft and bound_fragments(draft, bindings)
    body["methods"][0]["sentences"][0]["text"] = "<script>malicious()</script>"
    with pytest.raises(WriterError, match="text_invalid"):
        compile_document(body, case.summaries, case.cards, case.aliases, "en")


@pytest.mark.asyncio
async def test_v2_native_schema_uses_actual_sdk_zero_retries_and_stage_models(
    case, tmp_path
):
    """Exercise actual SDK encoding, usage accounting and model routing."""
    import httpx
    from openai import AsyncOpenAI

    from litagent.llm.client import OpenAICompatibleClient
    from litagent.llm.routing import ModelBoundClient
    from litagent.safety.budget import CostBudget

    outputs = [
        *[
            {
                key: copy.deepcopy(case.body[key])
                for key in (["title", section] if i == 0 else [section])
            }
            for i, section in enumerate(BODY)
        ],
        case.summaries,
        {"issues": [], "explanation": "No controlled issue."},
    ]
    sent = []

    def transport(request):
        """Replace HTTP only; all provider and worker contracts remain real."""
        p = json.loads(request.content)
        sent.append(p)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": p["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(outputs[len(sent) - 1]),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
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
    worker = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=case.selection)
        ),
        run_id_provider=lambda: "sdk-v2",
        repair_state_path_provider=lambda: tmp_path / "r.json",
    )
    task = SubTask(
        "adversarial_review",
        "test",
        "adversarial_review",
        input_data={
            "query": case.packet["query"],
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v2",
            "review_structure": "academic_v2",
            "report_language": "en",
            "verification": {"judge": False, "jev": False},
            "upstream_results": {},
        },
    )
    try:
        output = await AdversarialReviewWorker(
            pro, worker, ReviewerWorker(lite)
        ).execute(task)
        assert output["writing_bindings"]["writer_contract"] == CONTRACT
        assert [p["model"] for p in sent] == ["pro"] * (len(BODY) + 1) + ["lite"]
        assert all(
            p["response_format"]["json_schema"]["strict"]
            and p["thinking"]["type"] == "disabled"
            for p in sent
        )
        assert budget.used == 210 and budget.snapshot()["requests_sent"] == 7
    finally:
        await owner.close()
