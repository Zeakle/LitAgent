"""Run flat model responses through actual SDK and existing consumers."""

import copy
import json

import pytest
from test_native_responses import owner_for
from test_native_wire_session import Gateway, make_session
from test_writer_v2 import case  # noqa: F401

from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest, strict_acceptance
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    compile_document,
    writer_budget_plan,
)


class FlatGateway(Gateway):
    """Emit the shallow wire at HTTP only, retaining the real application path."""

    def __init__(self, *args, **kwargs):
        """Resolve the new provider input view inside the HTTP mock only."""
        super().__init__(*args, **kwargs)
        original = self.evaluator.chat

        async def chat(messages, **options):
            messages = copy.deepcopy(messages)
            payload = json.loads(messages[1]["content"])
            if payload.get("global_input_contract") == "unit-local-global-input-v1":
                # Decode only inside the fake HTTP provider. Actual outbound
                # bytes, native schema, reservation and host consumers stay real.
                units, sections = {}, {}
                groups = {
                    kind: []
                    for kind in ("scientific", "report_structure", "runtime_method")
                }
                layout = []
                for section in payload["report_sections"]:
                    for part in section["layout"]:
                        if "fragment" not in part:
                            layout.append(copy.deepcopy(part))
                            continue
                        entry = {
                            "source_fragment_id": part["fragment"],
                            "source_unit_id": part["unit"],
                            "section": section["section"],
                            "citation_refs": part["citations"],
                            "assertions_by_kind": {
                                k: part[k] for k in groups if k in part
                            },
                        }
                        entry.update(
                            {k: part[k] for k in ("text", "prose") if k in part}
                        )
                        layout.append(entry)
                payload["report_layout"] = layout
                payload.update(
                    body_analysis_contract="body-analysis-v1",
                    global_findings_contract="anchored-global-findings-v2",
                    evaluation_input_contract="flat-evaluation-input-v1",
                    global_identity_contract="flat-global-identities-v1",
                    report_input_contract="host-report-input-v1",
                    writer_content_contract="grounded-writer-content-v1",
                    summary_constraints_contract="effective-summary-limits-v1",
                )
                for entry in layout:
                    if "source_fragment_id" not in entry:
                        continue
                    uid, fid = entry["source_unit_id"], entry["source_fragment_id"]
                    fragment = {
                        "source_fragment_id": fid,
                        "citation_aliases": [
                            payload["citation_aliases"][ref - 1]
                            for ref in entry["citation_refs"]
                        ],
                    }
                    if "text" in entry:
                        fragment["text"] = entry["text"]
                    units.setdefault(uid, []).append(fragment)
                    if uid not in sections.setdefault(entry["section"], []):
                        sections[entry["section"]].append(uid)
                    for kind, rows in entry["assertions_by_kind"].items():
                        for row in rows:
                            cid, text, refs = row
                            groups[kind].append([int(cid[1:]), cid, fid, text, refs])
                payload["source_units"] = units
                payload["source_units_by_section"] = sections
                payload["assertions_by_kind"] = groups
                payload["assertion_columns"] = [
                    "sequence",
                    "claim_id",
                    "source_fragment_id",
                    "claim_text",
                    "evidence_aliases",
                ]
            if isinstance(payload["source_units"], dict):
                payload["source_units"] = [
                    {"source_unit_id": uid, "source_fragments": fragments}
                    for uid, fragments in payload["source_units"].items()
                ]
                payload["report_structure"] = [
                    {"source_unit_id": uid, "section": section}
                    for section, ids in payload["source_units_by_section"].items()
                    for uid in ids
                ]
            texts = {
                p["source_fragment_id"]: p["prose"]
                for p in payload.get("report_layout", [])
                if "source_fragment_id" in p
            }
            fragments = {}
            parents = {}
            for unit in payload["source_units"]:
                if "fragment_columns" in payload:
                    unit["source_fragments"] = [
                        dict(zip(payload["fragment_columns"], row))
                        for row in unit["source_fragments"]
                    ]
                for fragment in unit["source_fragments"]:
                    if fragment.get("context_only") or (
                        "fragment_columns" in payload
                        and fragment["source_fragment_id"] is None
                    ):
                        continue
                    fid = fragment["source_fragment_id"]
                    parents[fid] = unit["source_unit_id"]
                    fragments[fid] = fragment
                    if "prose" in fragment:
                        texts[fid] = fragment.pop("prose")
                    if fid in texts:
                        aliases = fragment.get("citation_aliases", [])
                        fragment["text"] = texts[fid] + (
                            " " + " ".join(f"[E:{a}]" for a in aliases)
                            if aliases
                            else ""
                        )
            if "assertion_columns" in payload:
                if "assertions_by_kind" in payload:
                    # Adapt only the controlled HTTP peer to the new host view.
                    # Real schemas, classification and source checks still run.
                    expanded = [
                        {
                            **dict(zip(payload["assertion_columns"], row, strict=True)),
                            "kind": kind,
                        }
                        for kind, rows in payload["assertions_by_kind"].items()
                        for row in rows
                    ]
                    expanded.sort(key=lambda claim: claim["sequence"])
                    assert [c.pop("sequence") for c in expanded] == list(
                        range(1, len(expanded) + 1)
                    )
                    payload["assertions"] = expanded
                else:
                    payload["assertions"] = [
                        dict(zip(payload["assertion_columns"], row))
                        for row in payload["assertions"]
                    ]
                for claim in payload["assertions"]:
                    if claim["kind"] is None:
                        claim["kind"] = "scientific"
                    if "fragment_columns" in payload:
                        claim["source_unit_id"] = parents[claim["source_fragment_id"]]
                        if claim["evidence_aliases"] is None:
                            claim.pop("evidence_aliases")
                        else:
                            claim["evidence"] = [
                                {"evidence_alias": a, "material_ref": a}
                                for a in claim.pop("evidence_aliases")
                            ]
            if "cited_materials" in payload:
                for claim in payload["assertions"]:
                    if "evidence" not in claim and "evidence_aliases" not in claim:
                        claim["evidence"] = [
                            {"evidence_alias": a, "material_ref": a}
                            for a in fragments[claim["source_fragment_id"]][
                                "citation_aliases"
                            ]
                        ]
            messages[1]["content"] = json.dumps(payload)
            response = await original(messages, **options)
            if (
                self.fault == "unbound_runtime"
                and "allowed_citation_aliases" in payload
            ):
                value = json.loads(response.content)
                first = next(iter(value["fragments"].values()))
                first["assertions"][0]["kind"] = "runtime_method"
                response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Generate an independent flat fixture, with no application-side codec."""
        if "requested_body_section" in payload:
            key = payload["requested_body_section"]
            result = {"title": self.case.body["title"]} if key == BODY[0] else {}
            result["units"] = [
                {**copy.deepcopy(u), "paragraph_start": si == 0}
                for p in self.case.body[key]
                for si, u in enumerate(p["sentences"])
            ]
            return result
        value = super().answer(payload, native_schema)
        if "targets" in payload:
            value["patches"] = [
                {"target_id": p["target_id"], **u}
                for p in value["patches"]
                for u in p["sentences"]
            ]
        elif "body_units" in payload:
            value = {
                key: [
                    {**u, "paragraph_start": si == 0}
                    for p in paragraphs
                    for si, u in enumerate(p["sentences"])
                ]
                for key, paragraphs in value.items()
            }
        return value


async def setup_flat(case, tmp_path, gateway):
    """Install the complete v5 plan and model-bound SDK clients."""
    owner, budget, events = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    assert session.wire_contract == "flat-writer-v1"
    return owner, budget, events, session


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
async def test_flat_sdk_generation_and_complete_scientific_evaluation(
    case, tmp_path, unsupported
):
    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped

    gateway = FlatGateway(case, unsupported=unsupported)
    owner, budget, _, session = await setup_flat(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summary = await session.summaries(body, "summary_initial")
        assert body == case.body and summary == case.summaries
        survey, bindings = compile_document(
            body, summary, case.cards, case.aliases, "en", envelope=session.envelope
        )
        await session.call(
            "reviewer",
            "Review the complete draft.",
            {"report_layout": []},
            {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 128,
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["issues", "explanation"],
                "additionalProperties": False,
            },
            stage="reviewer",
            max_tokens=4096,
        )
        details = await evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=survey,
            context={
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
            },
            config=EvalConfig(),
        )
        assert details["coverage"]["complete"]
        assert (details["unsupported_count"] > 0) is unsupported
        if unsupported:
            assert strict_acceptance(details)["status"] == "failed"
        assert budget.snapshot()["usage_uncertain"] is False
        assert all(
            call["writer_wire_contract"] == "flat-writer-v1"
            for call in session.journal["calls"].values()
        )
        assert all(
            row["request"].get("thinking", {"type": "disabled"}) == {"type": "disabled"}
            for row in gateway.sent
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_flat_two_shared_patches_and_third_is_not_sent(case, tmp_path):
    gateway = FlatGateway(case)
    owner, _, _, session = await setup_flat(case, tmp_path, gateway)
    try:
        original = copy.deepcopy(case.body)
        unit = original["methods"][0]["sentences"][0]
        target = {
            "m": {
                "path": ["methods", 0, 0],
                "original": unit,
                "explanation": "Correct the authorized method.",
            }
        }
        first = await session.patch(original, target, cause="local")
        second = await session.patch(original, target, cause="evaluation")
        assert first == second and original == case.body
        with pytest.raises(WriterError, match="repair_limit"):
            await session.patch(original, target, cause="evaluation")
        assert len(gateway.sent) == 2
        assert len(session.state["attempts"]) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_flat_summary_repair_uses_only_frozen_body_premises(case, tmp_path):
    gateway = FlatGateway(case)
    owner, _, _, session = await setup_flat(case, tmp_path, gateway)
    packet = copy.deepcopy(session.packet)
    try:
        target = {
            "abstract": {
                "path": ["abstract", 0, 0],
                "original": case.summaries["abstract"][0]["sentences"][0],
                "explanation": "Preserve the actual frozen body premise.",
            }
        }
        candidate = await session.patch(
            case.summaries,
            target,
            cause="evaluation",
            summary_body=case.body,
        )
        payload = gateway.sent[-1]["payload"]
        assert candidate != case.summaries
        assert payload["body_units"] == body_units(case.body)
        assert payload["requirements"] == packet["requirements"]
        assert (
            not {
                "sources",
                "cards",
                "chapter_recommendations",
                "host_priority_card_refs",
            }
            & payload.keys()
        )
        assert session.packet == packet
        assert len(session.state["attempts"]) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["outer_eof", "unit_eof", "missing_usage", "timeout", "cancel"]
)
async def test_flat_raw_failures_remain_durable_and_never_replay(case, tmp_path, fault):
    gateway = FlatGateway(case, fault=fault)
    owner, _, _, session = await setup_flat(case, tmp_path, gateway)
    try:
        if fault == "outer_eof":
            await session.initial_body()
            assert all(
                c.get("syntax_normalization") for c in session.journal["calls"].values()
            )
        else:
            with pytest.raises(BaseException):
                await session.initial_body()
        count = len(gateway.sent)
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.initial_body()
        assert len(gateway.sent) == count
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize("unbound_runtime", [False, True])
@pytest.mark.parametrize(
    "budget_policy", ["planned-budget-v5", "planned-budget-v6", "planned-budget-v7"]
)
async def test_flat_real_frozen_worker_runner_and_report_exit(
    case, tmp_path, monkeypatch, unsupported, unbound_runtime, budget_policy
):
    from litagent.config import load_config
    from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy
    from litagent.writing_v2 import EOF_CONTAINER_POLICY

    if unbound_runtime:
        case.body["introduction"][0]["sentences"][0][
            "text"
        ] = "This report searched five public papers."
    gateway = FlatGateway(
        case,
        unsupported=unsupported,
        fault="unbound_runtime" if unbound_runtime else None,
    )
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"], {}, source_outcomes=[], started_at=1700000000
        ),
        "fact_cards_sha256": digest(case.cards),
    }
    before = copy.deepcopy(bundle)
    plan = plan_frozen_budget(bundle, budget_policy=budget_policy)
    try:
        result = await run_frozen(
            agent,
            bundle,
            "flat-frozen",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy=budget_policy,
            expected_budget_plan=plan,
        )
        assert bundle == before
        assert result["metadata"]["generation_strategy"] == "body_summary_v3"
        assert (
            result["metadata"]["writer_document"]["writer_wire_contract"]
            == "flat-writer-v1"
        )
        details = result["evaluation"]["faithfulness"]["details"]
        if unbound_runtime:
            assert not details["coverage"]["complete"]
            assert details["stages"]["unbound_runtime_claims"]
            assert not result["delivery"]["publishable"]
            assert not details["stages"].get("batches")
        else:
            assert details["coverage"]["complete"]
        assert agent._cost_budget._planned_backend.plan == plan
        if unsupported:
            assert not result["delivery"]["publishable"]
            if not unbound_runtime:
                guidance = [
                    row["payload"]["previous_local_failure"]
                    for row in gateway.sent
                    if "targets" in row["payload"]
                    and (row["payload"].get("previous_local_failure") or {}).get(
                        "diagnostics"
                    )
                ]
                assert guidance and len(guidance[0]["diagnostics"]) > 1
        assert len(result["metadata"]["targeted_repair_state"]["attempts"]) <= 2
        assert all(
            row["request"].get("thinking", {"type": "disabled"}) == {"type": "disabled"}
            for row in gateway.sent
        )
    finally:
        await agent.cleanup()
