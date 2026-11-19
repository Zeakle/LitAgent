"""Verify compact status transport without altering scientific verdicts."""

import copy
import csv
import json
from pathlib import Path

import pytest

from litagent.eval.engine_policy import (
    ARCHIVED_INLINE_UNCITED_ENGINE,
    configured_engine,
    validate_engine,
)
from litagent.eval.ragas_global_wire import (
    STATUS_CONTRACT,
    STATUS_INSTRUCTION,
    pack_global_input,
    unpack_global_input,
)
from litagent.safety.budget import strict_request_reservation


def payload():
    """Keep identical prose at distinct locations with distinct citations."""
    claims = [
        {
            "claim_id": f"claim_{i}",
            "claim_text": '相同“表述”\\n"😀"',
            "kind": "scientific",
            "section": "比较",
            "source_unit_id": f"unit:{i}:0",
            "source_fragment_id": f"fragment:{i}",
            "evidence_ids": [f"arxiv:{i}"],
            "status": state,
        }
        for i, state in enumerate(("supported", "unsupported", "unverifiable"))
    ]
    return {
        "assertions": claims,
        "source_units": [
            {
                "source_unit_id": c["source_unit_id"],
                "text": f'{c["claim_text"]}[E:{c["evidence_ids"][0]}]。',
                "extracted_claim_ids": [c["claim_id"]],
            }
            for c in claims
        ],
        "review_methodology": {},
    }


@pytest.mark.parametrize("by_unit", [False, True])
def test_status_roundtrip_preserves_every_occurrence(by_unit):
    original = payload()
    untouched = copy.deepcopy(original)
    wire = pack_global_input(
        original, by_unit=by_unit, inline_sources=True, status_codes=True
    )
    assert wire["assertion_status_contract"] == STATUS_CONTRACT
    rows = (
        [u["claims"][0] for u in wire["source_units"]]
        if by_unit
        else wire["assertions"]
    )
    assert [r[-1] for r in rows] == [1, 0, -1]
    assert unpack_global_input(wire) == untouched == original
    assert [r[-1] for r in rows] == [1, 0, -1]
    for code, name in ((1, "supported"), (0, "unsupported"), (-1, "unverifiable")):
        assert f"{code} for {name}" in STATUS_INSTRUCTION


@pytest.mark.parametrize("by_unit", [False, True])
@pytest.mark.parametrize("invalid", [True, False, 1.0, "1", None, 2, -2, []])
def test_illegal_status_codes_are_not_coerced(by_unit, invalid):
    wire = pack_global_input(payload(), by_unit=by_unit, status_codes=True)
    row = wire["source_units"][0]["claims"][0] if by_unit else wire["assertions"][0]
    row[-1] = invalid
    with pytest.raises(ValueError, match="ragas_global_status_invalid"):
        unpack_global_input(wire)


@pytest.mark.parametrize(
    "fault", ["missing_contract", "unknown_contract", "missing", "duplicate", "owner"]
)
def test_status_projection_rejects_lost_or_misbound_rows(fault):
    wire = pack_global_input(payload(), status_codes=True)
    if fault == "missing_contract":
        del wire["assertion_status_contract"]
    elif fault == "unknown_contract":
        wire["assertion_status_contract"] = "future"
    elif fault == "missing":
        wire["assertions"].pop()
    elif fault == "duplicate":
        wire["assertions"].append(wire["assertions"][0])
    else:
        wire["assertions"][0][0] = 1
    with pytest.raises(ValueError, match="ragas_global_status_"):
        unpack_global_input(wire)


def test_old_wire_and_archived_engine_remain_readable_only():
    original = payload()
    assert unpack_global_input(pack_global_input(original)) == original
    assert (
        validate_engine(ARCHIVED_INLINE_UNCITED_ENGINE, archived=True)
        == ARCHIVED_INLINE_UNCITED_ENGINE
    )
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(ARCHIVED_INLINE_UNCITED_ENGINE)
    from litagent.config import load_config

    config = load_config("config/separate-review-evaluation.yaml")
    assert "global_status_contract" not in configured_engine(config)
    config.safety.planned_run_limits.citation_contract = "flat-writer-v2"
    assert configured_engine(config)["global_status_contract"] == STATUS_CONTRACT


def vit_replay():
    """Reconstruct immutable local trial 03 through production context builders."""
    from litagent.closeout import PRESET
    from litagent.config import load_config
    from litagent.eval.ragas_native import source_samples
    from litagent.output_capacity import configured_output_tokens
    from litagent.run_limits import configured_run_limits
    from litagent.runner import LitAgent
    from litagent.writing_v2 import writer_budget_plan

    root = Path("artifacts/acceptance/inline-citations-vit-235d5bf1dba2")
    path = root / "vit-235d5bf1dba2-inline-reassessment-03/result.json"
    if not path.exists():
        pytest.skip("requires the immutable local ViT 03 archive")
    old = json.loads(path.read_bytes())
    draft = json.loads((root / "runs/vit-235d5bf1dba2-inline-draft.json").read_bytes())
    original = json.loads(
        Path(
            "artifacts/acceptance/workbench-vit-386-20261110/runs/235d5bf1dba2.json"
        ).read_bytes()
    )
    metadata = draft["report"]["metadata"]
    survey = draft["report"]["survey"]
    results = {
        "extract": original["nodes"]["worker:extract"]["output"],
        "adversarial_review": {
            **metadata,
            "final_draft": survey,
            "evidence_selection": {
                "valid": True,
                **original["nodes"]["worker:adversarial_review"]["output"][
                    "evidence_selection"
                ],
            },
        },
    }
    config = load_config("config/separate-review-evaluation.yaml")
    config.safety.planned_run_limits.citation_contract = "flat-writer-v2"
    config.observability.enabled = False
    config.llm.base_url = "https://offline.invalid/api/v3"
    doc = metadata["writer_document"]
    plan = writer_budget_plan(
        doc["packet"],
        doc["aliases"],
        budget_policy="planned-budget-v7",
        methodology=metadata["review_methodology"],
        capacity=config.capacity,
        run_limits=configured_run_limits(config),
        semantic_engine=configured_engine(config),
        output_tokens=configured_output_tokens(config),
        **PRESET,
    )
    agent = LitAgent(config)
    context = agent._build_eval_context(
        query=draft["query"], survey=survey, results=results
    )
    _, units, _, samples = source_samples(survey, context)
    return old, draft, results, config, plan, context, units, samples


def global_request(case, assertions):
    """Use the same full request builder as preflight, sending and reassessment."""
    from litagent.eval.engine_policy import (
        anchored_native_review,
        native_unit_response_contract,
        unit_native_review,
    )
    from litagent.eval.ragas_native import global_review_request

    _, draft, _, config, plan, context, units, _ = case
    env = plan["envelope"]
    return global_review_request(
        draft["report"]["survey"],
        context,
        config.eval,
        units,
        assertions,
        force_compact=True,
        anchored=anchored_native_review(env),
        unit_view=unit_native_review(env),
        unit_response_contract=native_unit_response_contract(env),
    )


def test_vit03_complete_request_and_pending_status_bound():
    case = vit_replay()
    old, _, _, config, plan, *_ = case
    details = old["evaluation"]["faithfulness"]["details"]
    claims = details["assertions"]
    assert len(claims) == 505
    assert sum(c["reason_code"] == "uncited_claim" for c in claims) == 101
    canonical, messages, schema = global_request(case, claims)
    assert unpack_global_input(json.loads(messages[1]["content"])) == canonical
    # Reconstruct the old request without silently changing its archived verdict.
    previous = copy.deepcopy(messages)
    previous[0]["content"] = previous[0]["content"].removesuffix(
        "\n" + STATUS_INSTRUCTION
    )
    previous[1]["content"] = json.dumps(
        pack_global_input(canonical, inline_sources=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    limit = plan["context_reservation_limit"]
    assert (
        strict_request_reservation(previous, config.eval.max_tokens, schema)
        == 213903
        > limit
    )
    ceiling = strict_request_reservation(messages, config.eval.max_tokens, schema)
    assert ceiling <= limit == 212337
    for state in ("supported", "unsupported"):
        finished = copy.deepcopy(claims)
        for claim in finished:
            if claim["status"] == "unverifiable":
                claim["status"] = state
        _, wire, final_schema = global_request(case, finished)
        assert final_schema == schema
        assert (
            strict_request_reservation(wire, config.eval.max_tokens, final_schema)
            <= ceiling
        )
    assert old["evaluation"]["faithfulness"]["score"] is None


def test_legal_long_statements_still_need_full_global_admission():
    """A passing fixed replay must not be reported as a capacity-wide proof."""
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

    from litagent.safety.planned_budget import nested_json_size

    case = vit_replay()
    old, _, _, config, plan, *_ = case
    assertions = copy.deepcopy(
        old["evaluation"]["faithfulness"]["details"]["assertions"]
    )
    for claim in assertions[:3]:
        claim["claim_text"] = "x" * 2394
        StatementGeneratorOutput.model_validate(
            {"statements": [claim["claim_text"]]}, strict=True
        )
        assert (
            nested_json_size(claim["claim_text"])
            <= plan["envelope"]["claim_max_json_bytes"]
        )
    _, messages, schema = global_request(case, assertions)
    assert (
        strict_request_reservation(messages, config.eval.max_tokens, schema)
        > plan["context_reservation_limit"]
    )


@pytest.mark.parametrize("oversized", [False, True])
def test_full_admission_preserves_future_slots_and_reports_global_limit(oversized):
    from types import SimpleNamespace

    from litagent.eval.ragas_adapter import NativeRagasAdapter
    from litagent.eval.ragas_native import _check_native_admission
    from litagent.safety.budget import CostBudget
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    old, draft, _, config, plan, context, units, samples = vit_replay()
    details = old["evaluation"]["faithfulness"]["details"]
    for sample, recorded in zip(samples, details["partial_samples"], strict=True):
        assert sample["id"] == recorded["id"]
        sample["statements"] = copy.deepcopy(recorded["statements"])
        sample["verdicts"] = copy.deepcopy(recorded["verdicts"] or [])
        if not sample["contexts"]:
            sample["assessment_origin"] = "host_uncited"
    if oversized:
        samples[0]["statements"][:3] = ["x" * 2394] * 3
    budget = CostBudget(config.safety.max_cost_tokens)
    budget.activate_plan(plan)
    before = budget.snapshot()
    llm = SimpleNamespace(_cost_budget=budget)
    adapter = NativeRagasAdapter(
        llm, config.eval, keyed_assessment=True, inline_scopes=True
    )
    progress = {}
    args = dict(
        llm=llm,
        survey=draft["report"]["survey"],
        context=context,
        config=config.eval,
        units=units,
        adapter=adapter,
        envelope=plan["envelope"],
        progress=progress,
    )
    if oversized:
        with pytest.raises(ValueError, match="budget_plan_contract_breached"):
            _check_native_admission("assessment", samples, **args)
        failure = progress["capacity_admission"]["rejection"]
        assert failure["reservation"] == 214091
        assert failure["unexecuted_stages"] == ["assessment", "global"]
        assert failure["request_cap"] == failure["context_limit"] == 212337
    else:
        batches = _check_native_admission("assessment", samples, **args)
        assert len(batches) == 5
        assert progress["capacity_admission"]["global"]["reservation"] == 207077
    assert budget.snapshot() == before
    validate_accounting_snapshot(before, expected_plan=plan)


def uncited_audit(case):
    """Audit migration provenance without granting sources to uncited claims."""
    old, draft, *_ = case
    report = draft["report"]
    meta = report["metadata"]
    migration = {
        tuple(row["path"]): row for row in meta["citation_migration"]["ranges"]
    }
    scopes = {
        (s["source_start"], s["source_end"]): s
        for s in meta["writing_bindings"]["citation_scopes"]
    }
    packet = meta["writer_document"]["packet"]
    original = json.loads(
        Path(
            "artifacts/acceptance/workbench-vit-386-20261110/runs/235d5bf1dba2.json"
        ).read_bytes()
    )
    prior_units = {
        tuple(unit["path"]): unit
        for unit in original["report"]["metadata"]["writing_bindings"]["units"]
    }
    by_evidence = {c["evidence_id"]: c["source_ref"] for c in packet["cards"].values()}
    rows = []
    for claim in old["evaluation"]["faithfulness"]["details"]["assertions"]:
        if claim.get("reason_code") != "uncited_claim":
            continue
        scope = scopes[claim["source_start"], claim["source_end"]]
        entry = migration[tuple(scope["path"])]
        original = entry["ranges"][scope["scope_index"]]
        assert original["refs"] == scope["evidence_ids"] == claim["evidence_ids"] == []
        assert (
            report["survey"][claim["source_start"] : claim["source_end"]]
            == claim["source_quote"]
        )
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "source_quote": claim["source_quote"],
                "source_start": claim["source_start"],
                "source_end": claim["source_end"],
                "path": scope["path"],
                "scope_index": scope["scope_index"],
                "previous_unit_refs": entry["old_refs"],
                "previous_unit_evidence_ids": prior_units[tuple(scope["path"])][
                    "evidence_ids"
                ],
                "actual_scope_evidence_ids": [],
                "migration_reason": original["basis"],
                "reason_code": claim["reason_code"],
                "previous_source_refs": [
                    by_evidence[eid]
                    for eid in prior_units[tuple(scope["path"])]["evidence_ids"]
                ],
                "classification_status": "requires_semantic_review",
                "binding_check": "matches_explicit_migration",
            }
        )
    return {
        "report_sha256": old["report_sha256"],
        "rows": rows,
        "sources": packet["sources"],
        "evidence_snapshot_sha256": old["evidence_snapshot_sha256"],
    }


def test_vit03_uncited_claims_match_explicit_migration_without_lending_sources():
    audit = uncited_audit(vit_replay())
    assert len(audit["rows"]) == 101
    assert all(
        row["migration_reason"] == "no_local_source_assigned" for row in audit["rows"]
    )
    assert all(not row["actual_scope_evidence_ids"] for row in audit["rows"])


def test_request_rejection_is_shared_by_public_formats(tmp_path):
    from test_benchmark_evaluation_execution import result_fixture

    from litagent.benchmark.quality_suite import export_bundle
    from litagent.eval.results import public_summary, sanitize_summary, summary_tables

    case = vit_replay()
    old, draft, *_ = case
    report = copy.deepcopy(draft["report"])
    report["evaluation"] = copy.deepcopy(old["evaluation"])
    rejection = {
        "code": "global_request_preflight_exceeded",
        "stage": "global",
        "reservation": 214091,
        "context_limit": 212337,
        "request_cap": 212337,
        "unexecuted_stages": ["assessment", "global"],
    }
    report["evaluation"]["faithfulness"]["details"]["stages"]["capacity_admission"][
        "rejection"
    ] = rejection
    summary = public_summary(report)
    safe = sanitize_summary(summary)
    assert safe["request_preflight"] == rejection
    assert safe["metrics"]["faithfulness"]["score"] is None
    markdown, html = summary_tables([{"case_id": "vit", "evaluation_summary": safe}])
    for document in (markdown, html, json.dumps(safe)):
        assert "214091" in document and "212337" in document
    fixture = result_fixture()
    fixture["observations"][0]["evaluation_summary"] = safe
    export_bundle(tmp_path, fixture)
    exported = json.loads((tmp_path / "benchmark.json").read_text())
    assert exported["observations"][0]["evaluation_summary"] == safe
    with (tmp_path / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]["evaluation_summary"]) == safe
    for suffix in ("md", "html"):
        content = (tmp_path / f"benchmark.{suffix}").read_text()
        assert "Global request preflight exceeded" in content
        assert "214091" in content and "212337" in content
    summary["request_preflight"]["reservation"] = True
    assert "request_preflight" not in sanitize_summary(summary)


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_quality_view_explains_preflight_without_claiming_science_failure(locale):
    from test_quality_view import render_repair_state

    failure = {
        "code": "global_request_preflight_exceeded",
        "stage": "global",
        "reservation": 214091,
        "context_limit": 212337,
        "request_cap": 212337,
        "unexecuted_stages": ["assessment", "global"],
    }
    rendered = render_repair_state(
        {
            "evaluation": {
                "faithfulness": {
                    "status": "error",
                    "score": None,
                    "details": {
                        "stages": {"capacity_admission": {"rejection": failure}}
                    },
                }
            },
            "quality": {"status": "unverified"},
            "delivery": {"status": "blocked"},
        },
        locale,
    )
    assert "214091 / 212337" in rendered["text"]
    assert (
        "全局请求预检超限" if locale == "zh-CN" else "Global request preflight exceeded"
    ) in rendered["text"]
    assert ("不是实际费用" if locale == "zh-CN" else "not billed usage") in rendered[
        "text"
    ]


@pytest.mark.asyncio
async def test_vit03_replay_reaches_runner_repair_without_publishing(
    tmp_path, monkeypatch
):
    """Replay actual extraction through Reviewer, NLI, global and runner repair."""
    import importlib.util

    import httpx

    old, draft, results, config, plan, *_ = vit_replay()
    root = Path("artifacts/acceptance/inline-citations-vit-235d5bf1dba2")
    monkeypatch.syspath_prepend(str(root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "vit_status_replay", root / "offline-reassessment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = json.loads(
        Path(
            "artifacts/acceptance/workbench-vit-386-20261110/runs/235d5bf1dba2.json"
        ).read_bytes()
    )
    prior = archive["nodes"]["worker:adversarial_review"]["output"]
    source = copy.deepcopy(results["adversarial_review"])
    source.update(
        draft=draft["report"]["survey"],
        analysis_plan=prior["analysis_plan"],
        coverage_matrix=prior["coverage_matrix"],
    )
    source["writer_document"]["budget_envelope"] = plan["envelope"]
    from litagent.writing_v2 import RANGE_WRITING_PLAN, compile_document

    document = source["writer_document"]
    rendered, source["writing_bindings"] = compile_document(
        document["body"],
        document["summaries"],
        source["fact_cards"],
        document["aliases"],
        "zh-CN",
        source["review_methodology"],
        writing_plan=RANGE_WRITING_PLAN,
        envelope=plan["envelope"],
    )
    assert rendered == source["draft"]
    recorded = {
        r["id"]: r
        for r in old["evaluation"]["faithfulness"]["details"]["partial_samples"]
    }

    class Replay(module.MockGateway):
        async def __call__(self, request):
            reply = await super().__call__(request)
            body = json.loads(request.content)
            payload = json.loads(body["input"][-1]["content"])
            if "jobs" in payload and "context_pool" not in payload:
                wire = reply.json()
                value = {
                    key: {"statements": recorded[key]["statements"]}
                    for key in payload["jobs"]
                }
                wire["output"][0]["content"][0]["text"] = json.dumps(
                    value, ensure_ascii=False
                )
                return httpx.Response(200, json=wire)
            if "source_units" in payload:
                assert payload["assertion_status_contract"] == STATUS_CONTRACT
            return reply

    gateway = Replay()
    monkeypatch.setenv("LLM_API_KEY", "offline-only")
    monkeypatch.setattr(module, "RUN_ID", "vit-status-codes-replay")
    monkeypatch.setattr(
        module,
        "load_inputs",
        lambda: (draft["query"], source, results["extract"], config, plan),
    )
    monkeypatch.setattr(module, "MockGateway", lambda: gateway)
    await module.execute(tmp_path, mock=True)
    report = json.loads(
        (tmp_path / "runs/vit-status-codes-replay-offline.json").read_bytes()
    )["report"]
    details = report["evaluation"]["faithfulness"]["details"]
    assert len(details["assertions"]) == 505
    assert details["unsupported_count"] == 101
    assert details["unverified_count"] == 0
    assert details["stages"]["global_check"] == "completed"
    assert gateway.calls.count("repair_declined") == 1
    assert report["metadata"]["evidence_rewrite_attempts"][0]["attempted"]
    assert not report["delivery"]["publishable"]
    assert report["survey"] == draft["report"]["survey"]
