"""Verify the scoped threshold, real plan binding and observation ledger."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.benchmark.datasets import load_survey_dataset
from litagent.benchmark.survey_latency import (
    CASES,
    PROFILES,
    llm_outcomes,
    publish,
    render_benchmark_page,
    run_slots,
    save_observation,
    stage_spans,
    summary_rows,
)
from litagent.config import load_config
from litagent.eval import conflict_review, ragas_native
from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
from litagent.eval.results import (
    BENCHMARK_FAITHFULNESS_POLICY,
    BENCHMARK_IDENTITY_FIELD,
    BENCHMARK_POLICY_FIELD,
    CONFLICT_ONLY_IDENTITY,
    EvaluationRunResult,
    acceptance,
    contract_identity,
    derive_quality,
    execute_evaluation,
    result,
)
from litagent.output_capacity import configured_output_tokens
from litagent.run_limits import configured_run_limits
from litagent.runner import LitAgent, RunPolicy
from litagent.safety import planned_budget
from litagent.safety.planned_accounting import validate_accounting_snapshot

CONFIG = "config/agent-format-repair.yaml"


def completed_details(supported=19, total=20, pending=False, conflict=False):
    """Build exact support accounting for the normal result consumer."""
    claims = []
    for index in range(total):
        status = "supported" if index < supported else "unsupported"
        if pending and index == total - 1:
            status = "unverifiable"
        claims.append(
            {
                "claim_id": f"claim_{index}",
                "kind": "scientific",
                "status": status,
                "atomic": True,
                "classification_valid": True,
                "source_faithful": True,
                "source_quote": f"Scientific assertion {index}.",
            }
        )
    return {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "quality_policy": "strict_auto_v2",
        "source_report_sha256": "sha",
        "threshold": 0.8,
        "ragas_score": supported / total,
        "assertions": claims,
        "fact_card_map": {"card": {}},
        "conflicts": [{"claim_a": "claim_0", "claim_b": "claim_1"}] if conflict else [],
        "inventory_check": {"status": "passed"},
        "coverage": {"source_units": total, "processed_units": total},
        "stages": {
            "global_scope": conflict_review.CONTRACT,
            "global_check": "completed",
            "global_sections": {"conflicts": {"status": "passed"}},
            "expected_assertions": total,
            "processed_assertions": total,
            "expected_batches": 1,
            "batches": [
                {"status": "completed", "claim_ids": [c["claim_id"] for c in claims]}
            ],
        },
    }


@pytest.mark.parametrize(
    "supported,total,pending,conflict,citation,expected",
    [
        (19, 20, False, False, True, "passed"),
        (18, 19, False, False, True, "failed"),
        (19, 20, True, False, True, "unverified"),
        (19, 20, False, True, True, "failed"),
        (19, 20, False, False, False, "failed"),
    ],
)
def test_experimental_gate_keeps_evidence_and_independent_checks(
    supported, total, pending, conflict, citation, expected
):
    details = {
        **completed_details(supported, total, pending, conflict),
        BENCHMARK_POLICY_FIELD: BENCHMARK_FAITHFULNESS_POLICY,
    }
    run = EvaluationRunResult.from_details(details, "sha")
    evaluation = {
        "citation_accuracy": result(
            "passed" if citation else "failed", score=float(citation)
        ),
        **run.metrics(),
    }
    quality = derive_quality(evaluation)
    assert quality["status"] == expected
    assert quality["strict_acceptance"]["science_required_ratio"] == 0.95
    assert evaluation["faithfulness"]["score"] == (
        None if pending else supported / total
    )
    assert evaluation["faithfulness"]["details"]["threshold"] == 0.95
    assert evaluation["faithfulness"]["details"]["assertions"][-1]["source_quote"]
    if pending:
        assert quality["strict_acceptance"]["status"] == "failed"


def test_benchmark_policy_is_isolated_and_historical_results_keep_original_identity():
    plain = load_config(CONFIG)
    assert BENCHMARK_POLICY_FIELD not in plain.eval.model_dump()
    old = EvaluationRunResult.from_details(completed_details(), "sha")
    assert old.details["evaluation_result_identity"] == CONFLICT_ONLY_IDENTITY
    assert old.metrics()["faithfulness"]["passed"]
    assert acceptance(old.details)["science_required_ratio"] == 1
    assert acceptance(old.details)["status"] == "failed"
    plain.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    with pytest.raises(ValueError, match="requires_isolated_native_run"):
        LitAgent(plain)
    assert LitAgent(plain, run_policy=RunPolicy.isolated_benchmark())


@pytest.mark.asyncio
async def test_actual_budget_envelope_selects_the_same_policy_for_every_evaluation(
    case, monkeypatch
):
    cfg = load_config(CONFIG)
    cfg.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    plan = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=configured_engine(cfg),
        capacity=cfg.capacity,
        methodology={},
        output_tokens=configured_output_tokens(cfg),
        run_limits=configured_run_limits(cfg),
        benchmark_faithfulness_policy=BENCHMARK_FAITHFULNESS_POLICY,
    )
    assert plan["status"] == "admitted"
    envelope = planned_budget.validate_writer_envelope(plan["envelope"])
    assert envelope[BENCHMARK_POLICY_FIELD] == BENCHMARK_FAITHFULNESS_POLICY
    assert envelope[BENCHMARK_IDENTITY_FIELD] == contract_identity(
        conflict_review.CONTRACT, BENCHMARK_FAITHFULNESS_POLICY
    )
    assert envelope[BENCHMARK_IDENTITY_FIELD] != CONFLICT_ONLY_IDENTITY
    accounting = planned_budget.PlannedBudget(plan).snapshot()
    assert validate_accounting_snapshot(accounting, expected_plan=plan) == accounting
    monkeypatch.setattr(planned_budget, "locked_writer_envelope", lambda _: envelope)

    async def native(**kwargs):
        details = completed_details()
        clean = EvaluationRunResult.from_details(details, "sha")
        return ragas_native.NativeEvaluationRunResult("sha", clean.details)

    monkeypatch.setattr(ragas_native, "execute_native", native)
    for phase in ("initial", "post_rewrite"):
        run = await execute_evaluation(
            llm=SimpleNamespace(),
            survey="text",
            context={"phase": phase},
            config=cfg.eval,
        )
        assert run.metrics()["faithfulness"]["score"] == 0.95
        assert acceptance(run.details)["status"] == "passed"
    with pytest.raises(ValueError, match="policy_mismatch"):
        await execute_evaluation(
            llm=SimpleNamespace(),
            survey="text",
            context={},
            config=load_config(CONFIG).eval,
        )


def timed(event, milliseconds, task=None, **data):
    at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=milliseconds
    )
    if task is not None:
        data["task_id"] = task
    return {"event": event, "at": at.isoformat(), "data": data}


def test_stage_spans_measure_parallel_wall_time_and_each_reassessment():
    events = [
        timed("worker.start", 0, "search_a"),
        timed("worker.start", 40, "recall_b"),
        timed("worker.complete", 100, "search_a"),
        timed("worker.complete", 150, "recall_b"),
        timed("fulltext.start", 150, "extract", operation_id="ft"),
        timed("fulltext.failed", 200, "extract", operation_id="ft"),
        timed("subspan.start", 240, "evaluation.post_rewrite"),
        timed("subspan.end", 310, "evaluation.post_rewrite"),
        timed("subspan.start", 460, "evaluation.post_rewrite"),
        timed("subspan.end", 500, "evaluation.post_rewrite"),
        timed("subspan.start", 510, "evidence_rewrite"),
        timed("subspan.end", 509, "evidence_rewrite"),
    ]
    assert stage_spans(events) == (
        {"搜索／召回": 150, "全文获取": 50, "evaluation.post_rewrite": [70, 40]},
        ["evidence_rewrite"],
    )


def test_rewrite_wall_clock_uses_evaluation_boundaries_and_actual_repair_calls():
    events = [
        timed("subspan.start", 1, "evaluation.initial"),
        timed("subspan.end", 10, "evaluation.initial"),
        timed("llm.start", 11, "evidence_rewrite"),
        timed("subspan.start", 30, "evaluation.post_rewrite"),
        timed("subspan.end", 40, "evaluation.post_rewrite"),
        timed("subspan.start", 80, "evaluation.post_rewrite"),
        timed("subspan.end", 90, "evaluation.post_rewrite"),
    ]
    assert stage_spans(events)[0]["修订（含准备）"] == [20]


def test_model_failures_count_http_rejections_dispatched_as_completion():
    events = [
        timed("llm.complete", 1, "extract", provider_http_status=200),
        timed(
            "llm.complete",
            2,
            "summary",
            provider_http_status=403,
            usage_available=False,
        ),
        timed("llm.failed", 3, "extract", provider_http_status=403),
        timed("llm.retry", 4, "extract"),
    ]
    assert llm_outcomes(events) == {"llm_failures": 2, "llm_retries": 1}


def test_full_matrix_and_one_table_link_the_exact_snapshots(tmp_path):
    slots = run_slots()
    assert len(slots) == 21
    assert [slot.case_id for slot in slots[:3]] == list(CASES)
    assert {slot.profile_id for slot in slots[3:6]} == set(PROFILES)
    assert [slot.profile_id for slot in slots[3:6]] == [
        slot.profile_id for slot in reversed(slots[12:15])
    ]
    slot = slots[3]
    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    source = tmp_path / "artifact.json"
    raw = {
        "run_id": slot.key,
        "status": "completed",
        "elapsed_ms": 250,
        "report": {
            "survey": "Report.",
            "evaluation": {"faithfulness": {"score": 0.95, "status": "passed"}},
            "delivery": {"status": "ready"},
        },
        "events": [
            timed("external_inputs_ready", 10, fingerprint="frozen"),
            timed(
                "subspan.end", 200, "evaluation.initial", output={"faithfulness": 0.95}
            ),
            timed("survey.complete", 210),
            timed(
                "benchmark.accounting",
                230,
                known_tokens=2500,
                requests=5,
                usage_uncertain=False,
            ),
        ],
    }
    source.write_text(json.dumps(raw))
    observed = save_observation(
        slot,
        dataset.case_by_id(slot.case_id),
        source,
        tmp_path,
        set(dataset.corpus_paper_ids),
        snapshot="frozen",
    )
    assert observed["downstream_ms"] == 200
    assert observed["usage"]["paid_requests"] == 5
    assert observed["usage"]["llm_failures"] == 0
    assert observed["usage"]["llm_retries"] == 0
    assert observed["versions"][0]["task"] == "evaluation.initial"
    publish(tmp_path, slots, {slot.key: observed}, "partial", "stop_reason")
    result_rows = json.loads((tmp_path / "benchmark.json").read_text())["rows"]
    assert len(result_rows) == 21
    assert result_rows[3]["evaluation"]["faithfulness"]["score"] == 0.95
    assert (tmp_path / observed["snapshot"]).read_bytes() == source.read_bytes()
    lines = (tmp_path / "benchmark.md").read_text().splitlines()
    matched = next(line for line in lines if "archive%3A" in line)
    assert "95.00%" in matched and "0.2" in matched
    assert observed["snapshot"] not in matched

    observed_bad = save_observation(
        slot,
        dataset.case_by_id(slot.case_id),
        source,
        tmp_path,
        set(dataset.corpus_paper_ids),
        snapshot="different",
    )
    assert observed_bad["downstream_ms"] is None
    assert "external_input_fingerprint_mismatch" in observed_bad["reasons"]


def test_benchmark_page_shows_real_workbench_links_and_offline_delivery_separately():
    row = {
        "run_id": "actual-run",
        "snapshot": "snapshots/private/run.json",
        "case_id": "few-shot-metric",
        "profile_id": "external-abstract",
        "source": "workbench",
        "repetition": 1,
        "status": "completed",
        "delivery": "blocked",
        "evaluation": {"faithfulness": {"status": "passed", "score": 0.95}},
        "browser_ms": {"final_report": 3100, "terminal": 3000},
    }
    diagnostic = {
        **row,
        "run_id": "diagnostic",
        "source": "workbench-diagnostic",
        "evaluation": {"faithfulness": {"status": "unverified", "score": 0.0}},
        "reasons": ["<script>alert(1)</script>"],
    }
    retry = {
        **row,
        "run_id": "separate-retry",
        "source": "retry",
        "delivery": "ready",
        "retry_of": {"run_id": "actual-run"},
        "downstream_ms": 6400,
    }
    page = render_benchmark_page(
        [row, diagnostic, retry],
        status="completed",
        revalidated={
            "revalidated_ready": 1,
            "rows": [
                {
                    "run_id": "actual-run",
                    "recorded_status": "blocked",
                    "revalidated_status": "ready",
                }
            ],
        },
    )
    assert "原始交付 0/1 · 同批归档离线复核 1/1" in page
    assert "计入真实复跑交付 1/1 · 修复复跑 1 次 · 诊断复验 1 次" in page
    assert "离线复核通过" in page and "95.00%" in page and "0.00%" not in page
    assert "archive%3Aactual-run&amp;view=execution" in page
    assert "archive%3Adiagnostic&amp;view=execution" in page
    assert "archive%3Aseparate-retry&amp;view=execution" in page and "6.4" in page
    assert "snapshots/private" not in page and "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_summary_uses_raw_repetitions_and_same_input_pairs():
    first = {
        "source": "replay",
        "case_id": CASES[0],
        "repetition": 1,
        "profile_id": PROFILES[0],
        "external_inputs": "shared",
        "group": "ready",
        "downstream_ms": 500,
    }
    second = {
        **first,
        "profile_id": PROFILES[1],
        "downstream_ms": 700,
        "group": "blocked",
    }
    wrong_input = {**first, "profile_id": PROFILES[2], "external_inputs": "other"}
    summary = summary_rows([first, second, wrong_input])
    pairs = [row for row in summary if row["source"] == "paired"]
    assert len(pairs) == 1
    assert pairs[0]["group"] == "ready → blocked"
    assert pairs[0]["stats"]["delta_ms"]["p50"] == 200
    assert pairs[0]["stats"]["delta_ms"]["by_case"] == {f"{CASES[0]}/r1": 200}
    many = [{**first, "repetition": i, "downstream_ms": i} for i in range(1, 41)]
    assert "p95" not in summary_rows(many[:39])[0]["stats"]["downstream"]
    assert summary_rows(many)[0]["stats"]["downstream"]["p95"] == pytest.approx(38.05)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop", [None, "budget", "unknown", "approved_scope", "missing_archive"]
)
async def test_campaign_reuses_one_actual_snapshot_per_topic(
    tmp_path, monkeypatch, stop
):
    """Drive the matrix through the real recorder projection with offline IO."""
    from contextlib import asynccontextmanager

    from litagent.benchmark.datasets import load_survey_profiles
    from scripts import run_survey_latency as campaign

    config = load_config(CONFIG)
    config.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    config.rag.raw_root = str(tmp_path / "seed")
    root = tmp_path / "results"
    archive = None
    dataset = load_survey_dataset(campaign.DATASET)
    profiles = load_survey_profiles(campaign.PROFILES_PATH)
    slots = run_slots()
    browser_starts = []
    replayed = []

    @asynccontextmanager
    async def fake_workbench(config, root, owner):
        nonlocal archive
        archive = owner
        yield "http://localhost", SimpleNamespace(config=config)

    async def no_index(runner, dataset, profiles):
        if stop == "approved_scope":
            assert [profile.profile_id for profile in profiles] == ["external-abstract"]

    async def browser(chrome, base, user_dir, query):
        run_id = f"{len(browser_starts):012x}"
        browser_starts.append(query)
        if stop == "missing_archive":
            return {
                "runId": run_id,
                "markers": {"first_progress": 10, "terminal": 50},
                "error": "final_report_unavailable",
            }
        artifact = {
            "run_id": run_id,
            "status": "completed",
            "elapsed_ms": 50,
            "report": {"survey": "Report.", "delivery": {"status": "ready"}},
            "events": [
                timed(
                    "benchmark.accounting",
                    50,
                    known_tokens=2,
                    requests=1,
                    usage_uncertain=stop == "unknown",
                )
            ],
            "nodes": {
                "subspan:query_decomposition": {
                    "task_id": "query_decomposition",
                    "output": {"sub_queries": [query]},
                },
                "worker:search_arxiv_q0": {
                    "task_id": "search_arxiv_q0",
                    "input": {"mode": "external", "source": "arxiv", "query": query},
                    "output": [],
                },
            },
        }
        if stop == "unknown":
            artifact["usage"] = {"total_tokens": 0}
            artifact["events"].insert(
                0,
                timed(
                    "llm.failed",
                    30,
                    "extract",
                    provider_request_id="test-request",
                    usage_available=False,
                ),
            )
        archive.save(artifact)
        return {"runId": run_id, "markers": {"first_progress": 10, "terminal": 50}}

    async def replay(config, case, root, owner, snapshot, *, run_id):
        slot = next(
            slot
            for slot in slots
            if run_id == slot.key or run_id.startswith(f"{slot.key}-retry-")
        )
        assert snapshot.query == case.query
        assert config.rag.enabled == (slot.profile_id != "external-only")
        replayed.append((slot, snapshot.fingerprint))
        return owner.save(
            {
                "run_id": run_id,
                "status": "completed",
                "elapsed_ms": 35,
                "report": {"survey": "Report.", "delivery": {"status": "ready"}},
                "events": [
                    timed("external_inputs_ready", 1, fingerprint=snapshot.fingerprint),
                    timed(
                        "benchmark.external_inputs.replayed",
                        2,
                        component="planner",
                        snapshot_fingerprint=snapshot.fingerprint,
                    ),
                    timed(
                        "benchmark.external_inputs.replayed",
                        3,
                        "search_arxiv_q0",
                        component="search",
                        snapshot_fingerprint=snapshot.fingerprint,
                    ),
                    timed("survey.complete", 29),
                    timed(
                        "benchmark.accounting",
                        30,
                        known_tokens=2,
                        requests=1,
                        usage_uncertain=False,
                    ),
                ],
            }
        )

    monkeypatch.setattr(campaign.SurveyBenchmarkRunner, "_prepare_corpora", no_index)
    monkeypatch.setattr(campaign, "workbench", fake_workbench)
    monkeypatch.setattr(campaign, "browser_run", browser)
    monkeypatch.setattr(campaign, "replay_run", replay)
    manifest = {
        "raw_pdf_cache_sha256": campaign.cache_identity(Path(config.rag.raw_root)),
        "per_run_tokens": 10,
        "per_run_requests": 4,
        "max_tokens": 12 if stop == "budget" else 210,
        "max_requests": 84,
    }
    if stop == "approved_scope":
        manifest["max_runs"] = 1
    await campaign.execute(root, config, dataset, profiles, slots, manifest, "unused")
    blob = json.loads((root / "benchmark.json").read_bytes())
    assert len(blob["rows"]) == 21
    if stop:
        count = 2 if stop == "budget" else 1
        reason = {
            "budget": "benchmark_budget_exhausted",
            "unknown": "benchmark_usage_unknown",
            "approved_scope": "outside_approved_run_scope",
            "missing_archive": "benchmark_usage_unknown",
        }[stop]
        assert len(browser_starts) == count
        assert not replayed
        assert blob["status"] == "partial"
        assert all(
            row["status"] == "未运行" and row["reasons"] == [reason]
            for row in blob["rows"][count:]
        )
        if stop == "missing_archive":
            row = blob["rows"][0]
            assert row["run_id"] == "000000000000"
            assert row["browser_ms"] == {"first_progress": 10, "terminal": 50}
            assert row["reasons"] == [
                "run_artifact_missing",
                "final_report_unavailable",
            ]
            assert [(s["group"], s["n"]) for s in blob["summaries"]] == [("failed", 1)]
            assert blob["summaries"][0]["stats"]["terminal"]["p50"] == 50
        if stop in {"approved_scope", "unknown"}:
            source = {
                "contract": campaign.CONTRACT,
                "per_run_tokens": 10,
                "per_run_requests": 4,
                "max_tokens": 210,
                "max_requests": 84,
            }
            source["approval_sha256"] = campaign.sha(source)
            campaign.write_json(root / "manifest.json", source)
            campaign.write_json(
                root / "started.json", {"approval_sha256": source["approval_sha256"]}
            )
        if stop == "unknown":
            with pytest.raises(ValueError, match="usage_receipt_required"):
                campaign.read_resumed_campaign(root, slots, dataset)
            carried = campaign.read_resumed_campaign(
                root, slots, dataset, carry_unsettled=True
            )
            usage = carried.observations[slots[0].key]["usage"]
            assert (
                usage["held_run_token_cap"] == 10 and usage["held_run_request_cap"] == 4
            )
            assert usage["billed_tokens"] is None and usage["usage_uncertain"] is True
            assert campaign.remaining_approved_budget(carried) == {
                "prior_tokens_spent_or_reserved": 10,
                "prior_requests_sent_or_reserved": 4,
                "remaining_tokens": 200,
                "remaining_requests": 80,
            }
            result = {
                "ResponseMetadata": {
                    "Action": "GetInferenceRequestUsage",
                    "Service": "ark",
                },
                "Result": {"Records": []},
            }
            with pytest.raises(ValueError, match="usage_receipt_invalid"):
                campaign.read_resumed_campaign(
                    root, slots, dataset, json.dumps(result).encode()
                )
            record = {"RequestID": "other", "InputTokens": 3, "OutputTokens": 4}
            result["Result"]["Records"] = [record]
            with pytest.raises(ValueError, match="usage_receipt_invalid"):
                campaign.read_resumed_campaign(
                    root, slots, dataset, json.dumps(result).encode()
                )
            record["RequestID"] = "test-request"
            receipt = json.dumps(result).encode()
            resumed = campaign.read_resumed_campaign(root, slots, dataset, receipt)
            row = resumed.observations[slots[0].key]
            assert row["usage"]["usage_uncertain"] is True
            assert row["usage"]["billed_tokens"] is None
            assert row["usage"]["reconciled_billed_tokens"] == 7
            destination = tmp_path / "continued"
            resumed.materialize(destination)
            assert (destination / row["usage_receipt"]["path"]).read_bytes() == receipt
            assert (destination / row["snapshot"]).read_bytes() == (
                root / row["snapshot"]
            ).read_bytes()
            campaign.write_json(destination / "manifest.json", source)
            campaign.write_json(
                destination / "started.json",
                {"approval_sha256": source["approval_sha256"]},
            )
            publish(destination, slots, resumed.observations, "partial", "pending")
            assert "7（已对账）" in (destination / "benchmark.md").read_text()
            assert (
                campaign.read_resumed_campaign(destination, slots, dataset).observations
                == resumed.observations
            )
            (destination / row["usage_receipt"]["path"]).write_bytes(receipt + b"\n")
            with pytest.raises(ValueError, match="usage_receipt_mismatch"):
                campaign.read_resumed_campaign(destination, slots, dataset)
        if stop == "approved_scope":
            resumed = campaign.read_resumed_campaign(root, slots, dataset)
            destination = tmp_path / "continued"
            resumed.materialize(destination)
            await campaign.execute(
                destination,
                config,
                dataset,
                profiles,
                slots,
                manifest,
                "unused",
                resumed,
            )
            continued = json.loads((destination / "benchmark.json").read_bytes())
            assert continued["rows"][0] == blob["rows"][0]
            assert len(browser_starts) == 2
            assert continued["rows"][1]["case_id"] == "vit-foundations"
            assert continued["rows"][1]["status"] == "completed"
            assert continued["rows"][2]["status"] == "未运行"
        return
    assert len(browser_starts) == 3
    assert len(replayed) == 18
    assert blob["status"] == "completed"
    for case_id in CASES:
        assert (
            len(
                {
                    fingerprint
                    for slot, fingerprint in replayed
                    if slot.case_id == case_id
                }
            )
            == 1
        )
        assert len([row for row in blob["rows"] if row["case_id"] == case_id]) == 7
    assert all(
        row["downstream_ms"] == 28 for row in blob["rows"] if row["source"] == "replay"
    )
    assert all((root / row["snapshot"]).is_file() for row in blob["rows"])
    selected_slot = slots[3]
    observations = {
        slot.key: row for slot, row in zip(slots, blob["rows"], strict=True)
    }
    original = observations[selected_slot.key]
    observations[selected_slot.key] = {**original, "delivery": "blocked"}
    snapshots = {
        case_id: campaign.ExternalInputSnapshot.model_validate_json(
            (root / "inputs" / f"{case_id}.json").read_bytes()
        )
        for case_id in CASES
    }
    resumed = campaign.ResumedCampaign(
        manifest, "test-baseline", observations, snapshots, {}
    ).select_failed_replays([selected_slot.key])
    destination = tmp_path / "retry"
    await campaign.execute(
        destination,
        config,
        dataset,
        profiles,
        slots,
        {**manifest, "max_runs": 1, "retry_of": resumed.retry_of},
        "unused",
        resumed=resumed,
    )
    retried = json.loads((destination / "benchmark.json").read_bytes())["rows"][3]
    assert retried["run_id"].startswith(f"{original['run_id']}-retry-")
    assert retried["retry_of"] == {
        "run_id": original["run_id"],
        "snapshot_sha256": original["snapshot_sha256"],
    }
    assert retried["external_inputs"] == original["external_inputs"]
    assert (destination / "runs" / f"{retried['run_id']}.json").is_file()
    assert json.loads((root / "benchmark.json").read_bytes()) == blob

    async def cancelled_replay(*args, **kwargs):
        path = await replay(*args, **kwargs)
        artifact = json.loads(path.read_bytes())
        artifact.update(status="failed", error="CancelledError", report=None)
        path.write_text(json.dumps(artifact))
        raise asyncio.CancelledError

    monkeypatch.setattr(campaign, "replay_run", cancelled_replay)
    interrupted = tmp_path / "interrupted-retry"
    with pytest.raises(asyncio.CancelledError):
        await campaign.execute(
            interrupted,
            config,
            dataset,
            profiles,
            slots,
            {**manifest, "max_runs": 1, "retry_of": resumed.retry_of},
            "unused",
            resumed=resumed,
        )
    terminal = json.loads((interrupted / "benchmark.json").read_bytes())
    cancelled = terminal["rows"][3]
    assert terminal["status"] == "partial" and cancelled["status"] == "cancelled"
    assert cancelled["run_id"].startswith(f"{original['run_id']}-retry-")
    assert cancelled["retry_of"] == retried["retry_of"]
    assert (interrupted / cancelled["snapshot"]).is_file()


def test_retry_uses_original_inputs_and_only_schedules_settled_failed_replays():
    from scripts.run_survey_latency import ResumedCampaign

    def row(source="replay", delivery="blocked", unknown=False):
        return {
            "source": source,
            "delivery": delivery,
            "run_id": "archive-run",
            "snapshot_sha256": "a" * 64,
            "usage": {"usage_uncertain": unknown},
        }

    baseline = ResumedCampaign(
        {},
        "original",
        {"failed": row(), "ready": row(delivery="ready"), "unknown": row(unknown=True)},
        {"few-shot": object()},
        {"snapshots/failed/run.json": b"baseline", "inputs/few-shot.json": b"inputs"},
    )
    selected = baseline.select_failed_replays(["failed"])
    assert selected.retry_slots == ("failed",)
    assert selected.retry_of == {
        "failed": {"run_id": "archive-run", "snapshot_sha256": "a" * 64}
    }
    assert set(selected.observations) == {"ready", "unknown"}
    assert selected.files == {"inputs/few-shot.json": b"inputs"}
    assert selected.snapshots is baseline.snapshots
    assert baseline.files["snapshots/failed/run.json"] == b"baseline"
    for bad in (["ready"], ["unknown"], ["failed", "failed"]):
        with pytest.raises(ValueError, match="survey_retry_replay_slot_invalid"):
            baseline.select_failed_replays(bad)


def test_timed_out_partial_report_keeps_missing_scores_and_uncertain_usage(tmp_path):
    """A reserved amount and a skipped score are not measured outcomes."""
    slot = run_slots()[0]
    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    source = tmp_path / "artifact.json"
    source.write_text(
        json.dumps(
            {
                "run_id": "timed-out",
                "status": "completed",
                "usage": {"total_tokens": 97858},
                "report": {
                    "survey": "Survey incomplete.",
                    "partial": True,
                    "delivery": {"status": "blocked"},
                    "evaluation": {"faithfulness": {"score": 0.0, "skipped": True}},
                },
                "trace": {
                    "trace_id": "abc",
                    "trace_url": "http://localhost:3000/traces/abc",
                },
                "events": [
                    timed("worker.start", 1, "extract"),
                    timed("llm.retry", 100, "extract"),
                    timed("llm.failed", 216000, "extract", usage_available=False),
                    timed(
                        "worker.failed",
                        216001,
                        "extract",
                        error_type="LLMRequestTimeout",
                    ),
                    timed(
                        "benchmark.accounting",
                        216010,
                        known_tokens=151401,
                        requests=27,
                        usage_uncertain=True,
                    ),
                ],
            }
        )
    )
    observed = save_observation(
        slot,
        dataset.case_by_id(slot.case_id),
        source,
        tmp_path,
        set(dataset.corpus_paper_ids),
    )
    assert observed["group"] == "timeout"
    assert "extract_request_timeout" in observed["reasons"]
    assert observed["evaluation"]["faithfulness"]["score"] is None
    assert observed["usage"]["billed_tokens"] is None
    assert observed["usage"]["budget_accounted_tokens"] == 151401
    assert observed["usage"]["llm_failures"] == 1
    assert observed["usage"]["llm_retries"] == 1
    publish(tmp_path, [slot], {slot.key: observed}, "partial", "budget_usage_unknown")
    rendered = (tmp_path / "benchmark.md").read_text()
    assert "文献抽取超时" in rendered and "97,858（部分）" in rendered
    assert "archive%3Atimed-out" in rendered and "0.00% (" not in rendered
    assert "http://localhost:3000/traces/abc" not in rendered


@pytest.mark.asyncio
async def test_benchmark_sse_retains_run_isolated_progress_draft_and_accounting(
    tmp_path, monkeypatch
):
    """Exercise the regular endpoint and recorder with the model run replaced."""
    from fastapi.testclient import TestClient

    from litagent import api
    from litagent.contracts import build_config_summary
    from litagent.observability.recorder import ArchiveRepository
    from scripts.run_survey_latency import options_for

    cfg = load_config(CONFIG)
    cfg.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    archive = ArchiveRepository(tmp_path)
    monkeypatch.setattr(api.app.state, "config", cfg)
    monkeypatch.setattr(api.app.state, "tasks", {})
    monkeypatch.setattr(api.app.state, "flow_repository", archive)
    monkeypatch.setattr(api.app.state, "survey_semaphore", None)

    class Agent:
        def __init__(self, config, trace_hook, run_policy):
            assert run_policy == RunPolicy.isolated_benchmark()
            self.emit = trace_hook
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def benchmark_accounting(self):
            return {"known_tokens": 7, "requests": 1, "usage_uncertain": False}

        async def run(self, query, cancellation, options):
            self.emit("worker.start", {"task_id": "search_a"})
            self.emit("worker.complete", {"task_id": "search_a", "output": []})
            self.emit("subspan.start", {"task_id": "adversarial_review:synthesis:r1"})
            self.emit(
                "subspan.end",
                {
                    "task_id": "adversarial_review:synthesis:r1",
                    "output": {"draft": "## Draft"},
                },
            )
            self.emit("survey.complete", {"quality_status": "passed"})
            return {
                "survey": "## Final",
                "config_fingerprint": build_config_summary(
                    self.config, options.quality_policy
                )["fingerprint"],
                "delivery": {"status": "ready"},
            }

    monkeypatch.setattr(api, "LitAgent", Agent)
    monkeypatch.setattr(cfg.observability, "enabled", False)
    # Keep the production ASGI event loop alive across the submit and SSE calls.
    from litagent.interaction_store import SqliteReviewRepository

    monkeypatch.setattr(api.app.state, "latency_benchmark_config", cfg, raising=False)
    monkeypatch.setattr(
        api.app.state,
        "interactive_repository",
        SqliteReviewRepository(tmp_path / "interactive"),
    )
    with TestClient(api.app) as client:
        payload = options_for(cfg).model_dump(mode="json")
        response = client.post("/survey", json={"query": "question", **payload})
        assert response.status_code == 202, response.text
        run_id = response.json()["task_id"]
        stream = client.get(f"/survey/{run_id}/events")
        assert "event: done" in stream.text
    artifact = archive.get(run_id)
    assert artifact["run_id"] == run_id
    assert artifact["status"] == "completed"
    trace = {event["event"]: event["data"] for event in artifact["events"]}
    assert trace["subspan.end"]["output"]["draft"] == "## Draft"
    assert trace["benchmark.accounting"]["known_tokens"] == 7
