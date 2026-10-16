"""Verify bounded recovery without forgiving protocol faults or prior spending."""

import copy
import json
import shutil
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest
from test_strict_automatic import observation, suite
from test_strict_quality import draft, to_native

from litagent.agents.extraction_strategy import LLMStrategy, ResilientExtractionStrategy
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.quality_suite import QualitySuiteRunner, export_bundle
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.eval.scoped import check_request_budget
from litagent.evidence import build_evidence_items
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget


@pytest.mark.parametrize(
    "reason", [None, "strict_worker_failed", "strict_usage_unknown"]
)
def test_execution_preserves_real_stop_reason_without_faking_budget_exhaustion(reason):
    result = LitAgent._derive_execution(None, True, False, budget_stop_reason=reason)
    assert result["reason_codes"][0] == (reason or "cost_budget_exceeded")
    assert result["partial"] is True


def test_evaluation_preflight_preserves_protocol_stop_instead_of_budget_error():
    budget = CostBudget(max_tokens=500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    llm = Mock()
    llm._cost_budget = budget
    assert check_request_budget(llm, [], 100, 10000) > 100
    budget.stop_after_worker_failure()
    with pytest.raises(ValueError, match="^strict_worker_failed$"):
        check_request_budget(llm, [], 100, 10000)
    assert budget.used == 0


def paper(policy="strict_auto_v1"):
    """Supply a real locator rather than an invented extraction chunk."""
    return {
        "paper_id": "evit",
        "quality_policy": policy,
        "chunks": [{"chunk_key": "abstract", "text": "EViT improves speed."}],
    }


def extraction():
    """Keep unknown semantic fields explicitly absent."""
    return {
        "claim_records": [
            {
                "text": "EViT improves speed.",
                "source_chunk_key": "abstract",
                "confidence": 0.9,
                "fact_fields": {"method": "EViT", "dataset": None},
            }
        ],
        "metrics": {},
        "methods": ["EViT"],
        "datasets": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["empty", "locator", "confidence", "fields", "duplicate", "methods"]
)
async def test_automatic_extraction_rejects_incomplete_or_invalid_json(fault):
    payload = extraction()
    if fault == "empty":
        payload = {}
    elif fault == "locator":
        payload["claim_records"][0]["source_chunk_key"] = "other-paper"
    elif fault == "confidence":
        payload["claim_records"][0]["confidence"] = True
    elif fault == "fields":
        payload["claim_records"][0]["fact_fields"]["dataset"] = ["guessed"]
    elif fault == "duplicate":
        payload["claim_records"] *= 2
    else:
        payload["methods"] = "EViT"
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills = Mock()
    skills.to_metadata_text_for.return_value = ""
    with pytest.raises(ValueError, match="extraction_contract_invalid"):
        await LLMStrategy(llm, skills).extract(paper())
    assert llm.chat.await_count == 1


@pytest.mark.asyncio
async def test_legitimate_empty_extraction_is_distinct_from_invalid_response():
    payload = extraction()
    payload["claim_records"] = []
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills = Mock()
    skills.to_metadata_text_for.return_value = ""
    result = await LLMStrategy(llm, skills).extract(paper())
    assert result["claim_records"] == result["claims"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["method", "dataset"])
async def test_fact_field_array_failure_is_locatable_and_never_coerced(field):
    payload = extraction()
    payload["claim_records"][0]["fact_fields"][field] = ["first", "second"]
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills = Mock()
    skills.to_metadata_text_for.return_value = ""
    with pytest.raises(
        ValueError, match=f"extraction_contract_invalid:fact_field_type:{field}"
    ):
        await LLMStrategy(llm, skills).extract(paper())
    system = llm.chat.await_args.args[0][0]["content"]
    assert "single string or null" in system and "never an" in system
    assert "one continuous source phrase" in system
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_multiple_dataset_phrase_preserved_with_missing_fields():
    payload = extraction()
    payload["claim_records"][0]["fact_fields"]["dataset"] = "VTAB and ImageNet"
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills = Mock()
    skills.to_metadata_text_for.return_value = ""
    candidate = paper()
    candidate["chunks"][0]["text"] = "EViT improves speed on VTAB and ImageNet."
    result = await LLMStrategy(llm, skills).extract(candidate)
    assert result["claim_records"][0]["fact_fields"]["dataset"] == "VTAB and ImageNet"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_key", ["page:1:block:5", "CHUNK:page:1:block:5", "[CHUNK:page:1:block:5]"]
)
async def test_host_source_marker_preserves_exact_evidence_and_unverified_fact(raw_key):
    candidate = paper()
    candidate["chunks"][0].update(
        chunk_key="page:1:block:5",
        section="abstract",
        content_scope="selected_fulltext",
        page=1,
        block_index=5,
        source_spans=[{"start": 0, "end": 20}],
        raw_content_hash="original-pdf-sha",
    )
    payload = extraction()
    payload["claim_records"][0]["source_chunk_key"] = raw_key
    # Model-supplied audit metadata cannot impersonate host decoding.
    payload["source_key_normalizations"] = [{"forged": True}]
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    output = await LLMStrategy(llm, skills).extract(candidate)
    key = "page:1:block:5"
    assert output["claim_records"][0]["source_chunk_key"] == key
    assert output["source_key_normalizations"] == (
        []
        if raw_key == key
        else [
            {
                "record_index": 0,
                "raw_source_chunk_key": raw_key,
                "source_chunk_key": key,
            }
        ]
    )
    canonical = extraction()
    canonical["claim_records"][0]["source_chunk_key"] = key
    expected = build_evidence_items({**candidate, **canonical})
    actual = build_evidence_items({**candidate, **output})
    assert actual == expected
    assert actual[0]["supporting_text"] == candidate["chunks"][0]["text"]
    cards = build_fact_cards({item["evidence_id"]: item for item in actual})
    expected_cards = build_fact_cards({item["evidence_id"]: item for item in expected})
    assert cards == expected_cards
    card = next(iter(cards.values()))
    assert card["semantic_status"] == "not_verified"
    assert (card["chunk_key"], card["page"], card["block_index"]) == (key, 1, 5)
    assert card["raw_content_hash"] == "original-pdf-sha"
    system = llm.chat.await_args.args[0][0]["content"]
    assert 'Allowed source_chunk_key IDs: ["page:1:block:5"]' in system
    assert 'Example source_chunk_key: "page:1:block:5"' in system
    assert "copy ID only" in system and "match a supplied CHUNK marker" not in system
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_key, reason",
    [
        ("CHUNK:other-paper", "unknown"),
        ("[CHUNK:other-paper]", "unknown"),
        ("CHUNK:CHUNK:abstract", "unknown"),
        ("[CHUNK:abstract", "unknown"),
        (" CHUNK:abstract", "unknown"),
        ("CHUNK:Abstract", "unknown"),
        (None, "type"),
        (["abstract"], "type"),
    ],
)
async def test_source_marker_does_not_guess_unknown_partial_or_non_string_id(
    raw_key, reason
):
    payload = extraction()
    payload["claim_records"][0]["source_chunk_key"] = raw_key
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    with pytest.raises(
        ValueError, match=f"extraction_contract_invalid:source_key_{reason}:record:0"
    ):
        await LLMStrategy(llm, skills).extract(paper())
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_source_marker_collision_rejected_even_when_one_is_exact_bare_key():
    candidate = paper()
    candidate["chunks"].append({"chunk_key": "CHUNK:abstract", "text": "Other source"})
    payload = extraction()
    payload["claim_records"][0]["source_chunk_key"] = "CHUNK:abstract"
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    with pytest.raises(ValueError, match="source_key_ambiguous:record:0"):
        await LLMStrategy(llm, skills).extract(candidate)


@pytest.mark.asyncio
async def test_source_marker_alias_cannot_bypass_duplicate_claim_rejection():
    payload = extraction()
    payload["claim_records"].append(copy.deepcopy(payload["claim_records"][0]))
    payload["claim_records"][1]["source_chunk_key"] = "[CHUNK:abstract]"
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    with pytest.raises(
        ValueError, match="extraction_contract_invalid:duplicate_record"
    ):
        await LLMStrategy(llm, skills).extract(paper())


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["legacy", "strict_v1"])
async def test_historical_source_key_prompt_and_filter_are_unchanged(policy):
    payload = extraction()
    payload["claim_records"][0]["source_chunk_key"] = "CHUNK:abstract"
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    result = await LLMStrategy(llm, skills).extract(paper(policy))
    assert result["claim_records"] == []
    assert "source_key_normalizations" not in result
    assert "match a supplied CHUNK marker" in llm.chat.await_args.args[0][0]["content"]


def test_host_marker_protocol_changes_only_automatic_config_identity():
    from litagent.config import load_config

    config = load_config()
    for policy in ("legacy", "strict_v1"):
        assert (
            "automatic_source_key_protocol"
            not in build_config_summary(config, policy)["effective"]["extractor"]
        )
    summary = build_config_summary(config, "strict_auto_v1")
    assert summary["effective"]["extractor"]["automatic_source_key_protocol"] == (
        "host-source-key-v1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy, expected", [("strict_auto_v1", 185), ("strict_v1", 60), ("legacy", 60)]
)
async def test_extraction_timeout_matches_transport_boundary_without_changing_legacy(
    monkeypatch, policy, expected
):
    seen = []

    async def immediately(awaitable, timeout):
        seen.append(timeout)
        return await awaitable

    monkeypatch.setattr(
        "litagent.agents.extraction_strategy.asyncio.wait_for", immediately
    )
    llm, regex = AsyncMock(), AsyncMock()
    llm.extract.return_value = extraction()
    result = await ResilientExtractionStrategy(llm, regex, 60000).extract(paper(policy))
    assert seen == [expected] and result["extraction_mode"] == "llm"
    regex.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_protocol_failure_has_no_rule_fallback_only_for_auto():
    llm, regex = AsyncMock(), AsyncMock()
    llm.extract.side_effect = ValueError("extraction_contract_invalid")
    regex.extract.return_value = {"claims": []}
    strategy = ResilientExtractionStrategy(llm, regex, 60000)
    with pytest.raises(ValueError, match="extraction_contract_invalid"):
        await strategy.extract(paper())
    regex.extract.assert_not_awaited()
    assert (await strategy.extract(paper("strict_v1")))[
        "extraction_mode"
    ] == "regex_fallback"


@pytest.mark.asyncio
async def test_extractor_stops_queued_paid_requests_before_gather_finishes():
    budget = CostBudget(max_tokens=500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    calls = []

    async def extract(candidate):
        if budget.stop_reason:
            raise ValueError(budget.stop_reason)
        calls.append(candidate["paper_id"])
        raise ValueError("extraction_contract_invalid")

    strategy, loader = AsyncMock(), AsyncMock()
    strategy.extract.side_effect = extract
    candidates = [{**paper(), "paper_id": str(i)} for i in range(3)]
    loader.enrich.return_value = candidates
    worker = ExtractorWorker(
        strategy,
        max_concurrent=1,
        fulltext_loader=loader,
        failure_callback=budget.stop_after_worker_failure,
    )
    task = SubTask(
        "extract",
        "extract",
        "extractor",
        input_data={
            "quality_policy": "strict_auto_v1",
            "upstream_results": {"relevance_gate": candidates},
        },
    )
    with pytest.raises(ValueError, match="extraction_contract_invalid"):
        await worker.execute(task)
    assert calls == ["0"] and budget.stop_reason == "strict_worker_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "enabled", "provider_default"])
async def test_first_structured_generation_consumes_thinking_configuration(mode):
    item = {
        "paper_id": "evit",
        "text": "EViT improves speed.",
        "supporting_text": "EViT improves speed.",
        "chunk_key": "abstract",
    }
    facts = build_fact_cards({"a": item})
    selector = AsyncMock()
    selector.select.return_value = EvidenceSelection(
        candidate_count=1,
        selected_items={"a": item},
        section_evidence_ids={
            key: ["a"]
            for key in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        method_by_section={
            key: "lexical_fallback"
            for key in (
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            )
        },
        omitted_count=0,
        estimated_tokens=1,
    )
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(to_native(draft(facts), facts)), model="offline", usage={}
    )
    worker = SynthesisWorker(llm, evidence_selector=selector, thinking_mode=mode)
    task = SubTask(
        "write",
        "write",
        "synthesis",
        input_data={
            "query": "EViT",
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
            "report_language": "en",
        },
    )
    result = await worker.execute(task)
    assert result["quality_policy"] == "strict_auto_v1"
    options = llm.chat.await_args.kwargs
    assert options["transport_max_attempts"] == 1 and options["max_tokens"] == 20000
    assert options.get("extra_body") == (
        None if mode == "provider_default" else {"thinking": {"type": mode}}
    )
    llm.chat.reset_mock()
    llm.chat.return_value = LLMResponse(content="{}", model="offline", usage={})
    with pytest.raises(ValueError):
        await worker.execute(task)
    assert llm.chat.await_count == 1


def child(parent, executor, suite_id="recovery"):
    """Share the parent's campaign accounting without mutating its files."""
    return QualitySuiteRunner(
        parent.config,
        parent.dataset,
        parent.profile,
        root=parent.root,
        run_archive=parent.archive,
        suite_id=suite_id,
        executor=executor,
        prior_suite_id=parent.suite_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("publishable", [True, False])
async def test_single_round_limit_keeps_full_matrix_and_parent_budget(
    tmp_path, publishable
):
    """A bounded quality trial never implicitly starts another paid observation."""
    parent = suite(
        tmp_path, AsyncMock(return_value=observation(False, technical="auth_failed"))
    )
    await parent.run()
    original = (tmp_path / f"{parent.suite_id}.json").read_bytes()
    executor = AsyncMock(return_value=observation(publishable))
    runner = child(parent, executor)
    preview = await runner.run(dry_run=True, max_new_observations=1)
    assert len(preview["planned"]) == 18 and not preview["observations"]
    executor.assert_not_awaited()
    result = await runner.run(max_new_observations=1)
    assert executor.await_count == 1
    assert result["stop_reasons"] == ["quality_requested_observation_limit"]
    assert result["prior_rounds"] == 1 and len(result["observations"]) == 1
    assert len(result["planned"]) == 18 and result["counts"]["formal"]["planned"] == 15
    assert result["budgets"]["survey"]["charged"] == 20
    assert result["budgets"]["survey"]["requests"] == 2
    assert result["observations"][0]["acceptance"]["publishable"] == publishable
    assert (tmp_path / f"{parent.suite_id}.json").read_bytes() == original
    assert await runner.run(max_new_observations=1) == result
    assert executor.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 19, True, 1.5, "1"])
async def test_invalid_single_round_limit_rejected_before_request(tmp_path, limit):
    runner = suite(tmp_path, AsyncMock())
    with pytest.raises(ValueError, match="quality_observation_limit_invalid"):
        await runner.run(max_new_observations=limit)
    runner.executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_round_unknown_usage_keeps_technical_stop(tmp_path):
    failed = observation(False)
    failed["costs"]["survey"]["usage_complete"] = False
    runner = suite(tmp_path, AsyncMock(return_value=failed))
    result = await runner.run(max_new_observations=1)
    runner.executor.assert_awaited_once()
    assert result["stop_reasons"] == ["quality_usage_unknown"]
    assert result["budgets"]["survey"]["charged"] == 500000


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [None, 1])
async def test_single_round_limit_survives_restart_between_observations(
    tmp_path, requested
):
    """Persisted authorization caps apply even if a later command omits the flag."""
    runner = suite(tmp_path, AsyncMock(return_value=observation()))
    finished = await runner.run(max_new_observations=1)
    assert finished["max_new_observations"] == 1
    finished["status"] = "running"
    finished["stop_reasons"] = []
    runner.save(finished)
    runner.executor.reset_mock()
    result = await runner.run(max_new_observations=requested)
    runner.executor.assert_not_awaited()
    assert result["stop_reasons"] == ["quality_requested_observation_limit"]
    assert len(result["observations"]) == 1


@pytest.mark.asyncio
async def test_single_round_persisted_limit_cannot_be_expanded(tmp_path):
    runner = suite(tmp_path, AsyncMock())
    state = runner.initial()
    state["max_new_observations"] = 1
    runner.save(state)
    with pytest.raises(ValueError, match="quality_observation_limit_conflict"):
        await runner.run(max_new_observations=2)
    runner.executor.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra", [[], ["--max-product-tokens", "1"], ["--max-judge-tokens", "1"]]
)
async def test_quality_cli_forwards_single_round_and_rejects_budget_overrides(
    tmp_path, monkeypatch, extra
):
    """The actual CLI must consume the cap without changing fixed service budgets."""
    from litagent.benchmark import quality_suite
    from litagent.cli import _cmd_benchmark, build_parser

    runner = Mock()
    runner.run = AsyncMock(return_value={"status": "planned"})
    factory = Mock(return_value=runner)
    monkeypatch.setattr(quality_suite, "QualitySuiteRunner", factory)
    args = build_parser().parse_args(
        [
            "benchmark",
            "survey",
            "--stage",
            "quality",
            "--dataset",
            "benchmarks/survey/dataset.yaml",
            "--profiles",
            "benchmarks/survey/profiles.yaml",
            "--judge-config",
            "benchmarks/survey/judge.example.yaml",
            "--output-root",
            str(tmp_path),
            "--dry-run",
            "--max-observations",
            "1",
            *extra,
        ]
    )
    if extra:
        with pytest.raises(SystemExit, match="fixed per-service token budgets"):
            await _cmd_benchmark(args)
        factory.assert_not_called()
    else:
        await _cmd_benchmark(args)
        runner.run.assert_awaited_once_with(dry_run=True, max_new_observations=1)


def test_export_locale_uses_real_counts_for_both_languages(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable for export locale execution")
    result = suite(tmp_path, AsyncMock()).initial()
    from litagent.benchmark.quality_suite import summarize

    result["observations"] = [{**result["planned"][3], **observation()}]
    result["counts"] = summarize(result)
    export_bundle(tmp_path, result)
    script = r"""
const fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(process.argv[1],'utf8');
const scripts=[...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
const nodes=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(x=>[x[1],{textContent:''}]));
nodes.data.textContent=scripts[0];
global.document={documentElement:{lang:'zh-CN'},getElementById:id=>nodes[id]};
vm.runInThisContext(scripts[1]);
nodes.locale.onclick();
if(document.documentElement.lang!=='en'||nodes['formal-summary'].textContent!=='1/15 automatically delivered · 1/15 run · 14 not run')throw Error('incorrect English counts');
nodes.locale.onclick();
if(document.documentElement.lang!=='zh-CN'||nodes['formal-summary'].textContent!=='1/15 自动交付 · 1/15 已运行 · 14 未运行')throw Error('incorrect Chinese counts');
"""
    executed = subprocess.run(
        [node, "-e", script, str(tmp_path / "benchmark.html")],
        text=True,
        capture_output=True,
    )
    assert executed.returncode == 0, executed.stderr


@pytest.mark.asyncio
async def test_continuation_inherits_unknown_charge_requests_and_round_cap(tmp_path):
    failed = observation(False)
    failed["costs"]["survey"].update(usage_complete=False, requests=55)
    parent = suite(tmp_path, AsyncMock(return_value=failed))
    await parent.run()
    original = (tmp_path / f"{parent.suite_id}.json").read_bytes()
    executor = AsyncMock(return_value=observation())
    runner = child(parent, executor)
    preview = await runner.run(dry_run=True)
    assert preview["budgets"]["survey"]["charged"] == 500000
    assert (
        preview["budgets"]["survey"]["requests"] == 55 and preview["prior_rounds"] == 1
    )
    assert not (tmp_path / "continuations").exists()
    result = await runner.run()
    assert executor.await_count == 17
    assert result["stop_reasons"] == ["quality_campaign_round_limit"]
    assert (
        result["counts"]["formal"]["run"] == 14
        and result["counts"]["formal"]["not_run"] == 1
    )
    assert result["budgets"]["survey"]["charged"] == 500170
    assert result["budgets"]["survey"]["requests"] == 72
    assert (tmp_path / f"{parent.suite_id}.json").read_bytes() == original
    assert await runner.run() == result and executor.await_count == 17
    fork = child(parent, AsyncMock(), "fork")
    with pytest.raises(ValueError, match="quality_continuation_already_owned"):
        await fork.run()
    fork.executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_sha_change_is_rejected_before_any_new_request(tmp_path):
    parent = suite(
        tmp_path, AsyncMock(return_value=observation(False, technical="auth_failed"))
    )
    await parent.run()
    runner = child(parent, AsyncMock())
    path = tmp_path / f"{parent.suite_id}.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="quality_parent_identity_changed"):
        await runner.run()
    runner.executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_run_id_must_match_requested_file_identity(tmp_path):
    parent = suite(
        tmp_path, AsyncMock(return_value=observation(False, technical="auth_failed"))
    )
    result = await parent.run()
    result = copy.deepcopy(result)
    result["run_id"] = "different"
    (tmp_path / f"{parent.suite_id}.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="quality_parent_not_terminal_or_incompatible"):
        child(parent, AsyncMock())
