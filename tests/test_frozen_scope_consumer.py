"""Exercise corrected frozen scopes through the real CLI/runner transport boundary."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_frozen_quality_repair import factories
from test_frozen_repair_parent import parent
from test_reference_occurrence_policy import bound
from test_shared_evidence_scope import frozen, ledger

from litagent.config import load_config
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.observability.context import get_task_id
from litagent.runner import LitAgent
from scripts.run_frozen_quality_repair import prepare, run_trial


def source_with_shared_scope(tmp_path, config):
    """Extend a synthetic charged parent with a real frozen ranking producer."""
    source, root = parent(tmp_path)
    artifact = json.loads(source.read_text())
    scope = frozen()
    body = (
        "# Review\n\n## Representative Methods\n\nThe variant maps coordinates [E:b]."
    )
    artifact["query"] = "network"
    artifact["report"]["survey"] = body
    artifact["report"]["evaluation"] = bound(body)
    artifact["report"]["metadata"].update(scope["report"]["metadata"], query="network")
    artifact["nodes"] = scope["nodes"]
    artifact["nodes"]["extract"] = {
        "kind": "worker",
        "task_id": "extract",
        "output": [
            {
                "paper_id": item["paper_id"],
                "claims": [item["text"]],
                "evidence_items": [item],
            }
            for item in ledger().values()
        ],
    }
    artifact["events"].extend(scope["events"])
    source.write_text(json.dumps(artifact))
    plan = prepare(source, config, "parent-trial")["plan"]
    state = json.loads((root / "checkpoint.json").read_text())
    state.update(plan=plan, binding=plan["binding"])
    state["report"]["survey"] = body
    record_path = root / "runs" / "parent-trial.json"
    record = json.loads(record_path.read_text())
    record.update(query="network", report=copy.deepcopy(state["report"]))
    (root / "checkpoint.json").write_text(json.dumps(state))
    (root / "input-snapshot.json").write_text(json.dumps(artifact))
    record_path.write_text(json.dumps(record))
    return source, root


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_patch", [False, True])
async def test_real_frozen_cli_consumes_shared_method_permissions_without_paid_calls(
    tmp_path,
    monkeypatch,
    valid_patch,
):
    config = load_config()
    config.context.evidence_top_k_per_section = 1
    source, parent_root = source_with_shared_scope(tmp_path, config)
    source_before = source.read_bytes()
    parent_before = {str(p): p.read_bytes() for p in parent_root.rglob("*.json")}
    run_id = "offline-cli-scope"
    prepared = prepare(source, config, run_id, parent_root=parent_root)
    cards = prepared["artifact"]["report"]["metadata"]["fact_cards"]
    foundation = next(cid for cid, c in cards.items() if c["evidence_id"] == "a")
    targets = LitAgent._strict_repair_targets(
        prepared["artifact"]["report"]["evaluation"],
        prepared["artifact"]["report"]["survey"],
        "academic_v2",
        "en",
    )
    content = (
        json.dumps(
            {
                "patches": {
                    targets[0]["source_fragment_id"]: {
                        "first_sentence": {
                            "text": cards[foundation]["source_quote"],
                            "fact_card_ids": [foundation],
                            "support_quotes": [cards[foundation]["source_quote"]],
                        },
                        "additional_sentences": [],
                    }
                }
            }
        )
        if valid_patch
        else "{}"
    )
    evaluation = AsyncMock(
        return_value={
            "citation_accuracy": {"passed": True},
            "faithfulness": {
                "skipped": True,
                "details": {"evaluation_version": "survey-evaluation-v3"},
            },
        }
    )
    monkeypatch.setattr(LitAgent, "_evaluate", evaluation)
    calls = []

    def client_factory(**kwargs):
        # Only the provider transport is replaced; the real budget and chat gate run.
        client = object.__new__(OpenAICompatibleClient)
        client._cost_budget = kwargs["cost_budget"]
        client._trace_hook = kwargs["trace_hook"]
        client._max_tokens = 16384
        client._transport_max_attempts = 1

        async def attempt(messages, **options):
            calls.append((copy.deepcopy(messages), copy.deepcopy(options)))
            client._emit(
                "llm.start",
                {
                    "operation_id": "offline-cli-operation",
                    "task_id": get_task_id(),
                    "model": "offline",
                },
            )
            usage = {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "usage_available": True,
            }
            client._cost_budget.record(usage)
            client._emit(
                "llm.complete",
                {
                    **usage,
                    "operation_id": "offline-cli-operation",
                    "task_id": get_task_id(),
                    "model": "offline",
                    "content": content,
                },
            )
            return LLMResponse(content=content, model="offline", usage=usage)

        client._chat_attempt = attempt
        client.close = AsyncMock()
        return client

    deps, _ = factories()
    deps.pop("agent_factory")
    deps["client_factory"] = client_factory
    service = deps["service_factory"].return_value
    service.submit = AsyncMock(side_effect=AssertionError("Unexpected audit request"))
    child = tmp_path / "child"
    auth = {"actor": "user", "binding": prepared["plan"]["binding"]}
    state = await run_trial(
        source,
        config,
        root=child,
        run_id=run_id,
        parent_root=parent_root,
        live=True,
        authorization=auth,
        **deps,
    )
    assert len(calls) == 1 and state["status"] == "blocked"
    messages, options = calls[0]
    packet = json.loads(messages[1]["content"])
    variant = next(cid for cid, c in cards.items() if c["evidence_id"] == "b")
    assert "methods" in packet["cards"][foundation]["allowed_section_keys"]
    assert "methods" not in packet["cards"][variant]["allowed_section_keys"]
    patches = options["response_format"]["json_schema"]["schema"]["properties"][
        "patches"
    ]
    patch = next(iter(patches["properties"].values()))
    assert patch["properties"]["first_sentence"]["properties"]["fact_card_ids"][
        "items"
    ]["enum"] == [foundation]
    assert state["new_survey_requests"] == 1 and state["new_survey_tokens"] == 2
    assert (
        state["budget"]["requests_sent"] == 60
        and state["budget"]["used_tokens"] == 5722
    )
    assert (
        state["report"]["evaluation"] == {}
        and not state["report"]["delivery"]["publishable"]
    )
    assert (
        state["report"]["metadata"]["repair_scope_provenance"]["updated_scope_sha256"]
        == prepared["plan"]["repair_scope"]["updated_scope_sha256"]
    )
    assert source.read_bytes() == source_before
    assert {
        str(p): p.read_bytes() for p in parent_root.rglob("*.json")
    } == parent_before
    again = await run_trial(
        source,
        config,
        root=child,
        run_id=run_id,
        parent_root=parent_root,
        live=True,
        authorization=auth,
        **deps,
    )
    assert again == state and len(calls) == 1
    archive = json.loads((child / "runs" / f"{run_id}.json").read_text())
    assert archive["usage"]["total_tokens"] == 2
    assert not (child / "audits" / "index.sqlite3").exists()
    service.submit.assert_not_awaited()
    if valid_patch:
        candidate = json.loads((child / "candidate.json").read_text())
        assert "[E:a]" in candidate["candidate"] and not candidate["committed"]
        evaluation.assert_awaited_once()
        assert evaluation.await_args.kwargs["survey"] == candidate["candidate"]
        assert evaluation.await_args.kwargs["results"]["adversarial_review"][
            "evidence_selection"
        ]["section_evidence_ids"]["methods"] == ["a"]
    else:
        evaluation.assert_not_awaited()
