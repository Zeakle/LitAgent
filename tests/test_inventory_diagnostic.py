"""Exercise flat ownership and bounded protocol probes without paid transport."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_body_quality_comparison import prepared
from test_strict_automatic import frozen

from litagent.eval.scoped import (
    flat_inventory_response,
    flat_inventory_schema,
)
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.report_audit import atomic_json
from litagent.safety.budget import CostBudget, strict_request_reservation
from scripts import run_inventory_diagnostic as diagnostic


def units():
    """Keep explicit host ownership across two independently cited report units."""
    return [
        {
            "source_unit_id": f"u{i}",
            "scope_context": "",
            "citation_aliases": [f"e{i}"],
            "source_fragments": [
                {
                    "source_fragment_id": f"u{i}:f0",
                    "text": "EViT improves speed.",
                    "citation_aliases": [f"e{i}"],
                }
            ],
        }
        for i in range(2)
    ]


def flat_result(source_units):
    """Return semantics under exact host keys without model-owned citation arrays."""
    return {
        "fragments": {
            f["source_fragment_id"]: {
                "assertions": [{"kind": "scientific", "claim_text": f["text"]}],
                "exclusion_reason": "",
            }
            for u in source_units
            for f in u["source_fragments"]
        }
    }


@pytest.mark.parametrize(
    "fault", [None, "missing", "unknown", "duplicate", "wrong_owner", "omit_science"]
)
def test_flat_inventory_never_guesses_or_exempts_scientific_content(fault):
    source = units()
    data = flat_result(source)
    text = json.dumps(data)
    if fault == "missing":
        data["fragments"].pop("u1:f0")
    elif fault == "unknown":
        data["fragments"]["unknown"] = data["fragments"].pop("u1:f0")
    elif fault == "wrong_owner":
        data["fragments"]["u0:f0"]["source_unit_id"] = "u1"
    elif fault == "omit_science":
        data["fragments"]["u0:f0"] = {
            "assertions": [],
            "exclusion_reason": "Organization only.",
        }
    elif fault == "duplicate":
        item = json.dumps(data["fragments"]["u0:f0"])
        text = '{"fragments":{"u0:f0":' + item + ',"u0:f0":' + item + "}}"
    if fault != "duplicate":
        text = json.dumps(data)
    if fault:
        with pytest.raises(ValueError):
            flat_inventory_response(text, source, {}, research_gaps=True)
        return
    Draft202012Validator(flat_inventory_schema(source)).validate(data)
    progress = {}
    parsed = flat_inventory_response(text, source, progress, research_gaps=True)
    assert [u.source_unit_id for u in parsed.units] == ["u0", "u1"]
    for i, unit in enumerate(parsed.units):
        assert unit.assertions[0].source_fragment_id == f"u{i}:f0"
        assert unit.assertions[0].evidence_aliases == [f"e{i}"]
    assert len(progress["fragment_inventory"]) == 2


def test_flat_schema_rejects_ambiguous_host_partitions():
    source = units()
    source[1]["source_fragments"][0]["source_fragment_id"] = "u0:f0"
    with pytest.raises(ValueError, match="host_partition_invalid"):
        flat_inventory_schema(source)


def ready(tmp_path):
    """Build a valid terminal parent with unchanged frozen evidence and accounting."""
    source, config, manifest = prepared(tmp_path, formal_enabled=False)
    base = frozen()
    case = source / "vit-baseline"
    report = copy.deepcopy(base["report"])
    inputs = json.loads((case / "input-snapshot.json").read_text())
    report["metadata"]["evidence_snapshot_sha256"] = inputs["fact_cards_sha256"]
    report_sha = diagnostic.file_sha_text(report["survey"])
    atomic_json(
        case / "checkpoint.json", {"report": report, "report_sha256": report_sha}
    )
    budget = {
        s: {"charged": 0, "requests": 0, "usage_complete": True}
        for s in ("survey", "judge", "jev")
    }
    atomic_json(
        source / "comparison.json",
        {
            "binding": manifest["binding"],
            "status": "stopped",
            "pending": None,
            "budgets": budget,
        },
    )
    atomic_json(
        source / "terminal-summary.json",
        {
            "binding": manifest["binding"],
            "budgets": budget,
            "report_sha256": report_sha,
            "cumulative_after": manifest["historical"]["services"],
        },
    )
    payload = {
        "source_units": units(),
        "schema": {},
        "allowed_citation_aliases": ["e0", "e1"],
    }
    (case / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "llm.start",
                "data": {
                    "task_id": "evaluation.initial",
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                },
            }
        )
        + "\n"
    )
    root = tmp_path / "diagnostic"
    plan = diagnostic.prepare(root, source, config)
    return root, config, plan


class ProbeClient:
    """Replace provider transport while accounting for every distinct attempt."""

    def __init__(self, budget, hook, fault):
        self._cost_budget = budget
        self.hook = hook
        self.fault = fault
        self.calls = []

    async def chat(self, messages, **kwargs):
        self._cost_budget.begin_request()
        self.calls.append(messages)
        if self.fault == "unknown":
            self._cost_budget.charge_unknown_usage(
                strict_request_reservation(
                    messages, kwargs["max_tokens"], kwargs["response_format"]
                )
            )
            raise ValueError("strict_usage_incomplete")
        self._cost_budget.record({"prompt_tokens": 10, "completion_tokens": 10})
        payload = json.loads(messages[1]["content"])
        data = flat_result(payload["source_units"])
        if (
            kwargs["response_format"]["json_schema"]["name"]
            == "host_fragment_inventory_v2"
        ):
            data = {
                "units": {
                    u["source_unit_id"]: {
                        f["source_fragment_id"]: data["fragments"][
                            f["source_fragment_id"]
                        ]
                        for f in u["source_fragments"]
                    }
                    for u in payload["source_units"]
                }
            }
        return LLMResponse(
            content=(
                "{"
                if self.fault == "protocol" and len(self.calls) == 1
                else json.dumps(data)
            ),
            model=diagnostic.MODEL,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "usage_available": True,
            },
            finish_reason="stop",
            request_id="request-offline",
        )

    async def close(self):
        """No external resources are owned by the fake transport."""


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "protocol", "unknown"])
async def test_only_three_distinct_probes_run_and_unknown_usage_stops(tmp_path, fault):
    root, config, plan = ready(tmp_path)
    clients = []

    def factory(budget, hook):
        client = ProbeClient(budget, hook, fault)
        clients.append(client)
        return client

    assert (await diagnostic.run(root, config))["paid_calls"] == 0
    assert not (root / "checkpoint.json").exists()
    state = await diagnostic.run(root, config, live=True, client_factory=factory)
    expected = 1 if fault == "unknown" else 3
    assert len(clients[0].calls) == expected == state["budget"]["requests_sent"]
    if fault == "unknown":
        probe = json.loads((root / "short_nested.json").read_text())
        assert state["budget"]["used_tokens"] == strict_request_reservation(
            probe["messages"], probe["max_tokens"], probe["response_format"]
        )
    else:
        assert state["budget"]["used_tokens"] == expected * 20
    assert state["budget"]["usage_uncertain"] == (fault == "unknown")
    assert state["scientific_verified"] is False
    assert state["flat_full_inventory_complete"] == (fault != "unknown")
    if fault == "protocol":
        assert state["observations"][0]["status"] == "protocol_invalid"
    again = await diagnostic.run(root, config, live=True, client_factory=factory)
    assert again == state and len(clients) == 1
    assert plan["historical"]["jev"]["charged"] == 200000


@pytest.mark.asyncio
async def test_restart_charges_pending_once_and_never_creates_a_client(tmp_path):
    root, config, plan = ready(tmp_path)
    budget = CostBudget(max_tokens=200000)
    budget.configure_strict(halt_on_unknown_usage=True)
    atomic_json(
        root / "checkpoint.json",
        {
            "binding": plan["binding"],
            "status": "running",
            "pending": {"reserved_tokens": 9000},
            "observations": [],
            "budget": budget.snapshot(),
        },
    )
    factory = AsyncMock()
    result = await diagnostic.run(root, config, live=True, client_factory=factory)
    assert result["budget"]["used_tokens"] == 9000
    assert result["budget"]["requests_sent"] == 1
    assert result["pending"] is None
    assert (
        await diagnostic.run(root, config, live=True, client_factory=factory)
    ) == result
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_changed_source_is_rejected_before_transport(tmp_path):
    root, config, plan = ready(tmp_path)
    source = next(p for p in plan["hashes"] if p.endswith("terminal-summary.json"))
    Path(source).write_text("{}")
    with pytest.raises(ValueError, match="identity_changed"):
        await diagnostic.run(root, config, live=True, client_factory=AsyncMock())


@pytest.mark.asyncio
async def test_actual_sdk_request_id_is_preserved_locally(monkeypatch):
    events = []
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "offline",
        api_key="offline",
        trace_hook=lambda e, d: events.append((e, d)),
    )
    response = SimpleNamespace(
        model="actual-offline",
        _request_id="request-safe",
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="{}", tool_calls=None),
            )
        ],
    )
    monkeypatch.setattr(
        client._client.chat.completions, "create", AsyncMock(return_value=response)
    )
    try:
        result = await client.chat([{"role": "user", "content": "test"}])
    finally:
        await client.close()
    assert result.request_id == "request-safe" and result.model == "actual-offline"
    assert events[-1][1]["request_id"] == "request-safe"
    assert events[-1][1]["actual_model"] == "actual-offline"
