"""Exercise official RAGAS execution while replacing only external transport."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from litagent.config import EvalConfig
from litagent.eval.ragas_adapter import NativeRagasAdapter
from litagent.llm.client import LLMResponse


class Transport:
    """Provide deterministic HTTP responses without bypassing RAGAS consumers."""

    def __init__(self, statements, verdicts, mutate=None):
        """Keep original input order and all generated statements observable."""
        self.statements, self.verdicts = statements, verdicts
        self.mutate, self.calls = mutate, []

    async def chat(self, messages, **kwargs):
        """Return official schemas inside the thin per-job batch envelope."""
        payload = json.loads(messages[1]["content"])
        self.calls.append((payload, kwargs))
        nli = "context_pool" in payload
        values = {}
        for key, row in payload["jobs"].items():
            if not nli:
                value = {"statements": self.statements[row["answer"]]}
            elif isinstance(row["statements"], dict):
                value = {
                    "statements": {
                        cid: {"reason": "fixture", "verdict": self.verdicts[text]}
                        for cid, text in row["statements"].items()
                    }
                }
            else:
                value = {
                    "statements": [
                        {
                            "statement": text,
                            "reason": "fixture",
                            "verdict": self.verdicts[text],
                        }
                        for text in row["statements"]
                    ]
                }
            values[key] = value
        if self.mutate:
            self.mutate(values, nli)
        return LLMResponse(
            json.dumps(values),
            model="offline",
            finish_reason="stop",
            usage={
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("keyed", [False, True])
async def test_official_metric_executes_and_aggregates_by_assertions(
    monkeypatch, keyed
):
    """Different sample sizes must not produce an unweighted paragraph average."""
    from ragas.metrics.collections import Faithfulness

    actual = Faithfulness.ascore
    seen = []

    async def spy(metric, **kwargs):
        seen.append(kwargs)
        return await actual(metric, **kwargs)

    monkeypatch.setattr(Faithfulness, "ascore", spy)
    llm = Transport(
        {"first": ["a", "b", "c"], "second": ["d"]}, {"a": 1, "b": 1, "c": 1, "d": 0}
    )
    samples = [
        {"id": f"u{i}", "question": "q", "response": text, "contexts": [source]}
        for i, (text, source) in enumerate(
            [("first", "original A"), ("second", "original B")]
        )
    ]
    result = await NativeRagasAdapter(llm, EvalConfig(), keyed_assessment=keyed).score(
        samples
    )
    assert result["score"] == 0.75
    assert len(seen) == 2 and len(llm.calls) == 2
    assert seen[0]["retrieved_contexts"] == ["original A"]
    payload = llm.calls[-1][0]
    for i, original in enumerate(("original A", "original B")):
        refs = payload["jobs"][f"u{i}"]["context_refs"]
        assert [payload["context_pool"][ref] for ref in refs] == [original]
    assert all(k["transport_max_attempts"] == 1 for _, k in llm.calls)
    assert all(
        k["extra_body"] == {"thinking": {"type": "disabled"}} for _, k in llm.calls
    )
    assert all(k["response_format"]["type"] == "json_schema" for _, k in llm.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing", "foreign", "duplicate", "bad_verdict", "uncited"]
)
async def test_keyed_assessment_rejects_unbound_or_invalid_verdicts(fault):
    """Exact keys, proper verdict types and evidence are required independently."""

    class BrokenTransport(Transport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "context_pool" not in payload:
                return response
            output = json.loads(response.content)
            decisions = output["u0"]["statements"]
            if fault == "missing":
                decisions.pop("u0_s1")
            elif fault == "foreign":
                decisions["u1_s1"] = decisions.pop("u0_s1")
            elif fault == "bad_verdict":
                decisions["u0_s1"]["verdict"] = True
            response.content = json.dumps(output)
            if fault == "duplicate":
                response.content = response.content.replace(
                    '"u0_s1": ',
                    '"u0_s1": {"verdict": 0, "reason": "duplicate"}, "u0_s1": ',
                    1,
                )
            return response

    error = {
        "missing": "ragas_statement_coverage_invalid",
        "foreign": "ragas_statement_coverage_invalid",
        "duplicate": "ragas_duplicate_response_key",
        "bad_verdict": "ragas_verdict_invalid",
        "uncited": "ragas_uncited_positive_verdict",
    }[fault]
    llm = BrokenTransport({"answer": ["A claim."]}, {"A claim.": 1})
    with pytest.raises(ValueError, match=error):
        await NativeRagasAdapter(llm, EvalConfig(), keyed_assessment=True).score(
            [
                {
                    "id": "u0",
                    "question": "q",
                    "response": "answer",
                    "contexts": [] if fault == "uncited" else ["source"],
                }
            ]
        )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_392_keyed_replay_keeps_original_statements():
    """Exercise all 84 recorded decisions under the new representation offline."""
    from jsonschema import Draft202012Validator

    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    path = artifacts / "acceptance/workbench-vit-386-20261110/runs/3edbe50c8d4a.json"
    if not path.exists():
        pytest.skip("requires the local immutable 392 archive")
    node = json.loads(path.read_bytes())["nodes"][
        "llm:77599b2ddb18487eb99edb1962c76ea9"
    ]
    original = json.loads(node["input"]["messages"][1]["content"])
    recorded = json.loads(node["output"])
    assert (
        original["jobs"]["u26"]["statements"][1]
        != recorded["u26"]["statements"][1]["statement"]
    )

    class KeyedReplay(Transport):
        async def chat(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            if "context_pool" not in payload:
                return await super().chat(messages, **kwargs)
            self.calls.append((payload, kwargs))
            decisions = {}
            for key, job in payload["jobs"].items():
                assert (
                    list(job["statements"].values())
                    == original["jobs"][key]["statements"]
                )
                assert job["context_refs"] == original["jobs"][key]["context_refs"]
                # This positional projection is limited to replaying the test fixture.
                decisions[key] = {
                    "statements": dict(
                        reversed(
                            [
                                (
                                    cid,
                                    {
                                        "reason": item["reason"],
                                        "verdict": item["verdict"],
                                    },
                                )
                                for cid, item in zip(
                                    job["statements"],
                                    recorded[key]["statements"],
                                    strict=True,
                                )
                            ]
                        )
                    )
                }
            assert payload["context_pool"] == original["context_pool"]
            Draft202012Validator(
                kwargs["response_format"]["json_schema"]["schema"]
            ).validate(decisions)
            return LLMResponse(
                json.dumps(decisions),
                model="offline",
                finish_reason="stop",
                usage={"usage_available": True},
            )

    llm = KeyedReplay(
        {key: job["statements"] for key, job in original["jobs"].items()}, {}
    )
    samples = [
        {
            "id": key,
            "question": "q",
            "response": key,
            "contexts": [original["context_pool"][ref] for ref in job["context_refs"]],
        }
        for key, job in original["jobs"].items()
    ]
    result = await NativeRagasAdapter(
        llm,
        EvalConfig(strict_batch_size=20, context_tokens=235930, max_tokens=39322),
        keyed_assessment=True,
    ).score(samples)
    assert len(llm.calls) == 2
    assert sum(len(sample["verdicts"]) for sample in result["samples"]) == 84
    assert (
        next(sample for sample in samples if sample["id"] == "u26")["verdicts"][1][
            "statement"
        ]
        == original["jobs"]["u26"]["statements"][1]
    )
    assert (
        result["score"]
        == sum(
            item["verdict"] for job in recorded.values() for item in job["statements"]
        )
        / 84
    )


@pytest.mark.asyncio
async def test_complete_originals_over_sixteen_contexts_are_not_truncated():
    """A valid context list has no invented sixteen-source selection protocol."""
    originals = [f"original {i}: " + ("x" * 200) for i in range(20)]
    llm = Transport({"answer": ["statement"]}, {"statement": 1})
    result = await NativeRagasAdapter(llm, EvalConfig()).score(
        [{"id": "u0", "question": "q", "response": "answer", "contexts": originals}]
    )
    assert result["score"] == 1
    payload = llm.calls[-1][0]
    assert [
        payload["context_pool"][ref] for ref in payload["jobs"]["u0"]["context_refs"]
    ] == originals


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["omit", "duplicate", "rewrite", "invalid_verdict", "unknown_job"]
)
async def test_bad_nli_never_becomes_a_partial_score(fault):
    """Reject bad response identity even when the resulting JSON is well formed."""

    def mutate(value, nli):
        if not nli:
            return
        items = value["u0"]["statements"]
        if fault == "omit":
            items.pop()
        elif fault == "duplicate":
            items.append(copy.deepcopy(items[0]))
        elif fault == "rewrite":
            items[0]["statement"] = "invented"
        elif fault == "invalid_verdict":
            items[0]["verdict"] = True
        else:
            value["other"] = value.pop("u0")

    llm = Transport({"answer": ["a", "b"]}, {"a": 1, "b": 0}, mutate)
    with pytest.raises(ValueError):
        await NativeRagasAdapter(llm, EvalConfig()).score(
            [
                {
                    "id": "u0",
                    "question": "q",
                    "response": "answer",
                    "contexts": ["original"],
                }
            ]
        )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_same_statement_at_two_positions_is_preserved():
    """Repeated text is not used as a cross-position score cache."""
    llm = Transport({"first": ["same"], "second": ["same"]}, {"same": 1})
    samples = [
        {"id": f"u{i}", "question": "q", "response": text, "contexts": [f"source{i}"]}
        for i, text in enumerate(("first", "second"))
    ]
    result = await NativeRagasAdapter(llm, EvalConfig()).score(samples)
    assert [s["id"] for s in result["samples"]] == ["u0", "u1"]
    assert sum(len(s["verdicts"]) for s in result["samples"]) == 2


@pytest.mark.asyncio
async def test_empty_inventory_is_not_full_credit():
    """Official NaN remains unavailable when no factual statements were found."""
    llm = Transport({"answer": []}, {})
    with pytest.raises(ValueError, match="ragas_no_statements"):
        await NativeRagasAdapter(llm, EvalConfig()).score(
            [
                {
                    "id": "u0",
                    "question": "q",
                    "response": "answer",
                    "contexts": ["original"],
                }
            ]
        )
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_unknown_usage_stops_without_nli_or_replay():
    """A response with unknown billing must not release further paid work."""

    class Unknown(Transport):
        async def chat(self, *args, **kwargs):
            response = await super().chat(*args, **kwargs)
            response.usage = {"usage_available": False}
            return response

    llm = Unknown({"answer": ["a"]}, {"a": 1})
    states = []
    adapter = NativeRagasAdapter(llm, EvalConfig(), checkpoint=states.append)
    sample = {
        "id": "u0",
        "question": "q",
        "response": "answer",
        "contexts": ["original"],
    }
    with pytest.raises(ValueError, match="ragas_usage_unknown"):
        await adapter.score([sample])
    assert len(llm.calls) == 1
    assert states[-1]["requests"][0]["status"] == "invalid_response"
    with pytest.raises(ValueError, match="ragas_usage_unknown"):
        await adapter.score([sample])
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_cancellation_preserves_unknown_request_and_stops_worker():
    """Cancellation cannot leave an orphaned batch or silently send NLI."""
    import asyncio

    class Blocked:
        def __init__(self):
            self.started = asyncio.Event()
            self.calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            self.started.set()
            await asyncio.Future()

    llm = Blocked()
    states = []
    adapter = NativeRagasAdapter(llm, EvalConfig(), checkpoint=states.append)
    task = asyncio.create_task(
        adapter.score(
            [
                {
                    "id": "u0",
                    "question": "q",
                    "response": "answer",
                    "contexts": ["original"],
                }
            ]
        )
    )
    await llm.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.calls == 1 and adapter.worker.done()
    assert states[-1]["requests"][0]["status"] == "interrupted_no_replay"


@pytest.mark.asyncio
@pytest.mark.parametrize("last_write", [False, True])
async def test_checkpoint_failure_is_terminal_even_after_last_verdict(last_write):
    """A disk failure must neither strand futures nor publish an unrecorded pass."""
    import asyncio

    llm = Transport({"answer": ["a"]}, {"a": 1})

    def checkpoint(state):
        if not last_write or (
            len(state["requests"]) == 2
            and state["requests"][-1]["status"] == "completed"
        ):
            raise OSError("offline disk failure")

    adapter = NativeRagasAdapter(llm, EvalConfig(), checkpoint=checkpoint)
    sample = {
        "id": "u0",
        "question": "q",
        "response": "answer",
        "contexts": ["original"],
    }
    with pytest.raises(OSError, match="offline disk failure"):
        await asyncio.wait_for(adapter.score([sample]), timeout=2)
    assert len(llm.calls) == (2 if last_write else 0)


def test_native_adapter_disables_ragas_external_telemetry(monkeypatch):
    """Third-party analytics is outside the configured evaluation transport."""
    from ragas._analytics import do_not_track

    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "false")
    do_not_track.cache_clear()
    assert do_not_track() is False
    NativeRagasAdapter(Transport({}, {}), EvalConfig())
    assert do_not_track() is True
