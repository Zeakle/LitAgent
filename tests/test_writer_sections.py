"""Keep fixed chapter requests inside the original full scientific document contract."""

import asyncio
import copy
import json

import pytest
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.writing_v2 import (
    BODY,
    WriterError,
    WriterSession,
    body_section_schema,
    body_units,
    schema_for,
    validate_body,
)


def test_section_schemas_are_exact_projections_of_existing_fields(case):
    """The wire contains no new output fields or weakened source enums."""
    full = schema_for("body", case.aliases)
    before = copy.deepcopy(full)
    for i, section in enumerate(BODY):
        projected = body_section_schema(section, case.aliases)
        keys = ["title", section] if i == 0 else [section]
        assert projected["required"] == keys
        assert not projected["additionalProperties"]
        assert projected["properties"] == {k: full["properties"][k] for k in keys}
    assert full == before
    with pytest.raises(WriterError, match="section_invalid"):
        body_section_schema("not_a_chapter", case.aliases)


@pytest.mark.asyncio
async def test_fixed_chapters_keep_all_originals_and_merge_without_loss(case, tmp_path):
    """Repeated sources remain complete and prior prose is explicitly unverified."""
    p = Provider(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    body = await s.initial_body()
    assert body == case.body
    assert len(p.calls) == len(BODY) == 5
    for i, (payload, kwargs) in enumerate(p.calls):
        assert payload["requested_body_section"] == BODY[i]
        for key, value in case.packet.items():
            assert payload[key] == value
        assert set(payload["prior_chapters_unverified"]) == (
            set() if i == 0 else {"title", *BODY[:i]}
        )
        assert kwargs["max_tokens"] == 20000
        assert kwargs["transport_max_attempts"] == 1
    assert len(s.state["attempts"]) == 0
    assert all(
        v["status"] == "local_contract_completed"
        for v in s.journal["body_parts"].values()
    )
    assert validate_body(body, case.aliases)


@pytest.mark.parametrize(
    "fault", ["json", "other_chapter", "empty", "null", "layout", "title"]
)
@pytest.mark.asyncio
async def test_bad_chapter_stops_before_any_dependent_request(case, tmp_path, fault):
    """No skipped chapter, repaired JSON, summary call, or false completed body."""

    class Broken(Provider):
        """Corrupt one response without bypassing the actual chapter consumer."""

        async def chat(self, messages, **kwargs):
            """Keep valid earlier chapters and inject only the selected fault."""
            response = await super().chat(messages, **kwargs)
            target = 1 if fault == "title" else 2
            if len(self.calls) != target:
                return response
            value = json.loads(response.content)
            section = BODY[target - 1]
            if fault == "json":
                raw = response.content[:-1]
            else:
                if fault == "other_chapter":
                    value[BODY[-1]] = copy.deepcopy(case.body[BODY[-1]])
                elif fault == "empty":
                    value[section] = []
                elif fault == "null":
                    value[section][0]["sentences"][0]["text"] = None
                elif fault == "layout":
                    value[section][0]["sentences"][0]["text"] = "<script>bad()</script>"
                elif fault == "title":
                    value.pop("title")
                raw = json.dumps(value)
            response.content = raw
            return response

    p = Broken(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError):
        await s.initial_body()
    assert len(p.calls) == (1 if fault == "title" else 2)
    assert s.journal["failure"]["body_section"] == BODY[len(p.calls) - 1]
    assert "summary_initial" not in s.journal["calls"]
    assert len(s.state["attempts"]) == 0


@pytest.mark.asyncio
async def test_unknown_reference_retains_full_candidate_for_shared_repair(
    case, tmp_path
):
    """An invalid chapter cannot become evidence or disappear from the candidate."""
    case.body[BODY[0]][0]["sentences"][0]["fact_card_refs"] = ["unknown"]
    p = Provider(case)
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="reference_invalid") as caught:
        await s.initial_body()
    body = caught.value.candidate
    assert set(body) == {"title", *BODY} and len(p.calls) == 5
    assert body == case.body
    assert s.journal["body_parts"][BODY[0]]["status"] == "pending_reference_repair"
    assert all(
        BODY[0] not in payload["prior_chapters_unverified"] for payload, _ in p.calls
    )
    original = copy.deepcopy(body[BODY[0]][0]["sentences"][0])
    patched = await s.patch(
        body,
        {"first": {"path": [BODY[0], 0, 0], "original": original}},
        cause="local",
        claim_count=len(body_units(body)),
    )
    assert validate_body(patched, case.aliases)
    assert len(s.state["attempts"]) == 1
    assert all(patched[k] == body[k] for k in BODY[1:])
    assert body[BODY[0]][0]["sentences"][0] == original


@pytest.mark.asyncio
async def test_cancelled_prefix_does_not_resume_or_resend_any_chapter(case, tmp_path):
    """The existing no-replay rule applies even when earlier chapters completed."""

    class Cancelled(Provider):
        """Cancel after one completed response at the external transport boundary."""

        async def chat(self, messages, **kwargs):
            """Leave the next request uncertain without manufacturing a response."""
            if len(self.calls) == 1:
                raise asyncio.CancelledError()
            return await super().chat(messages, **kwargs)

    path = tmp_path / "r.json"
    p = Cancelled(case)
    s = WriterSession(p, path, "r", case.packet, case.aliases)
    with pytest.raises(asyncio.CancelledError):
        await s.initial_body()
    assert s.journal["calls"]["body:" + BODY[1]]["status"] == "cancelled_no_replay"
    later = Provider(case)
    before = s.journal_path.read_bytes()
    resumed = WriterSession(later, path, "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="no_replay"):
        await resumed.initial_body()
    assert later.calls == []
    assert s.journal_path.read_bytes() == before


@pytest.mark.asyncio
async def test_real_sdk_section_budget_stops_before_next_request(case, tmp_path):
    """Chapter splitting cannot spend a second request after the same budget is used."""
    import httpx
    from openai import AsyncOpenAI

    from litagent.llm.client import OpenAICompatibleClient
    from litagent.observability.context import reset_task_id, set_task_id
    from litagent.safety.budget import CostBudget

    sent = []

    def transport(request):
        """Return complete usage while preserving actual SDK request encoding."""
        payload = json.loads(request.content)
        sent.append(payload)
        assert payload["max_tokens"] == 20000
        keys = payload["response_format"]["json_schema"]["schema"]["properties"]
        return httpx.Response(
            200,
            json={
                "id": "budget-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({k: case.body[k] for k in keys}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12000,
                    "completion_tokens": 8000,
                    "total_tokens": 20000,
                },
            },
        )

    budget = CostBudget(max_tokens=60000)
    budget.configure_strict(halt_on_unknown_usage=True)
    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "pro", api_key="offline", cost_budget=budget
    )
    await owner.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    session = WriterSession(owner, tmp_path / "r.json", "r", case.packet, case.aliases)
    token = set_task_id("synthesis:sections")
    try:
        with pytest.raises(ValueError, match="budget"):
            await session.initial_body()
        assert len(sent) == 1
        assert budget.used == 20000 and budget.snapshot()["requests_sent"] == 1
        assert session.journal["failure"]["body_section"] == BODY[1]
        assert set(session.journal["calls"]) == {"body:" + BODY[0], "body:" + BODY[1]}
        rejected = session.journal["calls"]["body:" + BODY[1]]
        assert rejected["status"] == "rejected_not_sent"
        assert rejected["stage"] == "synthesis"
        assert session.journal["failure"]["request_sent"] is False
        assert budget._reserved_tokens == 0
    finally:
        reset_task_id(token)
        await owner.close()
