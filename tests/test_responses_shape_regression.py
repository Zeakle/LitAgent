"""Reproduce observed 011 shape failure without coercion or paid requests."""

import base64
import copy
import hashlib
import json

import httpx
import pytest
from test_native_responses import envelope, owner_for, session_for
from test_writer_v2 import BODY, case  # noqa: F401

from litagent.writing_v2 import WriterError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "chapter_object",
        "paragraph_array",
        "sentences_object",
        "text_array",
        "refs_string",
        None,
    ],
)
async def test_completed_strict_response_keeps_shape_gate_and_exact_receipt(
    case, tmp_path, fault
):
    """HTTP success never licenses shape conversion, dependent calls or replay."""
    sent, received = [], []

    def transport(request):
        """Mirror the observed object/array error using synthetic scientific text."""
        payload = json.loads(request.content)
        sent.append(payload)
        assert payload["text"]["format"]["strict"] is True
        data = json.loads(payload["input"][1]["content"])
        for key, value in case.packet.items():
            assert data[key] == value
        section = data["requested_body_section"]
        assert section == BODY[len(sent) - 1]
        result = {section: copy.deepcopy(case.body[section])}
        if len(sent) == 1:
            result["title"] = case.body["title"]
        if section == "taxonomy":
            if fault == "chapter_object":
                result[section] = result[section][0]
            elif fault == "paragraph_array":
                result[section][0] = result[section][0]["sentences"]
            elif fault == "sentences_object":
                result[section][0]["sentences"] = result[section][0]["sentences"][0]
            elif fault == "text_array":
                unit = result[section][0]["sentences"][0]
                unit["text"] = [unit["text"]]
            elif fault == "refs_string":
                unit = result[section][0]["sentences"][0]
                unit["fact_card_refs"] = unit["fact_card_refs"][0]
        raw = json.dumps(result, ensure_ascii=False)
        response = httpx.Response(200, json=envelope(raw))
        received.append((raw, response.content))
        return response

    owner, budget, events = await owner_for(transport)
    session = session_for(owner, case, tmp_path)
    try:
        if fault is None:
            assert await session.initial_body() == case.body
            assert len(sent) == 5
            assert len(session.journal["body_parts"]) == 5
        else:
            with pytest.raises(WriterError, match="writer_schema_invalid"):
                await session.initial_body()
            assert len(sent) == 2
            assert list(session.journal["body_parts"]) == ["introduction"]
            failure = session.journal["failure"]
            assert failure["body_section"] == "taxonomy"
            assert failure["diagnostics"][0]["validator"] == "type"
            assert failure["diagnostics"][0]["path"][0] == "taxonomy"
        for (name, call), (raw, response_bytes) in zip(
            session.journal["calls"].items(), received, strict=True
        ):
            assert name.startswith("body:")
            assert call["raw"] == raw
            assert call["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
            receipt = call["provider_receipt"]
            assert receipt["status_code"] == 200
            assert base64.b64decode(receipt["response_body_base64"]) == response_bytes
            assert call["usage"]["usage_available"] is True
        assert not session.state["attempts"]
        assert not session.journal["candidates"]
        assert "summary_initial" not in session.journal["calls"]
        assert budget.used == 120 * len(sent)
        assert budget.snapshot()["requests_sent"] == len(sent)
        assert budget._reserved_tokens == 0
        assert not budget.snapshot()["usage_uncertain"]
        assert len([e for e, _ in events if e == "llm.complete"]) == len(sent)
        before = session.journal_path.read_bytes()
        restarted = session_for(owner, case, tmp_path)
        with pytest.raises(WriterError, match="writer_checkpoint_no_replay"):
            await restarted.initial_body()
        assert session.journal_path.read_bytes() == before
        assert len(sent) == (5 if fault is None else 2)
    finally:
        await owner.close()
