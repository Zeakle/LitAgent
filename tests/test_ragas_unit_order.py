"""Exercise explicit and implicit ordering in versioned native global requests."""

import copy
import json

import pytest
from test_global_status_codes import payload

from litagent.config import load_config
from litagent.eval.engine_policy import (
    ARCHIVED_PRO_INVENTORY_ENGINE,
    configured_engine,
    validate_engine,
)
from litagent.eval.ragas_global_wire import (
    ORDERED_UNIT_CONTRACT,
    SOURCE_UNIT_ORDER,
    UNIT_CONTRACT,
    pack_global_input,
    unpack_global_input,
)


@pytest.mark.parametrize("interleaved", [False, True])
def test_unit_order_preserves_identity_even_with_repeated_prose(interleaved):
    canonical = payload()
    extra = {**canonical["assertions"][0], "claim_id": "claim_another"}
    canonical["assertions"].insert(2 if interleaved else 1, extra)
    canonical["source_units"][0]["extracted_claim_ids"].append(extra["claim_id"])
    wire = pack_global_input(
        canonical,
        by_unit=True,
        status_codes=True,
        inline_sources=True,
        unit_contract=ORDERED_UNIT_CONTRACT,
    )
    assert wire["claim_order"] == (
        [a["claim_id"] for a in canonical["assertions"]]
        if interleaved
        else SOURCE_UNIT_ORDER
    )
    assert unpack_global_input(json.loads(json.dumps(wire))) == canonical
    legacy = copy.deepcopy(wire)
    legacy["global_input_contract"] = UNIT_CONTRACT
    if not interleaved:
        with pytest.raises(ValueError, match="unit_projection_invalid"):
            unpack_global_input(legacy)
    wire["source_units"][1]["claims"].append(wire["source_units"][0]["claims"].pop())
    with pytest.raises(ValueError, match="unit_projection_invalid"):
        unpack_global_input(wire)


def test_stage_profile_selects_new_identity_and_old_inventory_stays_read_only():
    config = load_config("config/llm-stage-repair-stream.yaml")
    current = configured_engine(config)
    assert current["global_input_contract"] == ORDERED_UNIT_CONTRACT
    assert validate_engine(current) == current
    assert (
        validate_engine(ARCHIVED_PRO_INVENTORY_ENGINE, archived=True)
        == ARCHIVED_PRO_INVENTORY_ENGINE
    )
    with pytest.raises(ValueError, match="engine_identity_invalid"):
        validate_engine(ARCHIVED_PRO_INVENTORY_ENGINE)
    with pytest.raises(ValueError, match="engine_identity_invalid"):
        validate_engine(
            {**current, "identity": ARCHIVED_PRO_INVENTORY_ENGINE["identity"]},
            archived=True,
        )
    config.eval.native_global_input = None
    assert configured_engine(config)["identity"] != current["identity"]


@pytest.mark.asyncio
async def test_global_sender_uses_the_frozen_order_contract(monkeypatch):
    """Compare the sender's complete input and schema against the canonical report."""
    from unittest.mock import AsyncMock

    from litagent.eval import ragas_native as native
    from litagent.report_render import build_report_source_units

    config = load_config("config/llm-stage-repair-stream.yaml")
    # This replay verifies the previous complete-report wire.
    config.eval.native_global_scope = None
    env = {"semantic_engine": configured_engine(config)}
    survey = "# Report\n\n## Introduction\n\nResult A [E:e1]."
    units = build_report_source_units(survey)
    ctx = {"writing_bindings": {"citation_contract": "flat-writer-v2"}}
    claim = dict(
        payload()["assertions"][0],
        source_unit_id=units[-1].unit_id,
        section="Introduction",
    )
    expected, _, expected_fmt = native.global_review_request(
        survey,
        ctx,
        config.eval,
        units,
        [claim],
        anchored=True,
        force_compact=True,
        unit_response_contract=native.native_unit_response_contract(env),
    )
    monkeypatch.setattr(native, "locked_writer_envelope", lambda budget: env)
    transport = AsyncMock()
    transport._cost_budget = None
    transport.chat.side_effect = RuntimeError("recorded outbound request")
    with pytest.raises(RuntimeError, match="recorded outbound request"):
        await native._global_review(
            transport, survey, ctx, config.eval, units, [claim], {}, force_compact=True
        )
    args, kwargs = transport.chat.await_args
    messages = args[0]
    wire = json.loads(messages[1]["content"])
    assert wire["global_input_contract"] == ORDERED_UNIT_CONTRACT
    assert wire["claim_order"] == SOURCE_UNIT_ORDER
    assert unpack_global_input(wire) == expected
    assert kwargs["response_format"] == expected_fmt
    assert kwargs["max_output_tokens"] == config.eval.max_tokens
    assert ORDERED_UNIT_CONTRACT in messages[0]["content"]
