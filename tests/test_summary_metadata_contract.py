"""Prove metadata tables preserve complete frozen inputs and paid receipt lineage."""

import copy
import json
from pathlib import Path

import pytest
from test_frozen_summary_protocol import summary_prepared
from test_received_staged_body import received_prepared, reply, sdk
from test_staged_native_generation import provider_summaries
from test_staged_protocol_probe import accounted, prepared
from test_summary_reference_wire import parts

from litagent import staged_generation as staged
from litagent.report_audit import atomic_json, digest
from scripts import probe_frozen_summary as frozen
from scripts import probe_staged_generation as probe
from scripts import reconcile_frozen_summary as accounting


def packets(parts):
    """Use original production requests, including complete sources and policy."""
    body, _, messages, rf, binding = parts
    messages = copy.deepcopy(messages)
    original = json.loads(messages[1]["content"])
    sources = list(original["sources"].values())
    # Optional and extended fields must survive independently for each record.
    sources[0]["extra"] = {"year": None, "authors": ["甲", "Ω"]}
    sources[-1]["optional"] = None
    messages[1]["content"] = json.dumps(original)
    messages.append({"role": "system", "content": "Retain the bound citation policy."})
    old, old_rf = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
        host_layout=True,
    )
    new, new_rf = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
        host_layout=True,
        compact_metadata=True,
    )
    assert new_rf == old_rf and new[2:] == old[2:]
    return json.loads(old[1]["content"]), json.loads(new[1]["content"])


def test_every_source_card_optional_field_and_frozen_position_round_trips(parts):
    body, _, _, _, binding = parts
    original, compact = packets(parts)
    assert staged.expand_summary_metadata(compact) == original
    assert staged.expand_summary_body(
        compact["frozen_body"]
    ) == staged.expand_summary_body(original["frozen_body"])
    restored = staged.restore_summary_input(compact, binding, body=body)
    assert restored["frozen_body"] == body
    assert restored["sources"] == original["sources"]
    assert restored["cards"] == {
        cid: original["cards"][alias] for alias, cid in binding["reading_cards"].items()
    }
    assert "extra" in next(iter(restored["sources"].values()))


def test_distinct_source_shapes_preserve_missing_null_and_extension_fields(parts):
    original, _ = packets(parts)
    source = copy.deepcopy(next(iter(original["sources"].values())))
    source.pop("extra")
    original["sources"]["shape_missing"] = source
    original["sources"]["shape_null"] = {**source, "extra": None}
    compact = staged.compact_summary_metadata(original)
    restored = staged.expand_summary_metadata(compact)
    assert restored == original
    assert "extra" not in restored["sources"]["shape_missing"]
    assert restored["sources"]["shape_null"]["extra"] is None


def test_two_continuation_identities_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="continuation_conflict"):
        probe.reconcile_accounting(
            {}, tmp_path, predecessor_received={}, predecessor_frozen={}
        )


@pytest.mark.parametrize(
    "fault",
    [
        "body_index",
        "negative",
        "boolean",
        "row_width",
        "columns",
        "missing_section",
        "position",
        "text",
        "card_shape",
        "card_values",
        "card_columns",
        "section_set",
        "section_key",
        "source_shape",
        "source_names",
        "source_width",
        "source_extra",
        "source_missing",
        "card_source",
        "source_quote_missing",
        "source_quote_type",
    ],
)
def test_malformed_metadata_does_not_guess_values_or_positions(parts, fault):
    body, _, _, _, binding = parts
    _, packet = packets(parts)
    row = packet["frozen_body"]["rows"]["methods"][0]
    card = next(iter(packet["cards"]["rows"].values()))
    source = next(iter(packet["sources"]["rows"].values()))
    if fault == "body_index":
        row[3] = 1000
    elif fault == "negative":
        row[3] = -1
    elif fault == "boolean":
        row[3] = True
    elif fault == "row_width":
        row.pop()
    elif fault == "columns":
        packet["frozen_body"]["metadata_columns"].reverse()
    elif fault == "missing_section":
        del packet["frozen_body"]["rows"]["methods"]
    elif fault == "position":
        row[2] += 1
    elif fault == "text":
        row[4] += " changed"
    elif fault == "card_shape":
        card[1] = -1
    elif fault == "card_values":
        card[2].append(None)
    elif fault == "card_columns":
        packet["cards"]["columns"].reverse()
    elif fault == "section_set":
        card[3] = False
    elif fault == "section_key":
        packet["cards"]["section_sets"][0] = [999]
    elif fault == "source_shape":
        source[0] = True
    elif fault == "source_names":
        packet["sources"]["field_sets"][0] *= 2
    elif fault == "source_width":
        source.pop()
    elif fault == "source_missing":
        packet["sources"]["rows"].clear()
    elif fault == "card_source":
        card[0] = "unknown-source"
    elif fault == "source_quote_type":
        names = packet["sources"]["field_sets"][source[0]]
        source[names.index("source_quote") + 1] = None
    elif fault == "source_quote_missing":
        names = packet["sources"]["field_sets"][source[0]]
        index = names.index("source_quote")
        names.pop(index)
        for record in packet["sources"]["rows"].values():
            if record[0] == source[0]:
                record.pop(index + 1)
    else:
        packet["sources"]["unknown"] = "not ignored"
    with pytest.raises(ValueError):
        staged.restore_summary_input(packet, binding, body=body)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_original_prose_with_table_tokens_is_never_substituted(parts, language):
    from test_body_analysis import analysis_document
    from test_native_array_integration import request
    from test_staged_native_generation import split
    from test_writing_passages import facts

    body, _ = split(analysis_document(facts(), language))
    text = 'metadata_ref c1 field_sets sources.rows "section_set" Ω'
    body["sections"]["methods"]["blocks"][0]["sentences"][0]["text"] = text
    messages, rf = request(facts(), language, array_wire=True)
    binding = staged.summary_reference_binding(body)
    prepared, _ = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
        host_layout=True,
        compact_metadata=True,
    )
    packet = json.loads(prepared[1]["content"])
    restored = staged.restore_summary_input(packet, binding, body=body)
    assert restored["frozen_body"] == body
    assert (
        restored["frozen_body"]["sections"]["methods"]["blocks"][0]["sentences"][0][
            "text"
        ]
        == text
    )


@pytest.mark.asyncio
async def test_summary_receipt_advances_once_without_resetting_shared_repair(
    summary_prepared, monkeypatch
):
    root, manifest, _, _, summaries = summary_prepared
    sent = []

    async def send(request):
        packet = json.loads(request.content)
        sent.append(packet)
        # Preserve a complete, known-usage failure, as in the actual v7 receipt.
        value = provider_summaries(summaries, packet)
        value["abstract"][0]["heading"] = "摘要"
        return reply(value)

    sdk(monkeypatch, send)
    result = await frozen.run(root, live=True)
    assert result["status"] == "failed_no_replay" and len(sent) == 1
    first = accounting.reconcile(Path(manifest["parent"]), root)
    second = accounting.reconcile(Path(manifest["parent"]), root)
    assert first == second and len(sent) == 1
    inputs, body, repair = first
    assert inputs["parent_budget"] == result["budget"]
    assert inputs["parent_budget"]["used_tokens"] == 500
    assert inputs["parent_budget"]["requests_sent"] == 9
    assert inputs["historical"] == result["cumulative"]
    assert repair == manifest["repair_state"] and len(repair["attempts"]) == 1
    assert body == manifest["frozen_body"]
    assert all(probe.sha(Path(p)) == h for p, h in inputs["inputs"].items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["state", "events", "raw", "body", "repair", "checkpoint", "input"]
)
async def test_latest_receipt_rejects_accounting_and_material_drift(
    summary_prepared, monkeypatch, fault
):
    root, manifest, _, _, summaries = summary_prepared

    async def send(request):
        p = json.loads(request.content)
        return reply(provider_summaries(summaries, p))

    sdk(monkeypatch, send)
    result = await frozen.run(root, live=True)
    assert result["status"] == "protocol_completed"
    if fault in {"state", "checkpoint"}:
        path = root / ("state.json" if fault == "state" else "generation.json")
        v = json.loads(path.read_text())
        if fault == "state":
            v["budget"]["used_tokens"] -= 1
        else:
            v["state"]["attempts"] = []
    elif fault == "events":
        path = root / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["event"] == "llm.complete":
                row["data"]["usage_available"] = False
        path.write_text("\n".join(json.dumps(row) for row in rows))
        v = None
    elif fault == "input":
        path = root / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        start = next(row["data"] for row in rows if row["event"] == "llm.start")
        packet = json.loads(start["messages"][1]["content"])
        packet["frozen_body"]["rows"]["methods"][0][7] += " changed"
        start["messages"][1]["content"] = json.dumps(packet, ensure_ascii=False)
        path.write_text("\n".join(json.dumps(row) for row in rows))
        journal_path = next(root.glob("generation.json.generation-*.json"))
        journal = json.loads(journal_path.read_text())
        journal["steps"]["summaries"]["input_sha256"] = digest(
            [start["messages"], start["response_format"]]
        )
        atomic_json(journal_path, journal)
        v = None
    elif fault == "raw":
        path = next(root.glob("generation.json.generation-*.json"))
        v = json.loads(path.read_text())
        v["steps"]["summaries"]["raw_response"] += " "
    else:
        path = root / "manifest.json"
        v = json.loads(path.read_text())
        if fault == "body":
            v["frozen_body"]["body"]["title"] += " drift"
        else:
            v["repair_state"]["attempts"] = []
        v["binding"] = digest({k: x for k, x in v.items() if k != "binding"})
    if v is not None:
        atomic_json(path, v)
    with pytest.raises(ValueError):
        accounting.reconcile(Path(manifest["parent"]), root)
