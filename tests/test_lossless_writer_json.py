"""Verify that recovery only edits JSON syntax and preserves the archive."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from test_native_responses import envelope, owner_for
from test_unavailable_cost_exports import display_case, exported
from test_writer_v2 import case

from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.engine_policy import configured_engine
from litagent.eval.results import CONTRACT as RESULT_CONTRACT
from litagent.frozen_writing import plan_frozen_budget
from litagent.llm.routing import ModelBoundClient
from litagent.output_capacity import configured_output_tokens
from litagent.quality import build_fact_cards, digest
from litagent.run_limits import LOSSLESS_JSON_POLICY, configured_run_limits
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_flat import encode, parse
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    body_section_schema,
    body_units,
    chapter_request_input,
    compile_document,
    source_packet,
    writer_format_status,
)

POLICY = {"syntax_policy": LOSSLESS_JSON_POLICY}


@pytest.mark.parametrize(
    "change, kind",
    [
        (lambda raw: raw + "}  \r\n", "extra_root_close"),
        (lambda raw: raw[:-2], "outer_eof_close"),
        (lambda raw: raw[:-3] + ",},],}", "trailing_comma"),
        (lambda raw: "  ```json\r\n" + raw + "\r\n```\n", "markdown_fence"),
    ],
)
def test_four_recoveries_preserve_original_literals(case, change, kind):
    schema = body_section_schema("methods", case.aliases)
    text = '证据说明有 }、], 和转义引号 "，以及反斜杠 \\。'
    value = {"methods": [{"sentences": [{"text": text, "fact_card_refs": ["c1"]}]}]}
    raw = json.dumps(encode(value, schema), ensure_ascii=False)
    damaged = change(raw)
    parsed = parse(damaged, schema, **POLICY)
    assert parsed["canonical"] == value
    assert parsed["normalized_raw"] == raw or parsed["normalized_raw"].rstrip() == raw
    assert {edit["kind"] for edit in parsed["normalization"]["edits"]} == {kind}
    unit = parsed["bindings"]["units"][0]
    for span, expected in [
        (unit["text_span"], text),
        (unit["reference_spans"][0], "c1"),
    ]:
        assert json.loads(damaged[span["start"] : span["end"]]) == expected
    assert parse(raw, schema, **POLICY) == parse(raw, schema)


@pytest.mark.parametrize(
    "change",
    [
        lambda raw: raw + "}}",
        lambda raw: raw + "{}",
        lambda raw: "here is the output\n```json\n" + raw + "\n```",
        lambda raw: raw[:-3],  # Unit itself is not closed.
        lambda raw: raw[:-1] + ",,}",
        lambda raw: raw.replace('"paragraph_start": true', '"paragraph_start": "true"')
        + "}",
        lambda raw: raw.replace('"text":', '"text": "duplicate", "text":') + "}",
        lambda raw: raw.replace('"text":', '"extra": 1, "text":') + "}",
        lambda raw: raw.replace('"text":', '"extra":') + "}",
        lambda raw: raw.replace('"c1"', '"outside"') + "}",
        lambda raw: '{"units": [{"text": "unfinished\\',
        lambda raw: "```json\n" + "[" * 2000 + raw + "]" * 2000 + "\n```",
        lambda raw: raw.replace(', "fact_card_refs"', ' "fact_card_refs"') + "}",
    ],
)
def test_recovery_rejects_ambiguous_or_invalid_content(case, change):
    schema = body_section_schema("methods", case.aliases)
    value = {
        "methods": [
            {"sentences": [{"text": "Verbatim text.", "fact_card_refs": ["c1"]}]}
        ]
    }
    raw = json.dumps(encode(value, schema))
    with pytest.raises(WriterError):
        parse(change(raw), schema, **POLICY)


@pytest.mark.asyncio
async def test_archived_075_replays_through_real_request_and_reference_consumer(
    tmp_path,
):
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/ragas-native-single-20261109-075"
    )
    names = [
        "execution.json",
        "result.json",
        "repair.json.writer-v2.json",
        "final-budget.json",
    ]
    if not all((root / name).is_file() for name in names):
        pytest.skip("Original 075 archive is local to the project")
    originals = {name: (root / name).read_bytes() for name in names}
    execution, result, journal, accounting = (
        json.loads(originals[name]) for name in names
    )
    old_plan = accounting["plan"]
    validate_accounting_snapshot(accounting, expected_plan=old_plan)
    config = load_config(
        Path(__file__).resolve().parents[1] / "config/extended-repair.yaml"
    )
    bundle = execution["bundle"]
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        capacity=config.capacity,
        output_tokens=configured_output_tokens(config),
        preparation_budget_snapshot=execution["preparation_snapshot"],
        semantic_engine=configured_engine(config),
        run_limits=configured_run_limits(config),
        **profile_flags(PROFILE),
        evaluation_result_contract=RESULT_CONTRACT,
    )
    assert plan["identity"] != old_plan["identity"]
    for key in ("planned_requests", "planned_tokens", "stages", "repair_limit"):
        assert plan[key] == old_plan[key]
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    packet, aliases = source_packet(
        cards, selection, bundle["query"], bundle["requirements"], "zh-CN"
    )
    section = "experiments"
    entry = journal["calls"]["body:" + section]
    raw = entry["raw"]
    assert hashlib.sha256(raw.encode()).hexdigest() == entry["raw_sha256"]
    prefix = {}
    for name in BODY[: BODY.index(section)]:
        prefix.update(journal["calls"]["body:" + name]["reference_canonical"])
    system, payload, schema = chapter_request_input(
        packet, aliases, section, prefix, prefix, RANGE_WRITING_PLAN, plan["envelope"]
    )
    assert digest(schema) == entry["schema_sha256"]
    with pytest.raises(WriterError, match="writer_json_invalid"):
        parse(raw, schema, allow_eof=True)
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(
            200,
            json=envelope(
                raw,
                usage={
                    "input_tokens": entry["usage"]["prompt_tokens"],
                    "output_tokens": entry["usage"]["completion_tokens"],
                    "total_tokens": entry["usage"]["prompt_tokens"]
                    + entry["usage"]["completion_tokens"],
                },
            ),
        )

    owner, _, _ = await owner_for(handler)
    owner._cost_budget.restore(execution["preparation_snapshot"])
    owner._cost_budget.activate_plan(plan)

    def session():
        return WriterSession(
            ModelBoundClient(owner, "pro", "generation"),
            tmp_path / "repair.json",
            "075-offline",
            packet,
            aliases,
            provider_api="responses",
            writing_plan=RANGE_WRITING_PLAN,
            **POLICY,
        )

    try:
        writer = session()
        part = await writer.call("body:" + section, system, payload, schema)
        received, origins = writer.reference_input(part)
        assert len(origins) == 10
        saved = writer.journal["calls"]["body:" + section]
        assert all(
            saved[key] == entry[key]
            for key in ("input_sha256", "provider_schema_sha256")
        )
        assert saved["normalized_raw"] == raw[:-1]
        assert saved["syntax_normalization"]["edits"] == [
            {
                "kind": "extra_root_close",
                "start": len(raw) - 1,
                "end": len(raw),
                "replacement": "",
            }
        ]
        assert saved["received_canonical"] == received
        # The archive never produced the fifth chapter or summaries. Add only
        # temporary fixtures to exercise the normal full-document compiler.
        document = {
            **prefix,
            **part,
            "open_problems": [
                {
                    "sentences": [
                        {
                            "text": "这是用于编译的局限章节夹具。",
                            "fact_card_refs": ["c1"],
                        }
                    ]
                }
            ],
        }
        units = body_units(document)
        summaries = {
            name: [
                {
                    "sentences": [
                        {"text": "这是用于编译的摘要夹具。", "body_unit_refs": [uid]}
                    ]
                }
            ]
            for name, uid in zip(
                ("abstract", "conclusion"), (next(iter(units)), next(reversed(units)))
            )
        }
        report, bindings = compile_document(
            document,
            summaries,
            cards,
            aliases,
            writing_plan=RANGE_WRITING_PLAN,
            envelope=plan["envelope"],
        )
        experiment_units = [
            unit for unit in bindings["units"] if unit["path"][0] == section
        ]
        assert len(experiment_units) == 10
        for rendered in experiment_units:
            _, pi, si = rendered["path"]
            original = part[section][pi]["sentences"][si]
            assert report[rendered["source_start"] : rendered["source_end"]].startswith(
                original["text"]
            )
            assert rendered["evidence_ids"] == [
                cards[aliases[ref]]["evidence_id"] for ref in original["fact_card_refs"]
            ]
        assert all(
            json.loads(raw[unit["text_span"]["start"] : unit["text_span"]["end"]])
            in [
                sentence["text"]
                for paragraph in part[section]
                for sentence in paragraph["sentences"]
            ]
            for unit in saved["wire_bindings"]["units"]
        )
        assert writer_format_status(writer.journal)["recovered_count"] == 1
        assert not writer.state["attempts"]
        resumed = session()
        assert resumed.reference_input(part)[0] == received
        with pytest.raises(WriterError, match="no_replay"):
            await resumed.call("body:" + section, system, payload, schema)
        assert len(sent) == 1
        snapshot = owner._cost_budget.snapshot()
        assert snapshot["requests_sent"] == plan["spent_requests"] + 1
        validate_accounting_snapshot(snapshot, expected_plan=plan)
        tampered = json.loads(writer.journal_path.read_text())
        tampered["calls"]["body:" + section]["syntax_normalization"]["edits"][0][
            "start"
        ] -= 1
        writer.journal_path.write_text(json.dumps(tampered))
        with pytest.raises(WriterError, match="writer_syntax_receipt_conflict"):
            session()
    finally:
        await owner.close()
    assert result["evaluation"]["complete"] is False
    assert all(
        (root / name).read_bytes() == content for name, content in originals.items()
    )


def test_writer_format_projection_reaches_four_exports(display_case):
    root, manifest, state, _ = display_case
    format_status = {
        "policy": LOSSLESS_JSON_POLICY,
        "recovered_count": 1,
        "calls": [
            {
                "call": "body:experiments",
                "stage": "synthesis",
                "status": "recovered",
                "kinds": ["extra_root_close"],
                "reason": None,
            },
            {
                "call": "summary_initial",
                "stage": "summary",
                "status": "unrecoverable",
                "kinds": [],
                "reason": "writer_json_invalid",
            },
        ],
    }
    state["cases"][0]["writer_format"] = format_status
    public, rows, markdown, html = exported(root, state, manifest)
    assert (
        public["cases"][0]["writer_format"]
        == json.loads(rows[0]["writer_format"])
        == format_status
    )
    for value in (
        "extra_root_close",
        "body:experiments",
        "summary_initial",
        "unrecoverable",
    ):
        assert value in markdown and value in html
