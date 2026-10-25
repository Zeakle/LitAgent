"""Verify intact-unit recovery without locally repairing provider JSON."""

import asyncio
import copy
import json

import httpx
import pytest
from test_native_responses import envelope, owner_for, session_for
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.observability.context import reset_task_id, set_task_id
from litagent.writing_v2 import INTACT_UNIT_REEMISSION_CONTRACT as POLICY
from litagent.writing_v2 import (
    WriterError,
    WriterSession,
    body_section_schema,
    body_units,
    intact_unit_footprint,
    scan_native_strings,
    schema_for,
)
from scripts import run_frozen_writing as campaign


@pytest.mark.parametrize("stage", ["body", "summary"])
def test_intact_coverage_survives_outer_eof_and_legal_field_order(case, stage):
    """Strings and references remain bound while serializer field order is free."""
    value = case.body if stage == "body" else case.summaries
    refs = case.aliases if stage == "body" else body_units(case.body)
    schema = schema_for(stage, refs)
    original = json.dumps(value, ensure_ascii=False)
    proof = intact_unit_footprint(original[:-1], schema)
    reordered = json.dumps(dict(reversed(list(value.items()))), sort_keys=True)
    accepted = intact_unit_footprint(reordered, schema)
    assert proof["semantic"] == accepted["semantic"]
    assert proof["coverage_complete"]
    assert proof["raw_sha256"] != accepted["raw_sha256"]
    assert len(proof["units"]) == (5 if stage == "body" else 2)


def test_escaped_json_in_scientific_text_is_not_a_field_or_unknown_reference(case):
    """Only actual record fields receive authority, never text inside quotations."""
    value = {"taxonomy": copy.deepcopy(case.body["taxonomy"])}
    value["taxonomy"][0]["sentences"][0][
        "text"
    ] = 'The paper quotes {"text":"fake","fact_card_refs":["c999"]} as an example.'
    proof = intact_unit_footprint(
        json.dumps(value)[:-1], body_section_schema("taxonomy", case.aliases)
    )
    assert len(proof["units"]) == 1
    assert proof["semantic"]["sections"]["taxonomy"][0][0]["references"] == [
        case.by_eid["b"]
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete_string",
        "unknown_field",
        "unknown_ref",
        "missing_ref",
        "empty",
        "duplicate",
        "bare_text",
        "scalar",
        "no_root",
        "no_paragraph",
    ],
)
def test_uncovered_or_invalid_semantics_never_qualify_for_reemission(case, fault):
    """Outer syntax damage cannot hide omitted values, unknown cards or bad units."""
    value = {"taxonomy": copy.deepcopy(case.body["taxonomy"])}
    unit = value["taxonomy"][0]["sentences"][0]
    if fault == "unknown_field":
        unit["extra"] = "must not disappear"
    elif fault == "unknown_ref":
        unit["fact_card_refs"] = ["c999"]
    elif fault == "missing_ref":
        unit.pop("fact_card_refs")
    elif fault == "empty":
        unit["text"] = ""
    elif fault == "scalar":
        unit["text"] = 42
    elif fault == "no_root":
        value = {"sentences": [unit]}
    elif fault == "no_paragraph":
        value = {"taxonomy": [unit]}
    raw = json.dumps(value)[:-1]
    if fault == "incomplete_string":
        raw = '{"taxonomy":[{"sentences":[{"text":"unfinished'
    elif fault == "duplicate":
        raw = raw.replace('"text":', '"text":"duplicate","text":', 1)
    elif fault == "bare_text":
        raw = "Here is the result: " + raw
    with pytest.raises(WriterError):
        intact_unit_footprint(raw, body_section_schema("taxonomy", case.aliases))


@pytest.mark.parametrize(
    "fault",
    ["text", "ref", "number", "dataset", "direction", "unit", "drop", "paragraph"],
)
def test_content_fingerprint_preserves_conditions_and_exact_citation_ownership(
    case, fault
):
    """A legal response with changed science is still an unauthorized replacement."""
    unit = {
        "text": "On DeiT-S, EViT reduces FLOPs by 30% on ImageNet.",
        "fact_card_refs": [case.by_eid["a"]],
    }
    value = {
        "experiments": [
            {"sentences": [unit]},
            {
                "sentences": [
                    {
                        "text": "Relation networks learn a comparison function.",
                        "fact_card_refs": [case.by_eid["b"]],
                    }
                ]
            },
        ]
    }
    schema = body_section_schema("experiments", case.aliases)
    before = intact_unit_footprint(json.dumps(value)[:-1], schema)
    changed = copy.deepcopy(value)
    target = changed["experiments"][0]["sentences"][0]
    replacements = {
        "text": ("EViT", "Evo-ViT"),
        "number": ("30%", "40%"),
        "dataset": ("ImageNet", "CIFAR-100"),
        "direction": ("reduces", "increases"),
        "unit": ("FLOPs", "latency"),
    }
    if fault in replacements:
        a, b = replacements[fault]
        target["text"] = target["text"].replace(a, b)
    elif fault == "ref":
        target["fact_card_refs"] = [case.by_eid["b"]]
    elif fault == "drop":
        changed["experiments"].pop()
    elif fault == "paragraph":
        changed["experiments"][0]["sentences"].extend(
            changed["experiments"].pop()["sentences"]
        )
    after = intact_unit_footprint(json.dumps(changed), schema)
    assert before["semantic"] != after["semantic"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"text":"a","text":"b"',
        '{"x":"unterminated',
        "```json\n{}\n```",
        '{"a":null}',
        "[" * 65,
        "{}" * 1024 + "{",
    ],
)
def test_scanner_is_bounded_and_rejects_ambiguous_values(raw):
    """No code fences, duplicate keys, incomplete strings or invented coercions."""
    with pytest.raises(WriterError):
        scan_native_strings(raw)


@pytest.mark.parametrize(
    "fault",
    [
        "floating_paragraph",
        "floating_unit",
        "empty_container",
        "extra_array",
        "nested_root",
        "root_array",
        "second_root",
        "nested_title",
    ],
)
def test_layout_ownership_requires_actual_container_ancestry(case, fault):
    """Complete text cannot borrow ownership from a previously seen layout key."""
    unit = copy.deepcopy(case.body["taxonomy"][0]["sentences"][0])
    if fault == "floating_paragraph":
        value = {"taxonomy": [], "sentences": [unit]}
    elif fault == "floating_unit":
        value = {"taxonomy": [{"sentences": []}, unit]}
    elif fault == "empty_container":
        value = {"taxonomy": [{}, {"sentences": [unit]}]}
    elif fault == "extra_array":
        value = {"taxonomy": [[{"sentences": [unit]}]]}
    elif fault == "nested_root":
        value = {"taxonomy": {"taxonomy": [{"sentences": [unit]}]}}
    elif fault == "root_array":
        value = [{"taxonomy": [{"sentences": [unit]}]}]
    elif fault == "second_root":
        raw = json.dumps({"taxonomy": [{"sentences": [unit]}]}) + "{}"
        with pytest.raises(WriterError):
            intact_unit_footprint(raw, body_section_schema("taxonomy", case.aliases))
        return
    else:
        value = {"introduction": [{"title": "Misplaced", "sentences": [unit]}]}
    with pytest.raises(WriterError):
        intact_unit_footprint(
            json.dumps(value),
            body_section_schema(
                "introduction" if fault == "nested_title" else "taxonomy", case.aliases
            ),
        )


@pytest.mark.parametrize(
    "chapter_array,sentence_array",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_singleton_containers_preserve_unambiguous_paragraph_ownership(
    case, chapter_array, sentence_array
):
    """Missing singleton array brackets do not grant any new layout ownership."""
    unit = copy.deepcopy(case.body["taxonomy"][0]["sentences"][0])
    paragraph = {"sentences": [unit] if sentence_array else unit}
    value = {"taxonomy": [paragraph] if chapter_array else paragraph}
    schema = body_section_schema("taxonomy", case.aliases)
    proof = intact_unit_footprint(json.dumps(value)[:-1], schema)
    valid = intact_unit_footprint(
        json.dumps({"taxonomy": [{"sentences": [unit]}]}), schema
    )
    assert proof["semantic"] == valid["semantic"]


@pytest.mark.asyncio
@pytest.mark.parametrize("floating", ["paragraph", "unit"])
async def test_misplaced_units_fail_before_reemission_request(case, tmp_path, floating):
    """No shared attempt or extra request may be spent assigning invented roles."""
    sent = []

    def handler(request):
        wire = json.loads(request.content)
        data = json.loads(wire["input"][1]["content"])
        sent.append(data)
        assert "shape_reemission" not in data
        section = data["requested_body_section"]
        if section == "introduction":
            value = {k: copy.deepcopy(case.body[k]) for k in ("title", section)}
        else:
            assert section == "taxonomy"
            unit = copy.deepcopy(case.body[section][0]["sentences"][0])
            value = (
                {"taxonomy": [], "sentences": [unit]}
                if floating == "paragraph"
                else {"taxonomy": [{"sentences": []}, unit]}
            )
        return httpx.Response(200, json=envelope(json.dumps(value)))

    owner, budget, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path, reemission_contract=POLICY)
    token = set_task_id("synthesis")
    try:
        with pytest.raises(WriterError, match="unassigned"):
            await session.initial_body()
        assert len(sent) == 2 and budget.used == 240
        assert not session.state["attempts"]
        assert list(session.journal["body_parts"]) == ["introduction"]
        assert session.journal["calls"]["body:taxonomy"]["status"] == "failed_no_replay"
    finally:
        reset_task_id(token)
        await owner.close()


def test_v1_reemission_is_read_only_and_cannot_resume_as_v2(case, tmp_path):
    """Historical profile descriptions preserve their actual contract version."""
    from litagent.writing_v2 import LEGACY_INTACT_UNIT_REEMISSION_CONTRACT as OLD_POLICY

    assert POLICY == "intact-unit-reemission-v2"
    assert (
        campaign.profile_description(campaign.LEGACY_REEMISSION_PROFILE, {})[
            "reemission_contract"
        ]
        == OLD_POLICY
    )
    assert (
        campaign.profile_parameters(campaign.LEGACY_REEMISSION_PROFILE)[
            "native_reemission_contract"
        ]
        == OLD_POLICY
    )
    with pytest.raises(ValueError, match="read_only"):
        campaign.check_profile(None, campaign.LEGACY_REEMISSION_PROFILE)
    with pytest.raises(WriterError, match="configuration"):
        WriterSession(
            Provider(case),
            tmp_path / "old.json",
            "old",
            case.packet,
            case.aliases,
            reemission_contract=OLD_POLICY,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "text", "ref", "repeat", "unknown_usage", "cancel"]
)
async def test_body_and_summary_use_one_ledger_and_require_new_received_json(
    case, tmp_path, fault
):
    """The exact authoring payload survives SDK requests and both recovery stages."""
    sent = []

    def handler(request):
        wire = json.loads(request.content)
        data = json.loads(wire["input"][1]["content"])
        sent.append(data)
        correction = "shape_reemission" in data
        if "body_units" in data:
            value = copy.deepcopy(case.summaries)
        else:
            value = {
                k: copy.deepcopy(case.body[k])
                for k in wire["text"]["format"]["schema"]["properties"]
            }
        if correction:
            assert data["shape_reemission"]["contract"] == POLICY
            assert isinstance(data["rejected_response"], str)
            assert wire["max_output_tokens"] == 20000
            if fault == "cancel":
                raise asyncio.CancelledError()
            if fault == "text":
                value["taxonomy"][0]["sentences"][0]["text"] += " Unauthorized."
            if fault == "ref":
                value["taxonomy"][0]["sentences"][0]["fact_card_refs"] = [
                    case.by_eid["a"]
                ]
        raw = json.dumps(value)
        if (
            not correction
            and (
                data.get("requested_body_section") == "taxonomy" or "body_units" in data
            )
        ) or (correction and fault == "repeat"):
            raw = raw[:-1]
        response = envelope(raw)
        if correction and fault == "unknown_usage":
            response.pop("usage")
        return httpx.Response(200, json=response)

    owner, budget, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path, reemission_contract=POLICY)
    token = set_task_id("synthesis")
    try:
        if fault is None:
            body = await session.initial_body()
            summary = await session.summaries(body, "summary_initial")
            assert body == case.body and summary == case.summaries and len(sent) == 8
            assert len(session.state["attempts"]) == 2
            assert (
                session.journal["calls"]["body:taxonomy"]["resolved_by"] == "repair_0"
            )
            assert (
                session.journal["calls"]["summary_initial"]["resolved_by"] == "repair_1"
            )
            assert "failure" not in session.journal
        else:
            with pytest.raises((ValueError, asyncio.CancelledError)):
                await session.initial_body()
            assert len(sent) == 3 and len(session.state["attempts"]) == 1
            assert list(session.journal["body_parts"]) == ["introduction"]
        assert all(
            v["stage"] == "native_initial" and v["kind"] == POLICY
            for v in session.state["attempts"]
        )
        if fault not in {"unknown_usage", "cancel"}:
            assert (
                budget.used == 120 * len(sent)
                and not budget.snapshot()["usage_uncertain"]
            )
            assert budget.snapshot()["phase_usage"] == [0, budget.used, 0]
        else:
            assert budget.snapshot()["usage_uncertain"] and budget.used > 240
        before = session.journal_path.read_bytes()
        restarted = session_for(owner, case, tmp_path, reemission_contract=POLICY)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.initial_body()
        assert session.journal_path.read_bytes() == before
    finally:
        reset_task_id(token)
        await owner.close()


@pytest.mark.asyncio
async def test_revised_summary_requires_full_reassessment_capacity_before_reemission(
    case, tmp_path
):
    """A small body cannot be assumed to contain just one assertion per unit."""
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=envelope(json.dumps(case.summaries)[:-1]))

    owner, budget, _ = await owner_for(handler, allowance=50000)
    session = session_for(owner, case, tmp_path, reemission_contract=POLICY)
    token = set_task_id("evidence_rewrite")
    try:
        with pytest.raises(WriterError, match="transaction_budget"):
            await session.summaries(case.body, "summary_after_patch")
        assert len(sent) == 1 and not session.state["attempts"] and budget.used == 120
        assert session.journal["failure"]["request_sent"] is False
        assert (
            session.journal["failure"]["diagnostics"][0]["projected_scientific_batches"]
            == 8
        )
    finally:
        reset_task_id(token)
        await owner.close()


def test_policy_is_versioned_server_only_and_legacy_identity_is_unchanged(
    case, tmp_path
):
    """Unknown policy versions cannot grant requests or reuse older checkpoints."""
    owner = Provider(case)
    path = tmp_path / "r.json"
    old = WriterSession(owner, path, "r", case.packet, case.aliases)
    explicit = WriterSession(
        owner, path, "r", case.packet, case.aliases, reemission_contract=None
    )
    assert old.identity == explicit.identity
    old.save()
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            owner, path, "r", case.packet, case.aliases, reemission_contract=POLICY
        )
    with pytest.raises(WriterError, match="configuration"):
        WriterSession(
            owner,
            tmp_path / "bad.json",
            "r",
            case.packet,
            case.aliases,
            reemission_contract="unknown",
        )
    with pytest.raises(WriterError, match="configuration"):
        WriterSession(
            owner,
            tmp_path / "mix.json",
            "r",
            case.packet,
            case.aliases,
            shape_recovery=True,
            reemission_contract=POLICY,
        )
    assert (
        campaign.profile_parameters(campaign.REEMISSION_PROFILE)[
            "native_reemission_contract"
        ]
        == POLICY
    )
    assert "native_reemission_contract" not in campaign.profile_parameters(
        campaign.SHAPE_PROFILE
    )


@pytest.mark.parametrize("tamper", [False, True])
def test_prepared_profile_rejects_rehashed_policy_drift(manifest, tamper):
    """Frozen identity cannot be used to grant another recovery mechanism."""
    from litagent.report_audit import atomic_json, digest

    root, config = manifest
    target = root.with_name("intact-reemission-suite")
    campaign.prepare(target, config, writing_profile=campaign.REEMISSION_PROFILE)
    value = campaign.validate(target, config)
    assert value["writing_parameters"]["native_reemission_contract"] == POLICY
    assert value["accounting_policy"] == campaign.CONSERVATIVE_ACCOUNTING
    if tamper:
        value["writing_parameters"]["native_reemission_contract"] = "unknown"
        value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
        atomic_json(target / "manifest.json", value)
        with pytest.raises(ValueError, match="parameters_conflict"):
            campaign.validate(target, config)
