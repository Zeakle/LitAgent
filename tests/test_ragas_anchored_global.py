"""Exercise cross-unit findings without loosening the report-quality gate."""

import copy
import json
from pathlib import Path

import pytest
import pytest_asyncio
from jsonschema import Draft202012Validator
from test_native_ragas_evaluation import SemanticTransport, native_case  # noqa: F401

from litagent.config import load_config
from litagent.eval.batched import normalize_global_findings
from litagent.eval.engine_policy import (
    ANCHORED_GLOBAL_REVIEW,
    ARCHIVED_ECHO_ENGINES,
    ARCHIVED_ENGINE_IDENTITY,
    ARCHIVED_FLAT_ENGINES,
    ARCHIVED_LIST_ENGINES,
    ARCHIVED_LONG_ANCHOR_ENGINES,
    ARCHIVED_PREFIXED_ENGINE_IDENTITY,
    KEYED_ASSESSMENT,
    KEYED_GLOBAL_ANCHORS,
    KEYED_GLOBAL_QUOTED_OMISSIONS,
    KEYED_GLOBAL_UNITS,
    configured_engine,
    validate_engine,
)
from litagent.eval.ragas_global_wire import (
    UNIT_CONTRACT,
    pack_global_input,
    unpack_global_input,
)
from litagent.eval.ragas_native import (
    canonical_assertions,
    global_review_request,
    parse_native_global_response,
    source_samples,
)
from litagent.eval.ragas_repair import VERSION, native_repair_diagnostics
from litagent.eval.responsibilities import parse_global_response
from litagent.report_render import build_report_source_units
from litagent.safety.budget import strict_request_reservation
from litagent.writing_v2 import current_repair_guidance_context, repair_targets


@pytest_asyncio.fixture
async def global_case(native_case):
    """Build the actual schema and well-formed global regions from two source units."""
    survey, context = native_case
    _, units, _, samples = source_samples(survey, context)
    for sample in samples:
        sample["statements"] = [sample["response"]]
        sample["verdicts"] = [
            {"statement": sample["response"], "verdict": 1, "reason": "supported"}
        ]
    assertions, _ = canonical_assertions(samples)
    cfg = load_config("config/agent-format-repair.yaml")
    payload, messages, fmt = global_review_request(
        survey, context, cfg.eval, units, assertions, force_compact=True, anchored=True
    )
    response = await SemanticTransport().chat(messages)
    return payload, messages, fmt, json.loads(response.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefixed", [False, True])
async def test_known_anchors_preserve_cross_unit_repetition(global_case, prefixed):
    payload, messages, fmt, raw = global_case
    assert unpack_global_input(json.loads(messages[1]["content"])) == payload
    assert "minimum sufficient supported scientific claims" in messages[0]["content"]
    assert "Do not add prefixes" in messages[0]["content"]
    claims = payload["assertions"][:2]
    assert claims[0]["source_unit_id"] != claims[1]["source_unit_id"]
    raw["issues"] = [
        {
            "code": "repetition",
            "anchor_ids": [
                ("claim:" if prefixed else "") + claim["claim_id"] for claim in claims
            ],
            "explanation": "The two paragraphs repeat this comparison.",
        }
    ]
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(raw)
    progress = {}
    normalized = parse_native_global_response(json.dumps(raw), payload, progress)
    assert [i.owner for i in normalized.issues] == ["report", "report"]
    assert [(i.source_unit_id, i.claim_ids) for i in normalized.issues] == [
        (claim["source_unit_id"], [claim["claim_id"]]) for claim in claims
    ]
    details = {
        "evaluation_version": VERSION,
        "stages": {
            **progress,
            "global_check": "completed",
            "global_findings_contract": ANCHORED_GLOBAL_REVIEW,
        },
        "inventory_check": {"status": "passed"},
        "assertions": payload["assertions"],
        "unsupported_claims": [],
        "global_issues": [
            {**issue.model_dump(), "owner": issue.owner} for issue in normalized.issues
        ],
    }
    original = copy.deepcopy(details)
    diagnostics = native_repair_diagnostics(details)
    assert [d["claim_id"] for d in diagnostics] == [c["claim_id"] for c in claims]
    assert all(
        d["preserve_supported_claim"] and d["status"] == "supported"
        for d in diagnostics
    )
    assert details == original
    details["global_issues"][1]["source_unit_id"] = claims[0]["source_unit_id"]
    with pytest.raises(ValueError, match="ragas_recovery_diagnostic_invalid"):
        native_repair_diagnostics(details)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", ["unknown", "duplicate", "mixed_duplicate", "omission_claim"]
)
async def test_anchored_findings_fail_closed(global_case, error):
    payload, _, _, raw = global_case
    claim = "claim:" + payload["assertions"][0]["claim_id"]
    raw["issues"] = [
        {
            "code": "missing_assertion" if error == "omission_claim" else "repetition",
            "anchor_ids": {
                "unknown": ["claim:invented"],
                "duplicate": [claim, claim],
                "mixed_duplicate": [claim, claim.removeprefix("claim:")],
                "omission_claim": [claim.removeprefix("claim:")],
            }[error],
            "explanation": "Inspect the original units.",
        }
    ]
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_native_global_response(json.dumps(raw), payload, {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('{"issues":[],"issues":[]}', "ragas_duplicate_response_key"),
        ('{"issues":[],"other":NaN}', "ragas_nonfinite_json"),
    ],
)
async def test_native_alias_mapping_retains_strict_json(global_case, content, reason):
    payload, _, _, _ = global_case
    with pytest.raises(ValueError, match=reason):
        parse_native_global_response(content, payload, {})


@pytest.mark.asyncio
async def test_keyed_units_follow_current_report_and_preserve_omission(native_case):
    survey, context = native_case
    cfg = load_config("config/agent-format-repair.yaml")

    def request(draft):
        _, units, _, samples = source_samples(draft, context)
        for sample in samples:
            sample["statements"] = [sample["response"]]
            sample["verdicts"] = [
                {"statement": sample["response"], "verdict": 1, "reason": "fixture"}
            ]
        assertions, _ = canonical_assertions(samples)
        return global_review_request(
            draft,
            context,
            cfg.eval,
            units,
            assertions,
            force_compact=True,
            anchored=True,
            unit_view=True,
            unit_response_contract=KEYED_GLOBAL_ANCHORS,
        )

    payload, messages, fmt = request(survey)
    assert unpack_global_input(json.loads(messages[1]["content"])) == payload
    response = await SemanticTransport().chat(messages)
    raw = json.loads(response.content)
    schema = Draft202012Validator(fmt["json_schema"]["schema"])
    schema.validate(raw)
    keys = [unit["coverage_key"] for unit in payload["source_units"]]
    ids = [unit["source_unit_id"] for unit in payload["source_units"]]
    raw["units"] = dict(reversed(list(raw["units"].items())))
    raw["units"][keys[0]] = {
        "complete": False,
        "explanation": "Missing labeled examples.",
    }
    raw["issues"] = [
        {
            "code": "missing_assertion",
            "anchor_ids": [keys[0]],
            "explanation": "Missing labeled examples.",
        }
    ]
    reviewed = parse_native_global_response(json.dumps(raw), payload, {})
    assert [unit.source_unit_id for unit in reviewed.units] == ids
    assert not reviewed.units[0].complete and reviewed.issues[0].owner == "inventory"
    assert reviewed.issues[0].source_unit_id == ids[0]

    amended, changed_messages, changed_fmt = request(
        survey.replace(
            "The benchmark includes", "A separate fact.\n\nThe benchmark includes"
        )
    )
    changed_keys = [unit["coverage_key"] for unit in amended["source_units"]]
    assert len(changed_keys) == len(keys) + 1
    assert [unit["source_unit_id"] for unit in amended["source_units"]][3:] != ids[3:]
    assert (
        changed_fmt["json_schema"]["schema"]["properties"]["units"]["required"]
        == changed_keys
    )
    amended_response = await SemanticTransport().chat(changed_messages)
    changed_raw = json.loads(amended_response.content)
    changed_raw["issues"] = [{**raw["issues"][0], "anchor_ids": [keys[-1]]}]
    changed_raw["units"][keys[-1]]["complete"] = False
    reviewed = parse_native_global_response(json.dumps(changed_raw), amended, {})
    assert len(reviewed.units) == len(changed_keys)
    assert (
        reviewed.issues[0].source_unit_id
        == amended["source_units"][len(keys) - 1]["source_unit_id"]
    )
    assert reviewed.issues[0].source_unit_id != ids[-1]
    with pytest.raises(ValueError, match="evaluation_global_unit_coverage_invalid"):
        parse_native_global_response(json.dumps(raw), amended, {})

    unknown = copy.deepcopy(raw)
    unknown["units"]["u999"] = unknown["units"].pop(keys[-1])
    assert not schema.is_valid(unknown)
    progress = {}
    with pytest.raises(ValueError, match="evaluation_global_unit_coverage_invalid"):
        parse_native_global_response(json.dumps(unknown), payload, progress)
    assert progress["global_validation_error"] == {
        "field": f"units.{keys[-1]}",
        "reason": "missing_key",
    }
    duplicate = json.dumps(raw).replace(
        '"units": {',
        f'"units": {{"{keys[0]}": {json.dumps(raw["units"][keys[0]])}, ',
        1,
    )
    with pytest.raises(ValueError, match="ragas_duplicate_response_key"):
        parse_native_global_response(duplicate, payload, {})

    first_claim = payload["assertions"][0]["claim_id"]
    for code, anchors in [
        ("missing_assertion", [keys[0]]),
        ("repetition", [first_claim]),
        ("category_count", [keys[0], first_claim]),
    ]:
        raw["issues"] = [
            {"code": code, "anchor_ids": anchors, "explanation": "fixture"}
        ]
        schema.validate(raw)
        reviewed = parse_native_global_response(json.dumps(raw), payload, {})
        assert reviewed.issues[0].code == code
    for code, anchors, offset, reason in [
        ("missing_assertion", ["u999"], "[0]", "unknown_or_wrong_role"),
        ("missing_assertion", [first_claim], "[0]", "unknown_or_wrong_role"),
        ("missing_assertion", [ids[0]], "[0]", "unknown_or_wrong_role"),
        ("missing_assertion", [keys[0], keys[0]], "[1]", "duplicate_key"),
        ("repetition", [keys[0]], "[0]", "unknown_or_wrong_role"),
        ("category_count", [], "", "missing_or_invalid_keys"),
        ("category_count", [4], "[0]", "unknown_or_wrong_role"),
    ]:
        raw["issues"] = [
            {"code": code, "anchor_ids": anchors, "explanation": "fixture"}
        ]
        if reason != "duplicate_key":
            assert not schema.is_valid(raw)
        progress = {}
        with pytest.raises(
            ValueError, match="evaluation_global_finding_identity_invalid"
        ):
            parse_native_global_response(json.dumps(raw), payload, progress)
        assert progress["global_validation_error"] == {
            "field": f"issues[0].anchor_ids{offset}",
            "reason": reason,
        }


def test_394_short_anchors_restore_exact_locations_and_fit_budget():
    """Keep the real invalid v1 reply and test a separate valid v2 wire."""
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    path = artifacts / "acceptance/workbench-vit-386-20261110/runs/83241b86baca.json"
    if not path.exists():
        pytest.skip("requires the local immutable 394 archive")
    archive = json.loads(path.read_bytes())
    node = archive["nodes"]["llm:88e96013e7fa4f94ae4c6eda958350ee"]
    old = unpack_global_input(json.loads(node["input"]["messages"][1]["content"]))
    original = json.loads(node["output"])
    assert len(original["units"]) == 46
    assert [issue["anchor_ids"] for issue in original["issues"][-2:]] == [
        ["unit:38:16698"],
        ["unit:39:16993"],
    ]
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_native_global_response(node["output"], old, {})
    survey = archive["report"]["survey"]
    by_id = {unit.unit_id: unit for unit in build_report_source_units(survey)}
    config = load_config("config/agent-format-repair.yaml")
    payload, messages, fmt = global_review_request(
        survey,
        old,
        config.eval,
        [by_id[unit["source_unit_id"]] for unit in old["source_units"]],
        old["assertions"],
        anchored=True,
        force_compact=True,
        unit_response_contract=KEYED_GLOBAL_ANCHORS,
    )
    assert unpack_global_input(json.loads(messages[1]["content"])) == payload
    assert payload["assertions"] == old["assertions"]
    assert (
        sum(claim["status"] == "unsupported" for claim in payload["assertions"]) == 44
    )
    fixture = copy.deepcopy(original)
    # This is an independent selection from the new directory, not a migration
    # or reinterpretation of the archived provider's unknown coordinates.
    fixture["issues"] = [
        {
            "code": "missing_assertion",
            "anchor_ids": [key],
            "explanation": "Offline fixture.",
        }
        for key in ("u39", "u40")
    ]
    for key in ("u39", "u40"):
        fixture["units"][key]["complete"] = False
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(fixture)
    reviewed = parse_native_global_response(json.dumps(fixture), payload, {})
    assert [issue.source_unit_id for issue in reviewed.issues] == [
        "unit:38:16993",
        "unit:39:17351",
    ]
    assert all(issue.owner == "inventory" for issue in reviewed.issues)
    assert strict_request_reservation(messages, config.eval.max_tokens, fmt) <= int(
        config.eval.context_tokens * (1 - config.eval.strict_context_margin)
    )


def test_390_duplicate_rows_cannot_satisfy_keyed_coverage():
    """Replay the actual 45-row response and its missing host identities."""
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    path = artifacts / "acceptance/workbench-vit-386-20261110/runs/decc5357575f.json"
    if not path.exists():
        pytest.skip("requires the local immutable 390 archive")
    archive = json.loads(path.read_bytes())
    node = [
        node
        for node in archive["nodes"].values()
        if node.get("task_id") == "evaluation.post_rewrite"
        and node.get("kind") == "llm"
    ][-1]
    old = unpack_global_input(json.loads(node["input"]["messages"][1]["content"]))
    response = json.loads(node["output"])
    with pytest.raises(ValueError, match="evaluation_global_unit_coverage_invalid"):
        parse_native_global_response(node["output"], old, {})
    journal = json.loads(
        (artifacts / "targeted-repairs/decc5357575f.json.writer-v2.json").read_bytes()
    )
    draft = journal["candidates"][0]["survey"]
    by_id = {unit.unit_id: unit for unit in build_report_source_units(draft)}
    config = load_config("config/agent-format-repair.yaml")
    payload, messages, fmt = global_review_request(
        draft,
        old,
        config.eval,
        [by_id[unit["source_unit_id"]] for unit in old["source_units"]],
        old["assertions"],
        anchored=True,
        force_compact=True,
        unit_response_contract=KEYED_GLOBAL_UNITS,
    )
    keys = {
        unit["source_unit_id"]: unit["coverage_key"] for unit in payload["source_units"]
    }
    assert len(response["units"]) == len(keys) == 45
    response["units"] = {
        keys[row["source_unit_id"]]: {
            k: v for k, v in row.items() if k != "source_unit_id"
        }
        for row in response["units"]
    }
    assert set(keys.values()) - response["units"].keys() == {"u43", "u44", "u45"}
    assert not Draft202012Validator(fmt["json_schema"]["schema"]).is_valid(response)
    with pytest.raises(ValueError, match="evaluation_global_unit_coverage_invalid"):
        parse_native_global_response(json.dumps(response), payload, {})
    assert strict_request_reservation(messages, config.eval.max_tokens, fmt) <= int(
        config.eval.context_tokens * (1 - config.eval.strict_context_margin)
    )


def test_archived_engine_is_readable_without_admitting_new_requests():
    configured = configured_engine(load_config("config/agent-format-repair.yaml"))
    assert configured["global_findings_contract"] == ANCHORED_GLOBAL_REVIEW
    assert configured["global_unit_response_contract"] == KEYED_GLOBAL_QUOTED_OMISSIONS
    assert configured["assessment_response_contract"] == KEYED_ASSESSMENT
    assert validate_engine(configured) == configured
    echo_descriptor = {
        k: v for k, v in configured.items() if k != "assessment_response_contract"
    }
    echo_descriptor["global_unit_response_contract"] = KEYED_GLOBAL_UNITS
    echo_descriptor["identity"] = ARCHIVED_ECHO_ENGINES[
        ANCHORED_GLOBAL_REVIEW, None, KEYED_GLOBAL_UNITS
    ]
    assert validate_engine(echo_descriptor, archived=True) == echo_descriptor
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(echo_descriptor)
    v1 = {
        **configured,
        "global_unit_response_contract": KEYED_GLOBAL_UNITS,
        "identity": ARCHIVED_LONG_ANCHOR_ENGINES[
            ANCHORED_GLOBAL_REVIEW, None, KEYED_GLOBAL_UNITS
        ][1],
    }
    assert validate_engine(v1, archived=True) == v1
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(v1)
    list_descriptor = {
        k: v for k, v in echo_descriptor.items() if k != "global_unit_response_contract"
    }
    archived = {
        k: v for k, v in list_descriptor.items() if k != "global_findings_contract"
    }
    archived["identity"] = ARCHIVED_ENGINE_IDENTITY
    assert validate_engine(archived, archived=True) == archived
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(archived)
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(
            {**archived, "global_findings_contract": ANCHORED_GLOBAL_REVIEW},
            archived=True,
        )
    for identity in (
        ARCHIVED_PREFIXED_ENGINE_IDENTITY,
        ARCHIVED_LIST_ENGINES[ANCHORED_GLOBAL_REVIEW, None],
    ):
        previous = {**list_descriptor, "identity": identity}
        assert validate_engine(previous, archived=True) == previous
        with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
            validate_engine(previous)
        with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
            validate_engine({**configured, "identity": identity}, archived=True)


def test_unit_wire_is_explicit_and_preserves_old_read_only_identity():
    cfg = load_config("config/agent-format-repair.yaml")
    assert "native_global_input" not in cfg.eval.model_dump()
    cfg.eval = cfg.eval.model_copy(update={"native_global_unit_response": None})
    old = configured_engine(cfg)
    cfg.eval = cfg.eval.model_copy(update={"native_global_input": UNIT_CONTRACT})
    selected = configured_engine(cfg)
    assert selected["global_input_contract"] == UNIT_CONTRACT
    assert selected["identity"] != old["identity"]
    assert validate_engine(selected) == selected
    archived_unit = {
        **{k: v for k, v in selected.items() if k != "assessment_response_contract"},
        "identity": ARCHIVED_LIST_ENGINES[ANCHORED_GLOBAL_REVIEW, UNIT_CONTRACT],
    }
    assert validate_engine(archived_unit, archived=True) == archived_unit
    previous = {
        **{k: v for k, v in old.items() if k != "assessment_response_contract"},
        "identity": ARCHIVED_FLAT_ENGINES[ANCHORED_GLOBAL_REVIEW],
    }
    assert validate_engine(previous, archived=True) == previous
    for value in (previous, {**previous, "global_input_contract": UNIT_CONTRACT}):
        with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
            validate_engine(value)
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(
            {**previous, "global_input_contract": UNIT_CONTRACT}, archived=True
        )


def test_unit_projection_preserves_interleaving_empty_units_and_negative_statuses(
    global_case,
):
    payload = copy.deepcopy(global_case[0])
    first, second = payload["assertions"][:2]
    end = {**first, "claim_id": "last", "status": "unsupported"}
    payload["assertions"] = [{**first, "status": "unverifiable"}, second, end]
    for unit in payload["source_units"]:
        unit["extracted_claim_ids"] = [
            row["claim_id"]
            for row in payload["assertions"]
            if row["source_unit_id"] == unit["source_unit_id"]
        ]
    wire = pack_global_input(payload, by_unit=True)
    assert wire["global_input_contract"] == UNIT_CONTRACT
    assert "assertions" not in wire
    assert wire["claim_order"] == [first["claim_id"], second["claim_id"], "last"]
    assert any(unit["claims"] == [] for unit in wire["source_units"])
    assert unpack_global_input(wire) == payload

    misplaced, duplicate, unknown, reordered = (copy.deepcopy(wire) for _ in range(4))
    source, other = (u for u in misplaced["source_units"] if u["claims"])
    other["claims"].append(source["claims"].pop())
    duplicate["claim_order"][-1] = first["claim_id"]
    unknown["claim_order"][-1] = "unknown"
    next(u for u in reordered["source_units"] if len(u["claims"]) == 2)[
        "claims"
    ].reverse()
    for changed in (misplaced, duplicate, unknown, reordered):
        with pytest.raises(ValueError, match="ragas_global_unit_projection_invalid"):
            unpack_global_input(changed)

    for changed in (
        {**end, "claim_id": first["claim_id"]},
        {**end, "source_unit_id": "invented"},
    ):
        with pytest.raises(ValueError, match="ragas_global_unit_projection_invalid"):
            pack_global_input(
                {**payload, "assertions": [first, second, changed]}, by_unit=True
            )


@pytest.mark.asyncio
async def test_unit_view_preserves_a_real_omission_from_an_empty_inventory(native_case):
    survey, context = native_case
    _, units, _, samples = source_samples(survey, context)
    missing = samples[0]["source_unit_id"]
    for sample in samples[1:]:
        sample["statements"] = [sample["response"]]
    assertions, _ = canonical_assertions(samples)
    config = load_config("config/agent-format-repair.yaml")
    payload, messages, _ = global_review_request(
        survey, context, config.eval, units, assertions, anchored=True, unit_view=True
    )
    wire = json.loads(messages[1]["content"])
    original = next(u for u in wire["source_units"] if u["source_unit_id"] == missing)
    assert original["text"] and original["claims"] == []
    response = await SemanticTransport().chat(messages)
    raw = json.loads(response.content)
    row = next(u for u in raw["units"] if u["source_unit_id"] == missing)
    row.update(
        complete=False, explanation="The original factual sentence was not extracted."
    )
    raw["issues"] = [
        {
            "code": "missing_assertion",
            "anchor_ids": [missing],
            "explanation": "No local assertion covers the labeled examples requirement.",
        }
    ]
    review = parse_native_global_response(json.dumps(raw), payload, {})
    (omission,) = review.issues
    assert omission.owner == "inventory" and omission.source_unit_id == missing
    assert not next(
        unit.complete for unit in review.units if unit.source_unit_id == missing
    )


def test_377_unit_view_keeps_full_input_and_the_original_omission_verdict():
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/native-repair-trial-20261110"
    )
    if not (root / "live-exit.json").exists():
        pytest.skip("requires the settled 377 archive")
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    (start,) = [
        e["data"]
        for e in events
        if e["event"] == "llm.start"
        and (e["data"].get("response_format") or {}).get("json_schema", {}).get("name")
        == "ragas_report_review"
    ]
    (response,) = [
        e["data"]["content"]
        for e in events
        if e["event"] == "llm.complete"
        and e["data"]["operation_id"] == start["operation_id"]
    ]
    (candidate,) = json.loads((root / "repair.json.writer-v2.json").read_bytes())[
        "candidates"
    ]
    original = unpack_global_input(json.loads(start["messages"][1]["content"]))
    survey = candidate["survey"]
    units = {unit.unit_id: unit for unit in build_report_source_units(survey)}
    config = load_config("config/agent-format-repair.yaml")
    payload, messages, fmt = global_review_request(
        survey,
        original,
        config.eval,
        [units[unit["source_unit_id"]] for unit in original["source_units"]],
        original["assertions"],
        anchored=True,
        unit_view=True,
    )
    wire = json.loads(messages[1]["content"])
    assert wire["global_input_contract"] == UNIT_CONTRACT
    assert unpack_global_input(wire) == payload == original
    assert len(payload["assertions"]) == 339 and len(wire["source_units"]) == 46
    unit = next(
        u for u in wire["source_units"] if u["source_unit_id"] == "unit:12:6690"
    )
    claims = {row[1]: row[2] for row in unit["claims"]}
    assert (
        "Matching Networks" in unit["text"]
        and "Matching Networks" in claims["claim_67"]
    )
    assert "骨干" in claims["claim_70"]
    assert "EVERY local claim" in messages[0]["content"]
    plan = json.loads((root / "ledger.json").read_bytes())["plan"]
    assert strict_request_reservation(messages, config.eval.max_tokens, fmt) <= min(
        plan["stages"]["global"]["request_cap"], plan["context_reservation_limit"]
    )
    (issue,) = parse_native_global_response(response, payload, {}).issues
    assert (
        issue.code == "missing_assertion"
        and issue.source_unit_id == unit["source_unit_id"]
    )
    details = candidate["evaluation"]["faithfulness"]["details"]
    assert (
        details["inventory_check"]["status"] == "failed" and not candidate["committed"]
    )
    issues = native_repair_diagnostics(details)
    assert {x["claim_id"] for x in issues} == {
        "claim_105",
        "claim_120",
        "claim_148",
        "claim_163",
        "claim_191",
        "claim_198",
        "claim_210",
        "claim_211",
        "claim_216",
    }
    document = candidate["writer_document"]
    targets = repair_targets(
        candidate["bindings"],
        issues,
        document["body"],
        document["summaries"],
        merge=True,
    )
    guidance = current_repair_guidance_context(
        survey,
        candidate["bindings"],
        document,
        targets,
        issues,
        diagnostic_categories=["evaluation"] * len(issues),
    )
    assert len(targets) == 8 and guidance["diagnostics"] == issues
    assert all(
        set(issue["target_ids"]) <= targets.keys() and issue["target_ids"]
        for issue in guidance["host_directory"]["issues"]
    )


def _archived_global_response(run_name):
    root = Path(__file__).resolve().parents[1] / "artifacts/acceptance" / run_name
    path = root / "progress-archive.json"
    if not path.exists():
        pytest.skip(f"requires the local immutable archive {run_name}")
    events = json.loads(path.read_bytes())["events"]
    review_starts = {
        event["data"]["operation_id"]: event["data"]
        for event in events
        if event["event"] == "llm.start"
        and (event["data"].get("response_format") or {})
        .get("json_schema", {})
        .get("name")
        == "ragas_report_review"
    }
    start, response = [
        (review_starts[e["data"]["operation_id"]], e["data"]["content"])
        for e in events
        if e["event"] == "llm.complete" and e["data"]["operation_id"] in review_starts
    ][-1]
    return (
        root,
        unpack_global_input(json.loads(start["messages"][1]["content"])),
        response,
    )


def test_078_replay_preserves_all_reported_omissions():
    root, payload, response = _archived_global_response(
        "ragas-native-single-20261110-078"
    )
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_global_response(response, payload, {})
    result = parse_native_global_response(response, payload, {})
    original = json.loads(response)["issues"]
    assert len(result.issues) == len(original) == 12
    complete_units = {unit.source_unit_id for unit in result.units if unit.complete}
    for finding, raw in zip(result.issues, original, strict=True):
        assert finding.owner == "inventory"
        assert finding.code == "missing_assertion"
        assert [finding.source_unit_id] == raw["anchor_ids"]
        assert finding.claim_ids == []
        assert finding.explanation == raw["explanation"]
        assert finding.source_unit_id in complete_units
    report = json.loads((root / "report.json").read_bytes())
    assert report["delivery"]["publishable"] is False
    assert report["evaluation"]["faithfulness"]["details"]["unsupported_count"] == 34


def test_077_cross_unit_findings_replay_preserves_all_negative_claims():
    """Replay the original error and adapt its same findings to the new wire."""
    root, old, response = _archived_global_response("ragas-native-single-20261110-077")
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_global_response(response, old, {})
    journal = json.loads((root / "repair.json.writer-v2.json").read_bytes())
    candidate = journal["candidates"][-1]
    survey = candidate["survey"]
    directory = {u.unit_id: u for u in build_report_source_units(survey)}
    cfg = load_config("config/agent-format-repair.yaml")
    payload, messages, fmt = global_review_request(
        survey,
        old,
        cfg.eval,
        [directory[u["source_unit_id"]] for u in old["source_units"]],
        old["assertions"],
        force_compact=True,
        anchored=True,
    )
    assert payload == {**old, "global_findings_contract": ANCHORED_GLOBAL_REVIEW}
    raw = json.loads(response)
    converted = copy.deepcopy(raw)
    converted["issues"] = [
        {
            "code": finding["code"],
            "anchor_ids": ["claim:" + cid for cid in finding["claim_ids"]]
            or ["unit:" + finding["source_unit_id"]],
            "explanation": finding["explanation"],
        }
        for finding in raw["issues"]
    ]
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(converted)
    progress = {}
    result = normalize_global_findings(
        parse_global_response(json.dumps(converted), payload, progress), payload
    )
    assert len(result.issues) == 5
    assert sum(len(i.claim_ids) for i in result.issues) == sum(
        len(i["claim_ids"]) for i in raw["issues"]
    )
    assert {i.explanation for i in result.issues} == {
        i["explanation"] for i in raw["issues"]
    }
    details = candidate["evaluation"]["faithfulness"]["details"]
    facts = {key: details[key] for key in ("evaluation_version", "assertions")}
    diagnostics = native_repair_diagnostics(
        {
            **facts,
            "stages": {
                **progress,
                "global_check": "completed",
                "global_findings_contract": ANCHORED_GLOBAL_REVIEW,
            },
            "inventory_check": {"status": "passed"},
            "global_issues": [
                {**i.model_dump(), "owner": i.owner} for i in result.issues
            ],
        }
    )
    document = candidate["writer_document"]
    targets = repair_targets(
        candidate["bindings"],
        diagnostics,
        document["body"],
        document["summaries"],
        merge=True,
    )
    guidance = current_repair_guidance_context(
        survey,
        candidate["bindings"],
        document,
        targets,
        diagnostics,
        diagnostic_categories=["evaluation"] * len(diagnostics),
    )
    assert len(targets) >= 2
    assert guidance["diagnostics"] == diagnostics
    assert all(issue["target_ids"] for issue in guidance["host_directory"]["issues"])
    plan = json.loads((root / "manifest.json").read_bytes())["plan"]
    assert strict_request_reservation(messages, cfg.eval.max_tokens, fmt) <= min(
        plan["stages"]["global"]["request_cap"], plan["context_reservation_limit"]
    )
