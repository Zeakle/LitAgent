"""Authenticate lossless conflict task views through actual writer consumers."""

import base64
import copy
import hashlib
import json

import pytest
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_repair_task_view import (
    decoded_issue_rows,
    encode,
    issue_rows,
    rebind_prior,
    rehash_view,
    replace_value,
    restore,
    task_input,
)
from test_semantic_conflict_identity import scope
from test_semantic_conflict_repair_pipeline import prepare_case, run_conflict
from test_source_map_pipeline import products, sent_with
from test_source_owner_pipeline import FLAGS
from test_structured_repair_guidance import guidance_case  # noqa: F401
from test_structured_repair_guidance import (
    context_for,
    prior_context,
)
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
import litagent.writing_v2 as writing
from litagent.capacity_repair import capacity_targets
from litagent.eval.semantic_conflict import (
    EFFECT_CONTRACT,
    POLICY_FIELDS,
    SEMANTIC_CONFLICT_POLICY,
    semantic_conflict_identity,
)
from litagent.quality import digest
from litagent.writing_v2 import WriterError

POSITIVE = "positive_status_with_negative_required_aspect"
NEGATIVE = "negative_status_with_complete_positive_diagnostics"
EFFECT_KEYS = {
    "contract",
    "policy",
    "policy_identity",
    "code",
    "original_status",
    "original_reason_code",
    "decision_sha256",
    "binding_sha256",
}
WIRE_FIELDS = {
    "repair_guidance_wire_revision",
    "repair_guidance_wire_identity",
    "repair_guidance_capacity_rule",
}
ASPECTS = ["predicate", "conditions", "quantities", "causal_relation"]


def unicode_case(case, text):
    """Keep native text, complete source and positional authority consistent."""
    body, _, packet, fid, fragment, raw = copy.deepcopy(case)
    for section in writing.BODY:
        body[section][0]["sentences"][0]["text"] = text
    quote = text + " A numeric result is 10 percent."
    packet["sources"]["s1"].update(
        source_quote=quote, content_sha256=hashlib.sha256(quote.encode()).hexdigest()
    )
    fragment.update(text=text + " [E:E1]", source_end=10 + len(text + " [E:E1]"))
    raw.update(
        claim_text=text,
        source_quote=fragment["text"],
        source_end=fragment["source_end"],
    )
    raw["support_validation"]["spans"][0].update(quote=quote, source_end=len(quote))
    return (
        body,
        capacity_targets(body, {"reason": "capacity"}, maximum=66),
        packet,
        fid,
        fragment,
        raw,
    )


def conflict_diagnostics(case, origin):
    """Make several typed groups with unique per-issue hashes, never a gold set."""
    result = []
    for index in range(6):
        raw = copy.deepcopy(case[-1])
        positive = index % 2 == 0
        raw.update(
            claim_id=f"claim-{origin}-{index}",
            statement_id=f"claim-{origin}-{index}",
            status="unverifiable",
            reason_code="unverifiable",
            local_error="evaluation_decision_consistency_invalid",
        )
        raw["aspect_checks"] = [
            {
                "aspect": name,
                "supported": not (positive and offset == 0),
                "support_ids": ["e1:full_original"],
                "explanation": "LOCAL-ONLY-受控解释 Δ café",
            }
            for offset, name in enumerate(ASPECTS)
        ]
        raw["support_validation"].update(
            required_aspects=ASPECTS, actual_aspects=ASPECTS
        )
        raw["assessment_conflict_effect"] = {
            "contract": EFFECT_CONTRACT,
            "policy": SEMANTIC_CONFLICT_POLICY,
            "policy_identity": semantic_conflict_identity(),
            "code": POSITIVE if positive else NEGATIVE,
            "original_status": "supported" if positive else "unsupported",
            "original_reason_code": "supported" if positive else "record_mismatch",
            "decision_sha256": digest(["decision", origin, index]),
            "binding_sha256": digest(["binding", origin, index]),
        }
        if index == 5:
            # A selected request can contain an ordinary diagnostic as well.
            raw.pop("assessment_conflict_effect")
            raw.update(
                status="unsupported",
                reason_code="insufficient_support",
                local_error=None,
            )
            raw["aspect_checks"][0]["supported"] = False
        result.append(raw)
    return result


def semantic_input(case, *, prior=True, selected=True):
    """Build both views with real host directories before choosing representation."""
    current = context_for(case, conflict_diagnostics(case, "current"))
    previous = old = None
    if prior:
        previous, old = prior_context(case)
        previous["diagnostics"] = conflict_diagnostics(case, "prior")
        directory = previous["host_directory"]
        template = copy.deepcopy(directory["issues"][0])
        directory["issues"] = [
            {**copy.deepcopy(template), "diagnostic_index": index}
            for index in range(len(previous["diagnostics"]))
        ]
        rebind_prior(previous, old)
    args = task_input(case, current=current, prior=previous, old=old)
    args[1]["writing_envelope"] = scope(
        semantic_conflict_policy=SEMANTIC_CONFLICT_POLICY if selected else None
    )
    return args


@pytest.mark.parametrize(
    "text",
    [
        "Method A works under condition X.",
        "方法甲仅在条件甲下成立。 Δ café 🧪",
        '条件 e\u0301 / é，路径\\字面值与引号"保持。',
    ],
)
@pytest.mark.parametrize("prior", [False, True])
def test_selected_public_rows_keep_all_eight_effect_keys_and_all_aspects(
    guidance_case, text, prior
):
    """An independent decoder and authenticated inverse both retain every value."""
    case = unicode_case(guidance_case, text)
    args = semantic_input(case, prior=prior)
    before = copy.deepcopy(args)
    wire, receipt = encode(args)
    assert restore(args, wire, receipt) == args[0]
    hashes = []
    for origin in ("current", "prior_candidate"):
        if wire[origin] is None:
            assert origin == "prior_candidate" and not prior
            continue
        content = wire[origin]
        assert "issue_templates" in wire
        rows = decoded_issue_rows(wire, origin)
        assert len(rows) == len(args[0][origin]["issues"]) == 6
        assert len(wire["issue_templates"]["issue_columns"]) >= 2
        for row, canonical in zip(rows, args[0][origin]["issues"], strict=True):
            assert row["claim_text"] == text
            assert row["target_ids"] == canonical["target_ids"]
            if "assessment_conflict_effect" not in canonical:
                assert "assessment_conflict_effect" not in row
                continue
            effect = row["assessment_conflict_effect"]
            assert set(effect) == EFFECT_KEYS
            assert effect == canonical["assessment_conflict_effect"]
            assert row["aspect_checks"] == canonical["aspect_checks"]
            assert [check["aspect"] for check in row["aspect_checks"]] == ASPECTS
            assert row["support_validation"] == canonical["support_validation"]
            hashes.append(effect["decision_sha256"])
    assert len(hashes) == len(set(hashes))
    assert args[1]["sources"] == case[2]["sources"]
    assert "LOCAL-ONLY-" not in json.dumps(wire)
    assert args == before


def test_unselected_task_view_keeps_old_shape_and_wire_identity(guidance_case):
    """The original table remains byte-compatible when the new pair is absent."""
    args = semantic_input(guidance_case, selected=False)
    wire, receipt = encode(args)
    assert all(
        "issue_shared" not in wire[name] for name in ("current", "prior_candidate")
    )
    assert restore(args, wire, receipt) == args[0]
    assert receipt["wire_identity"] == flat.repair_guidance_wire_identity(
        flat.REPAIR_TASK_VIEW_REVISION
    )


@pytest.mark.parametrize(
    "namespace", ["support_validation", "assessment_conflict_effect"]
)
@pytest.mark.parametrize("value", [None, False, [], {}, "not-an-object"])
def test_present_invalid_namespaces_cannot_disappear_as_absent(namespace, value):
    """A present malformed object must not be silently dropped during flattening."""
    with pytest.raises(WriterError):
        flat._semantic_conflict_issue_table([{namespace: value}])


def test_absent_namespaces_remain_absent_in_selected_table():
    """Missing optional data never invents an empty effect or validation."""
    shared, columns, rows = flat._semantic_conflict_issue_table([{"claim_id": "one"}])
    assert issue_rows(
        {"issue_shared": shared, "issue_columns": columns, "issues": rows}
    ) == [{"claim_id": "one"}]


def test_empty_table_does_not_invent_an_issue_or_authorize_an_empty_origin():
    """An empty pure representation stays empty; canonical checks retain authority."""
    shared, columns, rows = flat._semantic_conflict_issue_table([])
    assert (shared, columns, rows) == ({}, [], [])
    assert (
        issue_rows({"issue_shared": shared, "issue_columns": columns, "issues": rows})
        == []
    )


@pytest.mark.parametrize("origin_name", ["current", "prior_candidate"])
@pytest.mark.parametrize(
    "fault",
    [
        "effect_hash",
        "effect_status",
        "remove_effect_key",
        "aspect",
        "cross_origin",
        "shared_collision",
        "equal_hash_reorder",
    ],
)
def test_complete_wire_tampering_still_fails_after_every_view_hash_is_recomputed(
    guidance_case, origin_name, fault
):
    """Self-consistent receipt hashes cannot replace original issue authority."""
    args = semantic_input(guidance_case)
    wire, receipt = encode(args)
    origin = wire[origin_name]
    rows = decoded_issue_rows(wire, origin_name)
    table = wire["issue_templates"]
    template_index = issue_rows(origin)[0]["issue_template_index"]
    if fault == "effect_hash":
        replace_value(
            origin,
            0,
            "assessment_conflict_effect.decision_sha256",
            base64.b85encode(bytes.fromhex("f" * 64)).decode("ascii"),
        )
    elif fault == "effect_status":
        replace_value(
            table,
            template_index,
            "assessment_conflict_effect.original_status",
            "unsupported",
        )
    elif fault == "remove_effect_key":
        field = "assessment_conflict_effect.contract"
        group = table["issue_columns"][table["issues"][template_index][0]]
        if field in table["issue_shared"]:
            table["issue_shared"].pop(field)
        else:
            group["shared"].pop(field)
    elif fault == "aspect":
        checks = copy.deepcopy(rows[0]["aspect_checks"])
        checks.pop()
        encoded_checks = [
            [check[field] for field in wire["aspect_columns"]] for check in checks
        ]
        replace_value(table, template_index, "aspect_checks", encoded_checks)
    elif fault == "cross_origin":
        other = wire["prior_candidate" if origin_name == "current" else "current"]
        replace_value(
            origin,
            0,
            "assessment_conflict_effect.binding_sha256",
            issue_rows(other)[0]["assessment_conflict_effect"]["binding_sha256"],
        )
    elif fault == "shared_collision":
        field = next(iter(table["issue_shared"]))
        table["issue_columns"][0]["shared"][field] = copy.deepcopy(
            table["issue_shared"][field]
        )
    else:
        before = digest(wire)
        table["issue_shared"] = dict(reversed(list(table["issue_shared"].items())))
        assert digest(wire) == before
    rehash_view(wire, receipt)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        restore(args, wire, receipt)


def test_selector_accepts_only_complete_owner_or_exact_early_projection():
    """Reduced fields describe representation while the real owner retains authority."""
    selected = scope()
    early = writing.guidance_wire_fields(selected)
    assert set(early) == WIRE_FIELDS | set(POLICY_FIELDS)
    assert flat._semantic_conflict_task_view_selected(selected) is True
    assert flat._semantic_conflict_task_view_selected(early) is True
    parent = scope(semantic_conflict_policy=None)
    old_early = writing.guidance_wire_fields(parent)
    assert set(old_early) == WIRE_FIELDS
    assert flat._semantic_conflict_task_view_selected(parent) is False
    assert flat._semantic_conflict_task_view_selected(old_early) is False
    assert writing.guidance_wire_fields(None) == {}


@pytest.mark.parametrize("form", ["full", "early"])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_policy",
        "missing_identity",
        "unknown_policy",
        "bad_identity",
        "extra_early_field",
    ],
)
def test_partial_or_unknown_policy_pair_is_never_a_legacy_fallback(form, fault):
    """Removing or inventing selection data fails instead of enabling old output."""
    envelope = scope()
    if form == "early":
        envelope = writing.guidance_wire_fields(envelope)
    if fault == "missing_policy":
        envelope.pop("semantic_conflict_policy")
    elif fault == "missing_identity":
        envelope.pop("semantic_conflict_identity")
    elif fault == "unknown_policy":
        envelope["semantic_conflict_policy"] = "future-policy"
    elif fault == "bad_identity":
        envelope["semantic_conflict_identity"] = "e" * 64
    else:
        envelope["untrusted_owner_claim"] = True
    if form == "full":
        envelope["identity"] = digest(
            {k: v for k, v in envelope.items() if k != "identity"}
        )
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat._semantic_conflict_task_view_selected(envelope)


def build_captured_input(case, *, selected=True):
    """Use the actual final repair builder and retain its local receipt."""
    case = copy.deepcopy(case)
    body, targets, packet = case[:3]
    packet.update(
        query="核对原文 Δ",
        language="en",
        requirements=[],
        chapter_recommendations={key: ["c1"] for key in writing.BODY},
    )
    wrapper = context_for(case, conflict_diagnostics(case, "current"))
    envelope = scope(
        semantic_conflict_policy=SEMANTIC_CONFLICT_POLICY if selected else None
    )
    captures = []
    result = writing.repair_request_input(
        packet,
        body,
        {"r1": targets["r1"]},
        {"c1": "E1"},
        cause="evaluation",
        previous_failure=None,
        writing_plan=writing.RANGE_WRITING_PLAN,
        envelope=envelope,
        wire_contract="flat-writer-v1",
        guidance_context={"current": wrapper, "prior_candidate": None},
        guidance_wire_capture=captures.append,
    )
    assert len(captures) == 1
    return result, captures[0], packet, envelope


def reseal_record(record):
    """Recompute every local data hash while leaving true owner fields untouched."""
    for key in ("canonical", "authority", "payload", "receipt"):
        record[key + "_sha256"] = digest(record[key])
    record["identity"] = digest(
        {key: value for key, value in record.items() if key != "identity"}
    )


@pytest.mark.parametrize("selected", [False, True])
def test_actual_builder_binds_early_markers_and_final_sdk_projection(
    guidance_case, selected
):
    """Early and late input views remain distinct, each owned by the same envelope."""
    (system, payload, schema), record, packet, envelope = build_captured_input(
        guidance_case, selected=selected
    )
    expected = writing.guidance_wire_fields(envelope)
    assert record["payload"]["writing_envelope"] == expected
    assert set(expected) == WIRE_FIELDS | (set(POLICY_FIELDS) if selected else set())
    writing.check_guidance_wire_record(envelope, record, packet=packet)
    assert flat.ASSESSMENT_CONFLICT_REVIEW_INSTRUCTION not in system
    final_system, final_payload = writing.prepare_planned_input(
        system, payload, packet, {"c1": "E1"}, writing.RANGE_WRITING_PLAN, envelope
    )
    assert (flat.ASSESSMENT_CONFLICT_REVIEW_INSTRUCTION in final_system) is selected
    messages = writing.request_messages(
        final_system, final_payload, schema, wire_contract="flat-writer-v1"
    )
    assert json.loads(messages[1]["content"]) == final_payload
    assert final_payload["sources"] == packet["sources"]
    if selected:
        assert {
            key: final_payload["writing_envelope"][key] for key in POLICY_FIELDS
        } == {key: envelope[key] for key in POLICY_FIELDS}


@pytest.mark.parametrize(
    "fault", ["strip_pair", "inject_pair", "non_dict_payload", "unknown_field"]
)
def test_record_payload_cannot_self_select_an_owner_after_full_rehash(
    guidance_case, fault
):
    """Host receipt validation authenticates nested early fields before any sender."""
    _, record, packet, envelope = build_captured_input(
        guidance_case, selected=fault != "inject_pair"
    )
    if fault == "non_dict_payload":
        record["payload"] = []
    elif fault == "unknown_field":
        record["payload"]["writing_envelope"]["unexpected"] = False
    elif fault == "strip_pair":
        for key in POLICY_FIELDS:
            record["payload"]["writing_envelope"].pop(key)
    else:
        record["payload"]["writing_envelope"].update(
            {key: scope()[key] for key in POLICY_FIELDS}
        )
    reseal_record(record)
    with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
        writing.check_guidance_wire_record(envelope, record, packet=packet)


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_owner_selection_tamper_stops_before_repair_http_or_allowance(
    case, tmp_path, monkeypatch, selected
):
    """Inject local receipt damage; retain the actual writer, SDK and settled owner."""
    import test_structured_repair_pipeline as structured

    gateway = structured.LocalGateway(case)
    owner, budget, _ = await owner_for(gateway)
    flags = {**FLAGS}
    if selected:
        flags["semantic_conflict_policy"] = SEMANTIC_CONFLICT_POLICY
    plan = writing.writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **flags,
    )
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    original_builder = writing.repair_request_input

    def damaged_capture(*args, **kwargs):
        """Change only the saved local witness after the real producer runs."""
        capture = kwargs["guidance_wire_capture"]

        def damage(record):
            early = record["payload"]["writing_envelope"]
            if selected:
                for key in POLICY_FIELDS:
                    early.pop(key)
            else:
                early.update({key: scope()[key] for key in POLICY_FIELDS})
            reseal_record(record)
            capture(record)

        return original_builder(*args, **{**kwargs, "guidance_wire_capture": damage})

    try:
        body = await session.initial_body()
        before = copy.deepcopy(
            (session.state["attempts"], session.journal["calls"], budget.snapshot())
        )
        before_sent = len(gateway.sent)
        monkeypatch.setattr(writing, "repair_request_input", damaged_capture)
        target = {
            "writer:methods:0:0": {
                "path": ["methods", 0, 0],
                "original": copy.deepcopy(body["methods"][0]["sentences"][0]),
                "explanation": "Independent local review only.",
            }
        }
        with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
            await session.patch(body, target, cause="local")
        assert len(gateway.sent) == before_sent
        assert session.state["attempts"] == before[0]
        assert session.journal["calls"] == before[1]
        assert budget.snapshot() == before[2]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["positive", "negative"])
async def test_actual_two_repair_consumer_retains_both_origins_and_full_settlement(
    case, tmp_path, monkeypatch, direction
):
    """Only external transport is mocked in the full semantic recovery workflow."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, direction=direction, failing_candidates=1
    )
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 2 and gateway.inventories == 3
    seen = []
    for index, entry in enumerate(repairs):
        wire = entry["payload"]["repair_guidance"]
        assert (wire["prior_candidate"] is not None) is (index == 1)
        for origin in ("current", "prior_candidate"):
            if wire[origin] is None:
                continue
            assert "issue_templates" in wire
            effects = [
                row["assessment_conflict_effect"]
                for row in decoded_issue_rows(wire, origin)
                if "assessment_conflict_effect" in row
            ]
            assert all(set(effect) == EFFECT_KEYS for effect in effects)
            if origin == "current":
                assert effects
            else:
                assert decoded_issue_rows(wire, origin)
            seen.extend(effects)
        for source in case.cards.values():
            assert source["source_quote"] in json.dumps(entry["request"])
    assert seen and report["delivery"]["publishable"] is True
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"] is True
    assert report["evaluation"]["faithfulness"]["score"] == 1.0
    _, journal, ledger = products(tmp_path)
    assert [candidate["committed"] for candidate in journal["candidates"]] == [
        False,
        True,
    ]
    assert ledger["usage_uncertain"] is False
    assert ledger["requests_sent"] == len(gateway.sent)
    assert all(item["status"] == "settled" for item in ledger["entries"].values())


@pytest.mark.parametrize("target", ["selector", "table", "fields", "record"])
def test_selected_guidance_identity_binds_new_consumers_without_changing_legacy(
    monkeypatch, target
):
    """Changes to executable selected rules cannot reuse a prior frozen identity."""
    before = flat.semantic_conflict_guidance_identity()
    legacy = (
        flat.structured_repair_guidance_identity(),
        flat.repair_guidance_wire_identity(),
        flat.repair_guidance_wire_identity(flat.REPAIR_TASK_VIEW_REVISION),
    )

    def altered(*args, **kwargs):
        """Provide a distinct executable source without calling the changed rule."""
        raise AssertionError("Identity-only mutation must never execute.")

    module, name = {
        "selector": (flat, "_semantic_conflict_task_view_selected"),
        "table": (flat, "_semantic_conflict_issue_table"),
        "fields": (writing, "guidance_wire_fields"),
        "record": (writing, "check_guidance_wire_record"),
    }[target]
    monkeypatch.setattr(module, name, altered)
    assert flat.semantic_conflict_guidance_identity() != before
    assert (
        flat.structured_repair_guidance_identity(),
        flat.repair_guidance_wire_identity(),
        flat.repair_guidance_wire_identity(flat.REPAIR_TASK_VIEW_REVISION),
    ) == legacy
