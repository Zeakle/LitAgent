"""Keep bounded evaluation recovery on the real SDK and report-consumer path."""

import copy
import json
from pathlib import Path

import pytest
import test_mixed_inventory_pipeline as mixed
from test_complete_native_repair import source_selection
from test_flat_writer_pipeline import FlatGateway
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.quality import build_fact_cards, load_targeted_repairs
from litagent.writing_flat import restore_inventory_stability_transport
from litagent.writing_v2 import SCOPE, source_packet

FIXTURE = Path(__file__).parent / "fixtures/run043-decision-consistency.json"


def select_stable_policy(monkeypatch):
    """Forward the explicit server policy to actual frozen planning and execution."""
    real_plan, real_run = mixed.plan_frozen_budget, mixed.run_frozen

    def plan(*args, **kwargs):
        for key, value in SELECTION.items():
            kwargs.setdefault(key, value)
        return real_plan(*args, **kwargs)

    async def run(*args, **kwargs):
        for key, value in SELECTION.items():
            kwargs.setdefault(key, value)
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(mixed, "plan_frozen_budget", plan)
    monkeypatch.setattr(mixed, "run_frozen", run)


def captured(tmp_path):
    """Read actual persisted repair, writing and budget products."""
    root = tmp_path / "repair.json"
    return (
        json.loads(root.read_text()),
        json.loads(Path(str(root) + ".writer-v2.json").read_text()),
        json.loads(Path(str(root) + ".budget-ledger.json").read_text()),
    )


def phase_requests(gateway, key):
    """Select actual outbound HTTP payloads by their native assignment fields."""
    return [row for row in gateway.sent if key in row["payload"]]


def read_inventory_catalog_at_provider(gateway):
    """Decode shared input only inside the external HTTP reply fixture."""
    previous = gateway.evaluator.chat
    gateway.inventory_transport_views = []

    async def chat(messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        catalog_keys = ("citation_alias_sets", "preserved_assertion_columns")
        if not any(key in payload for key in catalog_keys):
            return await previous(messages, **kwargs)
        assert "inventory_recovery" in payload
        assert all(key in payload for key in catalog_keys)
        # Gateway.__call__ records actual HTTP bytes before this provider reader.
        # Decoded fixture input must never replace that request or its metering.
        actual = gateway.sent[-1]
        request_bytes = actual["request_bytes"]
        assert actual["payload"] == payload
        decoded = restore_inventory_stability_transport(payload)
        gateway.inventory_transport_views.append(
            {"actual_payload": copy.deepcopy(payload), "model_payload": decoded}
        )
        restored_messages = copy.deepcopy(messages)
        restored_messages[1]["content"] = json.dumps(decoded)
        response = await previous(restored_messages, **kwargs)
        assert actual["payload"] == payload
        assert actual["request_bytes"] == request_bytes
        return response

    gateway.evaluator.chat = chat


class StableGateway(mixed.MixedInventoryGateway):
    """Script only provider replies, retaining all application state transitions."""

    def __init__(self, case, *, drift=None, science_failure=False, unknown=False):
        fault = (
            "unknown" if unknown else "science_fail" if science_failure else "science"
        )
        super().__init__(case, fault)
        self.drift = drift
        self.drifted = False
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if drift is None or "allowed_citation_aliases" not in payload:
                return response
            value = json.loads(response.content)
            if drift == "delete":
                for fragment in value["fragments"].values():
                    if any(
                        "FLOPs by 44%" in a["claim_text"]
                        for a in fragment["assertions"]
                    ):
                        # Both propositions are present in this actual native unit.
                        # Keep one after deletion so the stability guard, rather
                        # than the older empty-cited-fragment gate, is exercised.
                        fragment["assertions"] = [
                            {
                                "kind": "scientific",
                                "claim_text": "Relation networks learn a comparison function.",
                            },
                            {
                                "kind": "scientific",
                                "claim_text": "Relation networks reduce FLOPs by 44%.",
                            },
                        ]
            if "inventory_recovery" not in payload:
                response.content = json.dumps(value)
                return response
            for fragment in value["fragments"].values():
                for assertion in list(fragment["assertions"]):
                    if "FLOPs by 44%" not in assertion["claim_text"]:
                        continue
                    self.drifted = True
                    if drift == "delete":
                        fragment["assertions"].remove(assertion)
                    elif drift == "reclassify":
                        assertion["kind"] = "report_structure"
                    elif drift == "soften":
                        assertion["claim_text"] = (
                            "Relation networks may reduce computation."
                        )
                    else:
                        raise AssertionError("Unknown controlled drift.")
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)

    def answer(self, payload, native_schema):
        """Give the deletion control two actual propositions in one host unit."""
        value = super().answer(payload, native_schema)
        if (
            self.drift == "delete"
            and payload.get("requested_body_section") == "methods"
        ):
            value["units"][0]["text"] = (
                "Relation networks learn a comparison function; "
                "Relation networks reduce FLOPs by 44%."
            )
        return value


async def execute(case, tmp_path, monkeypatch, gateway):
    """Use the existing complete Synthesis, Reviewer, report and Evaluation runner."""
    select_stable_policy(monkeypatch)
    return await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("science_failure", [False, True])
async def test_one_inventory_recovery_preserves_negative_then_allows_one_prose_repair(
    case, tmp_path, monkeypatch, science_failure
):
    """A controlled successful recovery leaves one shared slot for real prose repair."""
    gateway = StableGateway(case, science_failure=science_failure)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    state, journal, budget = captured(tmp_path)
    history = report["metadata"]["evaluation_history"]
    initial = history["initial_evaluation"]["faithfulness"]["details"]
    recovered = history["attempts"][0]["candidate_evaluation"]["faithfulness"][
        "details"
    ]
    assert initial["inventory_check"]["status"] == "failed"
    assert recovered["inventory_check"]["status"] == "passed"
    assert initial["source_report_sha256"] == recovered["source_report_sha256"]
    negative = [c for c in recovered["assertions"] if "FLOPs by 44%" in c["claim_text"]]
    assert negative and all(c["status"] == "unsupported" for c in negative)
    assert all(c["classification_valid"] and c["source_faithful"] for c in negative)
    feedback = state["inventory_recovery_0"]["feedback"]
    preserved = feedback["preserved_fragments"]
    protected = [
        row
        for row in preserved
        if any("FLOPs by 44%" in a["claim_text"] for a in row["assertions"])
    ]
    assert protected and all(row["assertions"] for row in protected)
    assert all(
        set(a) == {"kind", "claim_text"} for row in preserved for a in row["assertions"]
    )
    assert not any("status" in row or "supported" in row for row in preserved)
    assert len(feedback["preserved_fragments_sha256"]) == 64
    assert len(gateway.inventory_transport_views) == 1
    view = gateway.inventory_transport_views[0]
    assert "citation_alias_sets" in view["actual_payload"]
    assert "preserved_assertion_columns" in view["actual_payload"]
    assert "citation_alias_sets" not in view["model_payload"]
    assert "preserved_assertion_columns" not in view["model_payload"]
    assert any(row["payload"] == view["actual_payload"] for row in gateway.sent)
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["purpose"] == mixed.MIXED_CONTRACT
    assert attempts[1].get("purpose") != mixed.MIXED_CONTRACT
    assert gateway.inventories == 3 and gateway.repair_count == 1
    assert len(phase_requests(gateway, "targets")) == 1
    assert len(phase_requests(gateway, "body_units")) == 2
    assert len(phase_requests(gateway, "global_output_contract")) == 3
    assert len(phase_requests(gateway, "cited_materials")) == 3
    assert budget["requests_sent"] == budget["call_count"] == len(gateway.sent)
    assert all(e["status"] == "settled" for e in budget["entries"].values())
    assert budget["usage_uncertain"] is False
    final = report["evaluation"]["faithfulness"]["details"]
    assert (
        final["coverage"]["complete"] and final["stages"]["global_check"] == "completed"
    )
    if science_failure:
        assert not report["delivery"]["publishable"]
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]
        assert journal["candidates"] and not journal["candidates"][0]["committed"]
    else:
        assert report["delivery"]["publishable"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert final["supported_count"] == final["factual_claim_count"]
        assert "FLOPs by 44%" not in report["survey"]
    assert (
        load_targeted_repairs(tmp_path / "repair.json", "mixed-inventory-offline")[
            "attempts"
        ]
        == attempts
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["delete", "reclassify", "soften"])
async def test_undiagnosed_negative_cannot_disappear_before_reassessment(
    case, tmp_path, monkeypatch, drift
):
    """Reject changes to protected scientific content without silently restoring it."""
    gateway = StableGateway(case, drift=drift)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    assert gateway.drifted
    assert not report["delivery"]["publishable"]
    assert "FLOPs by 44%" in report["survey"]
    assert gateway.inventories == 2 and gateway.repair_count == 0
    assert len(phase_requests(gateway, "cited_materials")) == 1
    assert len(phase_requests(gateway, "global_output_contract")) == 1
    assert not phase_requests(gateway, "targets")
    _, _, budget = captured(tmp_path)
    assert budget["requests_sent"] == budget["call_count"] == len(gateway.sent)
    assert budget["usage_uncertain"] is False
    progress = [
        json.loads(p.read_text())["progress"]
        for p in (tmp_path / "evaluation").glob("*/progress.json")
    ]
    recovery = next(p for p in progress if p.get("inventory_recovery"))
    assert len(recovery["requests"]) == 1
    assert recovery["requests"][0]["status"] == "response_received"
    assert recovery["requests"][0]["response"]
    assert not recovery.get("batches")
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["skipped_reason"] == "inventory_recovery_preserved_fragment_changed"
    diagnostic = final["inventory_stability_check"]
    state, _, _ = captured(tmp_path)
    feedback = state["inventory_recovery_0"]["feedback"]
    protected = next(
        row
        for row in feedback["preserved_fragments"]
        if any("FLOPs by 44%" in a["claim_text"] for a in row["assertions"])
    )
    assert diagnostic["status"] == "failed"
    assert diagnostic["reason_code"] == final["skipped_reason"]
    assert diagnostic["source_report_sha256"] == feedback["source_report_sha256"]
    assert diagnostic["source_fragment_id"] == protected["source_fragment_id"]
    assert diagnostic["source_unit_id"] == protected["source_unit_id"]
    assert diagnostic["expected_count"] == len(protected["assertions"])
    assert diagnostic["actual_count"] == len(protected["assertions"]) - (
        1 if drift == "delete" else 0
    )
    assert diagnostic["expected_content_sha256"] != diagnostic["actual_content_sha256"]
    assert not {"claim_text", "source_quote", "response"} & diagnostic.keys()


@pytest.mark.asyncio
async def test_unknown_recovery_keeps_shared_attempt_and_reservation(
    case, tmp_path, monkeypatch
):
    """Missing provider usage cannot release funds or trigger another request."""
    gateway = StableGateway(case, unknown=True)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    _, _, budget = captured(tmp_path)
    assert not report["delivery"]["publishable"]
    assert budget["usage_uncertain"] and budget["stop_reason"] == "strict_usage_unknown"
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 1 and gateway.recovery_requests == 1
    assert not phase_requests(gateway, "targets")
    held = [e for e in budget["entries"].values() if e.get("inventory_recovery")]
    assert len(held) == 1 and held[0]["status"] == "interrupted_no_replay"
    assert held[0]["actual_tokens"] is None and held[0]["reserved_tokens"] > 0


class RawConflictGateway(FlatGateway):
    """Project only host IDs of a real043 contradictory decision at HTTP."""

    def __init__(self, case, raw_decision):
        super().__init__(case)
        self.original_raw = copy.deepcopy(raw_decision)
        self.injected_raw = None
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            value = json.loads(response.content)
            claim = next(c for c in payload["assertions"] if "On D," in c["claim_text"])
            assert claim["required_support_aspects"] == ["predicate", "conditions"]
            aliases = {e["evidence_alias"] for e in claim["evidence"]}
            permitted = [
                pid
                for pid, entry in payload["support_passages"].items()
                if entry[0] in aliases and entry[1] != entry[0]
            ]
            assert permitted
            raw = copy.deepcopy(raw_decision)
            raw["claim_id"] = claim["claim_id"]
            raw["support_basis"] = [{"support_id": permitted[0]}]
            for aspect in raw["aspect_checks"]:
                aspect["support_ids"] = [permitted[0]]
            value["decisions"] = [
                raw if d["claim_id"] == claim["claim_id"] else d
                for d in value["decisions"]
            ]
            self.injected_raw = raw
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 1])
async def test_real043_conflict_fields_stop_sdk_batch_and_preserve_usage(
    case, tmp_path, monkeypatch, index
):
    """ID projection is a protocol control, never a scientific regrading of043."""
    fixture_bytes = FIXTURE.read_bytes()
    raw = json.loads(fixture_bytes)["cases"][index]["raw_decision"]
    case.body["methods"][0]["sentences"][0][
        "text"
    ] = "On D, Relation networks learn a comparison function."
    gateway = RawConflictGateway(case, raw)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    assert FIXTURE.read_bytes() == fixture_bytes and gateway.original_raw == raw
    assert not report["delivery"]["publishable"]
    assert not phase_requests(gateway, "targets")
    assert len(phase_requests(gateway, "cited_materials")) == 1
    assert len(phase_requests(gateway, "allowed_citation_aliases")) == 1
    assert not phase_requests(gateway, "global_output_contract")
    progress_path = next((tmp_path / "evaluation").glob("*/progress.json"))
    progress = json.loads(progress_path.read_text())["progress"]
    batch = progress["batches"][0]
    assert batch["status"] == "invalid_response"
    assert batch["reason_code"] == "evaluation_decision_consistency_invalid"
    assert batch["usage"]["usage_available"] is True
    diagnostic = batch["diagnostics"]["decision_inconsistencies"][0]
    assert (
        diagnostic["subreason"] == "negative_status_with_complete_positive_diagnostics"
    )
    assert diagnostic["original_decision"]["status"] == raw["status"] == "unsupported"
    assert (
        diagnostic["original_decision"]["reason_code"]
        == raw["reason_code"]
        == "attribution_mismatch"
    )
    assert all(a["supported"] for a in diagnostic["original_decision"]["aspect_checks"])
    assert progress["requests"][-1]["response"]
    assert not progress.get("global_check")
    _, _, budget = captured(tmp_path)
    assert budget["usage_uncertain"] is False
    assert budget["requests_sent"] == budget["call_count"] == len(gateway.sent)
    assert all(e["status"] == "settled" for e in budget["entries"].values())


def replace_source(case, text):
    """Rebuild real source cards and the packet after changing a public control."""
    case.ledger["b"].update(text=text, supporting_text=text)
    case.cards = build_fact_cards(case.ledger)
    case.selection = source_selection(case.cards)
    case.selection.writing_scope_policy = SCOPE
    case.packet, case.aliases = source_packet(
        case.cards,
        case.selection,
        "Explain mechanisms and limitations.",
        case.packet["requirements"],
        "en",
    )
    case.by_eid = {
        case.cards[cid]["evidence_id"]: ref for ref, cid in case.aliases.items()
    }
    for paragraphs in case.body.values():
        if isinstance(paragraphs, list):
            for p in paragraphs:
                for unit in p["sentences"]:
                    unit["fact_card_refs"] = [case.by_eid["b"]]


class CausalGateway(FlatGateway):
    """Keep atom assignment separate from actual source support in provider replies."""

    def __init__(self, case, supported):
        super().__init__(case, declined=not supported)
        self.supported_causal = supported
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            claims = {c["claim_id"]: c for c in payload["assertions"]}
            value = json.loads(response.content)
            for decision in value["decisions"]:
                if (
                    "therefore improves"
                    not in claims[decision["claim_id"]]["claim_text"]
                ):
                    continue
                assert (
                    "causal_relation"
                    in claims[decision["claim_id"]]["required_support_aspects"]
                )
                decision["atomic"] = True
                if not supported:
                    decision.update(
                        status="unsupported", reason_code="insufficient_support"
                    )
                    for aspect in decision["aspect_checks"]:
                        if aspect["aspect"] == "causal_relation":
                            aspect.update(
                                supported=False,
                                support_ids=[],
                                explanation="The original does not establish the claimed causal gain.",
                            )
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        read_inventory_catalog_at_provider(self)


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("separator", [",", ";"])
async def test_causal_atom_retains_condition_and_still_requires_causal_support(
    case, tmp_path, monkeypatch, supported, separator
):
    """Comma/semicolon never replaces an independent atomicity or entailment verdict."""
    claim = f"On D, Relation networks learn a comparison function{separator} this therefore improves accuracy."
    original = (
        claim if supported else "On D, Relation networks learn a comparison function."
    )
    replace_source(case, original)
    case.body["methods"][0]["sentences"][0]["text"] = claim
    gateway = CausalGateway(case, supported)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    details = report["evaluation"]["faithfulness"]["details"]
    selected = [
        c for c in details["assertions"] if "therefore improves" in c["claim_text"]
    ]
    assert selected and all(c["atomic"] for c in selected)
    assert details["inventory_check"]["status"] == "passed"
    assert all("On D" in c["claim_text"] for c in selected)
    assert all(
        any(
            a["aspect"] == "causal_relation" and a["supported"] is supported
            for a in c["aspect_checks"]
        )
        for c in selected
    )
    assert bool(report["delivery"]["publishable"]) is supported
    if not supported:
        assert all(c["status"] == "unsupported" for c in selected)
        assert not [
            a
            for a in report["metadata"]["targeted_repair_state"]["attempts"]
            if a.get("purpose") == mixed.MIXED_CONTRACT
        ]


@pytest.mark.asyncio
async def test_hash_only_stability_feedback_rejects_before_send_or_shared_charge(
    case, tmp_path, monkeypatch
):
    """Corrupt actual host feedback without replacing its factory or consumers."""
    import litagent.eval.inventory_recovery as recovery

    real_feedback = recovery.inventory_recovery_feedback
    damaged = []

    def hash_only(*args, **kwargs):
        value = real_feedback(*args, **kwargs)
        if value is not None:
            assert kwargs["stable"] is True
            assert value["preserved_fragments"]
            assert len(value["preserved_fragments_sha256"]) == 64
            damaged.append(copy.deepcopy(value))
            value.pop("preserved_fragments")
        return value

    monkeypatch.setattr(recovery, "inventory_recovery_feedback", hash_only)
    gateway = StableGateway(case)
    report, _ = await execute(case, tmp_path, monkeypatch, gateway)
    assert len(damaged) == 1
    assert not report["delivery"]["publishable"]
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["skipped_reason"] == "inventory_recovery_stability_identity_invalid"
    diagnostic = final["inventory_stability_check"]
    assert diagnostic["reason_code"] == final["skipped_reason"]
    assert diagnostic["source_report_sha256"] == damaged[0]["source_report_sha256"]
    assert diagnostic["status"] == "failed"
    assert gateway.recovery_requests == 0 and gateway.inventories == 1
    assert len(phase_requests(gateway, "allowed_citation_aliases")) == 1
    assert len(phase_requests(gateway, "cited_materials")) == 1
    assert len(phase_requests(gateway, "global_output_contract")) == 1
    assert not phase_requests(gateway, "targets")
    state, _, budget = captured(tmp_path)
    assert state["state"]["attempts"] == []
    assert report["metadata"]["targeted_repair_state"]["attempts"] == []
    assert budget["requests_sent"] == budget["call_count"] == len(gateway.sent)
    assert budget["usage_uncertain"] is False
    assert all(
        e["sent"] and e["status"] == "settled" for e in budget["entries"].values()
    )
