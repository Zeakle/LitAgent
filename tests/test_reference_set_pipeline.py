"""Exercise explicit reference sets through actual HTTP and writer consumers."""

import copy
import json

import pytest
import test_flat_writer_pipeline as flat_tests
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.quality import digest
from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size
from litagent.writing_flat import REFERENCE_SET_CONTRACT, REFERENCE_SET_INSTRUCTION
from litagent.writing_v2 import WriterError, writer_budget_plan


class DuplicateGateway(FlatGateway):
    """Change only model reference arrays at the external HTTP boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duplicate_count = 1
        self.original_patch = False
        self.unknown_reference = False
        self.patch_text = None

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if self.patch_text is not None and "targets" in payload:
            for row in value["patches"]:
                row["text"] = self.patch_text
        if self.original_patch and "targets" in payload:
            for row in value["patches"]:
                target = payload["targets"][row["target_id"]]
                path = target["path"]
                row["text"] = payload["current_body"][path[0]][path[1]]["sentences"][
                    path[2]
                ]["text"]

        def visit(node):
            if isinstance(node, dict):
                for name in ("fact_card_refs", "body_unit_refs"):
                    if node.get(name):
                        node[name] += node[name][:1] * self.duplicate_count
                        if self.unknown_reference:
                            node[name][-1] = "unknown-source"
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        return value


async def setup(case, tmp_path, gateway, *, capacity=None):
    """Activate a real explicitly selected v7 plan and model-bound clients."""
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        reference_contract=REFERENCE_SET_CONTRACT,
        capacity=capacity,
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    return owner, make_session(owner, case, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
async def test_actual_synthesis_reviewer_evaluation_preserves_scientific_gate(
    case, tmp_path, monkeypatch, unsupported
):
    import litagent.frozen_writing as frozen

    instances, reports = [], []

    class CapturingGateway(DuplicateGateway):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    original_plan, original_run = frozen.plan_frozen_budget, frozen.run_frozen

    def selected_plan(*args, **kwargs):
        kwargs.setdefault("reference_contract", REFERENCE_SET_CONTRACT)
        return original_plan(*args, **kwargs)

    async def selected_run(*args, **kwargs):
        kwargs.setdefault("reference_contract", REFERENCE_SET_CONTRACT)
        result = await original_run(*args, **kwargs)
        reports.append(result)
        return result

    monkeypatch.setattr(flat_tests, "FlatGateway", CapturingGateway)
    monkeypatch.setattr(frozen, "plan_frozen_budget", selected_plan)
    monkeypatch.setattr(frozen, "run_frozen", selected_run)
    await flat_tests.test_flat_real_frozen_worker_runner_and_report_exit(
        case, tmp_path, monkeypatch, unsupported, False, "planned-budget-v7"
    )
    gateway = instances[0]
    details = reports[0]["evaluation"]["faithfulness"]["details"]
    assert details["factual_claim_count"] == 7
    assert details["coverage"]["complete"]
    if unsupported:
        assert details["unsupported_count"] > 0
        assert not reports[0]["delivery"]["publishable"]
    else:
        assert details["supported_count"] == 7
        assert details["unsupported_count"] == 0
    assert gateway.sent
    journal = json.loads((tmp_path / "repair.json.writer-v2.json").read_text())
    body_calls = [r for name, r in journal["calls"].items() if name.startswith("body:")]
    assert len(body_calls) == 5
    assert all(r["reference_contract"] == REFERENCE_SET_CONTRACT for r in body_calls)
    assert all(
        len(json.loads(r["raw"])["units"][0]["fact_card_refs"]) == 2 for r in body_calls
    )
    assert all(
        len(r["reference_canonical"][name][0]["sentences"][0]["fact_card_refs"]) == 1
        for name, r in [
            (n.split(":")[1], r)
            for n, r in journal["calls"].items()
            if n.startswith("body:")
        ]
    )
    inventories = [
        p for p in gateway.evaluator.base.calls if "allowed_citation_aliases" in p
    ]
    assert inventories
    assert all(p["source_units"] for p in inventories)
    assert len(journal.get("candidates", [])) <= 2


@pytest.mark.asyncio
async def test_received_references_survive_body_summary_and_targeted_patch(
    case, tmp_path
):
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summaries = await session.summaries(body, "summary_initial")
        received, origins = session.reference_input(body)
        assert received != body and len(origins) == 5
        assert session.reference_input(summaries)[0] != summaries
        target = {
            "m": {
                "path": ["methods", 0, 0],
                "original": body["methods"][0]["sentences"][0],
                "explanation": "Clarify the supported comparison.",
            }
        }
        candidate = await session.patch(body, target, cause="evaluation")
        raw_candidate, _ = session.reference_input(candidate)
        assert raw_candidate["introduction"] == received["introduction"]
        assert (
            raw_candidate["methods"][0]["sentences"][0]["fact_card_refs"]
            == [case.by_eid["b"]] * 2
        )
        assert len(session.state["attempts"]) == 1
        summary_target = {
            "a": {
                "path": ["abstract", 0, 0],
                "original": summaries["abstract"][0]["sentences"][0],
                "explanation": "Clarify the frozen finding.",
            }
        }
        revised = await session.patch(
            summaries, summary_target, cause="evaluation", summary_body=body
        )
        assert (
            session.reference_input(revised)[0]["conclusion"]
            == session.reference_input(summaries)[0]["conclusion"]
        )
        assert len(session.state["attempts"]) == 2
        assert len(gateway.sent) == 8
        for sent in gateway.sent:
            assert (
                sent["payload"]["writing_envelope"]["reference_contract"]
                == REFERENCE_SET_CONTRACT
            )
            assert REFERENCE_SET_INSTRUCTION in sent["request"]["input"][0]["content"]
        with pytest.raises(WriterError, match="repair_limit"):
            await session.patch(body, target, cause="evaluation")
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "raw", "source_path"])
async def test_missing_or_tampered_received_provenance_stops_before_next_request(
    case, tmp_path, fault
):
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        if fault == "missing":
            session.journal["reference_inputs"].pop(digest(body))
        elif fault == "raw":
            session.journal["calls"]["body:introduction"]["raw"] += " "
        else:
            record = session.journal["reference_inputs"][digest(body)]
            record["origins"][0]["source_path"][-1] = False
            record["identity"] = digest(
                {k: v for k, v in record.items() if k != "identity"}
            )
        before = len(gateway.sent)
        with pytest.raises(
            WriterError, match="writer_reference_input_identity_invalid"
        ):
            await session.summaries(body, "summary_initial")
        assert len(gateway.sent) == before
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_unknown_reference_keeps_original_error_and_never_normalizes_into_evidence(
    case, tmp_path
):
    gateway = DuplicateGateway(case)
    gateway.unknown_reference = True
    owner, session = await setup(case, tmp_path, gateway)
    try:
        with pytest.raises(WriterError, match="writer_reference_invalid"):
            await session.initial_body()
        assert len(gateway.sent) == 1
        assert not session.journal.get("reference_inputs")
        assert not session.state["attempts"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_regenerated_same_sha_and_failed_patch_keep_original_occurrence_ownership(
    case, tmp_path
):
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summary = await session.summaries(body, "summary_initial")
        original = copy.deepcopy(session.journal["reference_inputs"][digest(summary)])
        # Repeat the same canonical response with different received multiplicity.
        gateway.duplicate_count = 2
        assert await session.summaries(body, "summary_refresh") == summary
        assert session.journal["reference_inputs"][digest(summary)] == original
        assert (
            len(
                json.loads(session.journal["calls"]["summary_refresh"]["raw"])[
                    "abstract"
                ][0]["body_unit_refs"]
            )
            == 3
        )
        gateway.original_patch = True
        gateway.duplicate_count = 1
        target = {
            "m": {
                "path": ["methods", 0, 0],
                "original": body["methods"][0]["sentences"][0],
                "explanation": "An unchanged response must not replace the original.",
            }
        }
        original_body = copy.deepcopy(session.journal["reference_inputs"][digest(body)])
        with pytest.raises(WriterError, match="writer_repair_no_progress"):
            await session.patch(body, target, cause="evaluation")
        assert session.journal["reference_inputs"][digest(body)] == original_body
    finally:
        await owner.close()


def duplicate_document(body):
    """Build the exact received view emitted by DuplicateGateway."""
    result = copy.deepcopy(body)
    for paragraphs in result.values():
        if isinstance(paragraphs, list):
            for paragraph in paragraphs:
                for unit in paragraph["sentences"]:
                    field = (
                        "fact_card_refs"
                        if "fact_card_refs" in unit
                        else "body_unit_refs"
                    )
                    unit[field] += unit[field][:1]
    return result


@pytest.mark.asyncio
async def test_cumulative_received_body_bytes_cannot_be_hidden_by_dedup(case, tmp_path):
    canonical_bytes = nested_json_size(case.body)
    received_bytes = nested_json_size(duplicate_document(case.body))
    assert received_bytes > canonical_bytes
    capacity = CapacityConfig(body_json_bytes=(canonical_bytes + received_bytes) // 2)
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway, capacity=capacity)
    try:
        with pytest.raises(EnvelopeExceeded) as caught:
            await session.initial_body()
        assert caught.value.diagnostics[0]["field"] == "nested_json_bytes"
        assert caught.value.diagnostics[0]["actual"] == received_bytes
        assert len(gateway.sent) == 5
        assert not session.state["attempts"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_complete_patch_candidate_counts_untouched_received_references(
    case, tmp_path
):
    replacement = "A revised source-bound relation-network comparison under the recorded conditions."
    candidate = copy.deepcopy(case.body)
    candidate["methods"][0]["sentences"][0]["text"] = replacement
    received = duplicate_document(case.body)
    received_candidate = duplicate_document(candidate)
    maximum = max(nested_json_size(received), nested_json_size(candidate))
    assert maximum < nested_json_size(received_candidate)
    gateway = DuplicateGateway(case)
    gateway.patch_text = replacement
    owner, session = await setup(
        case, tmp_path, gateway, capacity=CapacityConfig(body_json_bytes=maximum)
    )
    try:
        body = await session.initial_body()
        original = copy.deepcopy(session.journal["reference_inputs"][digest(body)])
        target = {
            "m": {
                "path": ["methods", 0, 0],
                "original": body["methods"][0]["sentences"][0],
                "explanation": "Clarify only the target.",
            }
        }
        with pytest.raises(EnvelopeExceeded) as caught:
            await session.patch(body, target, cause="evaluation")
        assert caught.value.diagnostics[0]["actual"] == nested_json_size(
            received_candidate
        )
        assert nested_json_size(candidate) <= maximum
        assert session.journal["reference_inputs"][digest(body)] == original
        assert digest(candidate) not in session.journal["reference_inputs"]
        assert len(session.state["attempts"]) == 1
    finally:
        await owner.close()


def marked_feedback_fixture(value):
    """Rebind a real failed patch receipt to the new explicit offline contract."""
    import hashlib

    from test_repair_feedback import replay_034

    from litagent.safety.planned_budget import _digest
    from litagent.writing_flat import parse

    survey, bindings, prior, targets, _ = replay_034(value)
    document = copy.deepcopy(value["original"])
    for doc, binding in (
        (document, bindings),
        (prior["writer_document"], prior["bindings"]),
    ):
        scope = doc["budget_envelope"]
        scope["reference_contract"] = REFERENCE_SET_CONTRACT
        scope["identity"] = _digest({k: v for k, v in scope.items() if k != "identity"})
        binding["writer_envelope_identity"] = scope["identity"]
    receipt = prior["repair_receipt"]
    received = json.loads(receipt["raw"])
    unit = next(p for p in received["patches"] if p.get("fact_card_refs"))
    unit["fact_card_refs"].append(unit["fact_card_refs"][0])
    receipt["raw"] = json.dumps(received)
    receipt["raw_sha256"] = hashlib.sha256(receipt["raw"].encode()).hexdigest()
    parsed = parse(
        receipt["raw"], receipt["schema"], reference_contract=REFERENCE_SET_CONTRACT
    )
    assert digest(parsed["canonical"]) == receipt["canonical_sha256"]
    receipt.update(
        reference_contract=REFERENCE_SET_CONTRACT,
        wire_bindings_sha256=digest(parsed["bindings"]),
        received_canonical_sha256=digest(parsed["received_canonical"]),
    )
    return survey, bindings, document, prior, targets


@pytest.fixture
def feedback_case():
    """Load the actual 034 lossless public regression producer."""
    from test_repair_feedback import real_034

    return real_034.__wrapped__()


@pytest.mark.parametrize("fault", [None, "receipt", "current", "binding"])
def test_next_feedback_replays_same_reference_policy_and_rejects_tampering(
    feedback_case, fault
):
    from litagent.writing_v2 import repair_feedback_directory

    survey, bindings, document, prior, targets = marked_feedback_fixture(feedback_case)
    if fault == "receipt":
        prior["repair_receipt"].pop("reference_contract")
    elif fault == "current":
        document["budget_envelope"].pop("reference_contract")
    elif fault == "binding":
        prior["repair_receipt"]["wire_bindings_sha256"] = "0" * 64
    args = (
        feedback_case["original"]["packet"],
        survey,
        document,
        bindings,
        prior,
        feedback_case["previous_failure"],
        targets,
    )
    if fault is None:
        result = repair_feedback_directory(*args)
        assert result["contract"] == "host-repair-feedback-directory-v1"
        assert result["prior_report_sha256"] == prior["report_sha256"]
        assert result["fragments"]
    else:
        with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
            repair_feedback_directory(*args)
