"""Exercise structured repair guidance through actual SDK and report consumers."""

import copy
import json

import pytest
import test_evaluation_stability_pipeline as stability
import test_scientific_repair_pipeline as scientific
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, run_source_case, sent_with
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.writing_v2 import WriterError, body_units, writer_budget_plan

CONTRACT = "host-structured-repair-v1"
FLAGS = {
    **SELECTION,
    "repair_mapping_contract": "capacity-source-map-v3",
    "repair_guidance_contract": CONTRACT,
}
LOCAL_MARKER = "LOCAL_ONLY_DIAGNOSTIC_"


def select_structured_policy(monkeypatch, *, module=None):
    """Forward only the trusted selector without replacing actual consumers."""
    import litagent.frozen_writing as frozen

    for target in ([frozen, module] if module is not None else [frozen]):
        real_plan, real_run = target.plan_frozen_budget, target.run_frozen

        def plan(*args, _original=real_plan, **kwargs):
            for key, value in FLAGS.items():
                kwargs.setdefault(key, value)
            return _original(*args, **kwargs)

        async def run(*args, _original=real_run, **kwargs):
            for key, value in FLAGS.items():
                kwargs.setdefault(key, value)
            assert kwargs["expected_budget_plan"] is not None
            return await _original(*args, **kwargs)

        monkeypatch.setattr(target, "plan_frozen_budget", plan)
        monkeypatch.setattr(target, "run_frozen", run)


def long_note(label):
    """Produce one legal free explanation that must remain local verbatim."""
    prefix = LOCAL_MARKER + label + ":"
    return (prefix + "x" * 2000)[:2000]


def add_local_evaluation_notes(gateway):
    """Modify provider responses only, retaining raw outbound SDK records."""
    original = gateway.evaluator.chat
    gateway.local_notes = []

    async def chat(messages, **kwargs):
        response = await original(messages, **kwargs)
        value = json.loads(response.content)
        for decision in value.get("decisions", []):
            if decision["status"] != "unsupported":
                continue
            note = long_note(f"evaluation:{len(gateway.local_notes)}")
            decision["explanation"] = note
            gateway.local_notes.append(note)
            for aspect in decision["aspect_checks"]:
                if aspect["supported"]:
                    continue
                note = long_note(f"aspect:{len(gateway.local_notes)}")
                aspect["explanation"] = note
                gateway.local_notes.append(note)
        response.content = json.dumps(value)
        return response

    gateway.evaluator.chat = chat


def assert_guidance(entry, *, prior):
    """Check actual model input coverage without treating guidance as science."""
    payload = entry["payload"]
    guidance = payload["repair_guidance"]
    assert guidance["contract"] == CONTRACT
    assert guidance["current"]["coverage"]["complete"] is True
    assert guidance["current"]["coverage"]["diagnostics_total"] > 0
    assert guidance["current"]["coverage"]["issues_total"] == len(
        guidance["current"]["issues"]
    )
    assert (guidance["prior_candidate"] is not None) is prior
    if prior:
        previous = guidance["prior_candidate"]
        assert previous["coverage"]["complete"] is True
        assert previous["coverage"]["diagnostics_total"] > 0
        assert previous["coverage"]["issues_total"] == len(previous["issues"])
        assert (
            previous["source_report_sha256"]
            != guidance["current"]["source_report_sha256"]
        )
    assert LOCAL_MARKER not in json.dumps(entry["request"])
    for tid, target in payload["targets"].items():
        assert "issue_refs" not in target and "task" not in target
        assert any(
            tid in issue["target_ids"]
            for origin in ("current", "prior_candidate")
            if guidance[origin] is not None
            for issue in guidance[origin]["issues"]
        )
    return guidance


def assert_local_notes_retained(tmp_path, gateway):
    """Require full recorded provider explanations, not merely an omitted string."""
    paths = [tmp_path / "repair.json", tmp_path / "repair.json.writer-v2.json"]
    paths.extend((tmp_path / "evaluation").glob("*/progress.json"))
    persisted = "\n".join(path.read_text() for path in paths if path.exists())
    assert gateway.local_notes
    for note in gateway.local_notes:
        assert len(note) == 2000 and note in persisted


class LongScientificGateway(scientific.ScientificRepairGateway):
    """Keep real failed facts while making only their explanations lengthy."""

    def __init__(self, case, failing_candidates, *, reviewer=False):
        super().__init__(case, failing_candidates)
        self.reviewer = reviewer
        add_local_evaluation_notes(self)

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if self.reviewer and "issues" in native_schema["properties"]:
            uid = next(uid for uid in payload["units"] if ":methods:" in uid)
            note = long_note("reviewer")
            self.local_notes.append(note)
            return {
                "issues": [{"unit_id": uid, "explanation": note}],
                "explanation": "A located reviewer concern needs independent review.",
            }
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_candidates,reviewer", [(0, False), (1, True), (2, False)]
)
async def test_current_and_prior_guidance_keep_local_explanations_and_two_repairs(
    case, tmp_path, monkeypatch, failing_candidates, reviewer
):
    """Every candidate crosses full assessment while current concerns stay present."""
    gateway = LongScientificGateway(case, failing_candidates, reviewer=reviewer)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_structured_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    _, journal, budget = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    expected = min(2, failing_candidates + 1)
    assert len(repairs) == gateway.repair_count == expected
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    original_sha = None
    for index, entry in enumerate(repairs):
        guidance = assert_guidance(entry, prior=index > 0)
        current = guidance["current"]
        original_sha = original_sha or current["source_report_sha256"]
        assert current["source_report_sha256"] == original_sha
        if reviewer:
            assert {issue["category"] for issue in current["issues"]} >= {
                "evaluation",
                "reviewer",
            }
        assert "FLOPs by 44%" in json.dumps(entry["payload"]["current_body"])
        assert "FLOPs by 45%" not in json.dumps(entry["payload"]["current_body"])
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "body_units")) == expected + 1
    assert_local_notes_retained(tmp_path, gateway)
    assert budget["requests_sent"] == len(gateway.sent)
    assert budget["usage_uncertain"] is False
    assert all(entry["status"] == "settled" for entry in budget["entries"].values())
    assert journal["candidates"]
    if failing_candidates < 2:
        assert report["delivery"]["publishable"] is True
        final = report["evaluation"]["faithfulness"]["details"]
        assert final["coverage"]["complete"] is True
        assert final["supported_count"] == final["factual_claim_count"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert "FLOPs by" not in report["survey"]
    else:
        assert report["delivery"]["publishable"] is False
        assert "FLOPs by 44%" in report["survey"]
        assert not any(candidate["committed"] for candidate in journal["candidates"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "wrong_paper", "missing_core"])
async def test_capacity_guidance_never_exempts_source_or_core_checks(
    case, tmp_path, monkeypatch, mode
):
    """A compact capacity diagnosis does not authorize unsupported delivery."""
    select_structured_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, mode=mode)
    _, journal, budget = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    assert 1 <= len(repairs) <= 2
    guidance = assert_guidance(repairs[0], prior=False)
    assert {issue["category"] for issue in guidance["current"]["issues"]} == {
        "capacity"
    }
    originals = journal["structured_capacity_inputs"]
    assert originals
    for identity, recorded in originals.items():
        details = recorded["evaluation_details"]
        assert identity == digest(details)
        assert recorded["inventory_items_total"] == len(details["assertions"]) == 144
        assert recorded["inventory_sha256"] == digest(details["assertions"])
        assert recorded["source_report_sha256"] == details["source_report_sha256"]
        assert recorded["disposition"] == "inventory_not_scientifically_assessed"
    assert (report["delivery"]["publishable"] is True) is (mode == "valid")
    if mode != "valid":
        assert not any(candidate["committed"] for candidate in journal["candidates"])
    assert budget["requests_sent"] == len(gateway.sent)
    assert budget["usage_uncertain"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("science_failure", [False, True])
async def test_inventory_and_structured_prose_repair_share_two_attempts(
    case, tmp_path, monkeypatch, science_failure
):
    """The unchanged inventory protocol and one Pro edit share the same limit."""
    select_structured_policy(monkeypatch, module=stability.mixed)
    gateway = stability.StableGateway(case, science_failure=science_failure)
    report, _ = await stability.execute(case, tmp_path, monkeypatch, gateway)
    _, journal, budget = products(tmp_path)
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    assert sum("purpose" in attempt for attempt in attempts) == 1
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_guidance(repairs[0], prior=False)
    assert len(sent_with(gateway, "inventory_recovery")) == 1
    assert (report["delivery"]["publishable"] is True) is not science_failure
    if science_failure:
        assert not any(candidate["committed"] for candidate in journal["candidates"])
    assert budget["requests_sent"] == len(gateway.sent)
    assert budget["usage_uncertain"] is False


@pytest.mark.asyncio
async def test_unknown_structured_repair_keeps_hold_and_never_retries(
    case, tmp_path, monkeypatch
):
    """A missing provider usage record remains an occupied, nonreplayable call."""
    select_structured_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, unknown=True)
    _, journal, budget = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_guidance(repairs[0], prior=False)
    assert report["delivery"]["publishable"] is False
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert budget["usage_uncertain"] is True
    assert budget["stop_reason"] == "strict_usage_unknown"
    unknown = [
        e for e in budget["entries"].values() if e["sent"] and e["status"] != "settled"
    ]
    assert len(unknown) == 1
    assert unknown[0]["actual_tokens"] is None and unknown[0]["reserved_tokens"] > 0
    assert any(
        record["inventory_items_total"] == 144
        and record["inventory_sha256"]
        == digest(record["evaluation_details"]["assertions"])
        for record in journal["structured_capacity_inputs"].values()
    )


class LocalGateway(FlatGateway):
    """Answer explicitly requested local primitives without a scientific verdict."""

    def __init__(self, case):
        super().__init__(case)
        self.local_calls = 0

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if payload.get("cause") == "local":
            self.local_calls += 1
            for patch in value["patches"]:
                patch["text"] = (
                    "Relation networks learn a comparison function."
                    if self.local_calls == 1
                    else "In the cited setting, Relation networks learn a comparison function."
                )
                patch["fact_card_refs"] = [self.case.by_eid["b"]]
        return value


@pytest.mark.asyncio
async def test_explicit_local_primitives_have_body_paths_without_fabricated_offsets(
    case, tmp_path
):
    """Two explicit local calls test the common limit, not an automatic retry rule."""
    gateway = LocalGateway(case)
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    try:
        original = await session.initial_body()
        target = {
            "writer:methods:0:0": {
                "path": ["methods", 0, 0],
                "original": copy.deepcopy(original["methods"][0]["sentences"][0]),
                "explanation": long_note("explicit-local-target"),
            }
        }
        for _ in range(2):
            candidate = await session.patch(original, target, cause="local")
            assert candidate != original
        repairs = sent_with(gateway, "targets")
        assert len(repairs) == 2 and len(session.state["attempts"]) == 2
        for entry in repairs:
            packet = assert_guidance(entry, prior=False)["current"]
            assert packet["source_report_sha256"] is None
            assert packet["source_bindings_sha256"] is None
            assert {issue["category"] for issue in packet["issues"]} == {"local"}
            assert all(issue["body_paths"] for issue in packet["issues"])
        before = copy.deepcopy(budget.snapshot())
        with pytest.raises(WriterError, match="writer_repair_limit"):
            await session.patch(original, target, cause="local")
        assert budget.snapshot() == before and gateway.local_calls == 2
        assert session.journal["structured_guidance_contexts"]
    finally:
        await owner.close()


class SummaryGateway(FlatGateway):
    """Return a bad summary fact and a valid summary-only HTTP replacement."""

    def __init__(self, case):
        super().__init__(case)
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "cited_materials" in payload:
                claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
                for decision in value["decisions"]:
                    if "FLOPs by 44%" not in claims[decision["claim_id"]]["claim_text"]:
                        continue
                    decision.update(
                        status="unsupported",
                        reason_code="insufficient_support",
                        support_basis=[],
                    )
                    for aspect in decision["aspect_checks"]:
                        aspect.update(supported=False, support_ids=[])
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat
        add_local_evaluation_notes(self)

    def answer(self, payload, native_schema):
        value = super().answer(payload, native_schema)
        if "body_units" in payload and "targets" not in payload:
            value["abstract"][0]["text"] = "Relation networks reduce FLOPs by 44%."
        if "targets" in payload:
            assert "body_units" in payload
            for patch in value["patches"]:
                patch["text"] = "Relation networks learn a comparison function."
                patch["body_unit_refs"] = ["u3"]
        return value


@pytest.mark.asyncio
async def test_summary_guidance_keeps_frozen_body_and_requires_complete_evaluation(
    case, tmp_path, monkeypatch
):
    """Summary dependency IDs retain their own body premises and scientific gate."""
    gateway = SummaryGateway(case)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_structured_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, 0)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_guidance(repairs[0], prior=False)
    initial = next(
        row
        for row in sent_with(gateway, "body_units")
        if "targets" not in row["payload"]
    )
    assert repairs[0]["payload"]["body_units"] == initial["payload"]["body_units"]
    assert (
        body_units(report["metadata"]["writer_document"]["body"])
        == initial["payload"]["body_units"]
    )
    assert report["delivery"]["publishable"] is True
    assert "FLOPs by" not in report["survey"]
    assert_local_notes_retained(tmp_path, gateway)
    _, journal, _ = products(tmp_path)
    assert len(journal["candidates"]) == 1 and journal["candidates"][0]["committed"]


class UnlocatedGlobalGateway(FlatGateway):
    """Keep a valid complete assessment with a global unresolved requirement."""

    def __init__(self, case):
        super().__init__(case)
        self.local_notes = []
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            value = json.loads(response.content)
            if "coverage" in value:
                note = long_note("unlocated-global")
                self.local_notes.append(note)
                value["coverage"][0].update(
                    answered=False, claim_ids=[], explanation=note
                )
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


@pytest.mark.asyncio
async def test_unlocated_global_problem_remains_explicitly_unrepairable(
    case, tmp_path, monkeypatch
):
    """An explanation without a host location is retained rather than filtered."""
    gateway = UnlocatedGlobalGateway(case)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_structured_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, 0)
    assert report["delivery"]["publishable"] is False
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"] is True
    assert any(row["answered"] is False for row in details["core_coverage"])
    assert not sent_with(gateway, "targets")
    assert (
        report["metadata"]["writer_failure"]["code"] == "writer_guidance_unrepairable"
    )
    assert_local_notes_retained(tmp_path, gateway)
