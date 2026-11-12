"""Exercise claim-owned aspects through real request, SDK and host consumers.

Only the external HTTP peer supplies controlled responses. Native validity is a
protocol property and never substitutes for scientific or delivery acceptance.
"""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
import test_semantic_conflict_pipeline as pipeline
from jsonschema import Draft202012Validator
from test_claim_response_contract import allowed_for, complete_reply, permissions
from test_claim_response_pipeline import BoundResponseGateway, decision_branches
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_flat_writer_pipeline import FlatGateway
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_native_responses import owner_for
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.eval import batched, scoped
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest
from litagent.runner import LitAgent
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import planned_backend
from litagent.writing_v2 import BODY, writer_budget_plan

V2 = batched.EVALUATION_RESPONSE_CONTRACT_V2
PAIR = ("evaluation_response_contract", "evaluation_response_identity")


def aspect_flags():
    """Select the revision through the existing complete frozen parent stack."""
    return {**pipeline.FLAGS, "evaluation_response_contract": V2}


@pytest.fixture
def aspect_payload(permissions, case):
    """Keep equal sources with unequal aspects, disjoint sources and empty sets."""
    value = copy.deepcopy(permissions)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **aspect_flags(),
    )
    assert plan["status"] == "admitted"
    value["writer_envelope"] = plan["envelope"]
    for claim, required in zip(
        value["assertions"],
        [["predicate"], ["predicate", "conditions"], ["quantities"], ["predicate"], []],
        strict=True,
    ):
        claim["required_support_aspects"] = required
    value["schema"] = batched.assessment_schema(value)
    return value


def assert_own_permissions(schema, payload, aliases=None):
    """Inspect the actual final native branches against canonical host ownership."""
    branches = decision_branches(schema)
    by_claim = {}
    for branch in branches:
        for cid in branch["properties"]["claim_id"]["enum"]:
            assert cid not in by_claim
            by_claim[cid] = branch
    assert set(by_claim) == {claim["claim_id"] for claim in payload["assertions"]}
    to_wire = {canonical: alias for alias, canonical in (aliases or {}).items()}
    for claim in payload["assertions"]:
        branch = by_claim[claim["claim_id"]]
        properties = branch["properties"]
        checks = properties["aspect_checks"]
        basis = properties["support_basis"]
        allowed = {to_wire.get(pid, pid) for pid in allowed_for(payload, claim)}
        if allowed:
            assert set(basis["items"]["properties"]["support_id"]["enum"]) == allowed
            assert (
                set(checks["items"]["properties"]["support_ids"]["items"]["enum"])
                == allowed
            )
        else:
            assert basis["maxItems"] == 0
            assert checks["items"]["properties"]["support_ids"]["maxItems"] == 0
        required = set(claim["required_support_aspects"])
        if required:
            assert set(checks["items"]["properties"]["aspect"]["enum"]) == required
        else:
            assert checks["maxItems"] == 0
    assert '"enum": []' not in json.dumps(schema)
    return by_claim


@pytest.mark.asyncio
async def test_both_final_candidates_bind_own_sources_and_aspects_before_sdk_send(
    aspect_payload, monkeypatch
):
    """Measure unchanged real cost logic, then send both exact candidates via SDK."""
    measured = []
    original = scoped.strict_request_reservation
    before = copy.deepcopy(aspect_payload)

    def measure(messages, output, native):
        cost = original(messages, output, native)
        measured.append((copy.deepcopy(messages), copy.deepcopy(native), cost))
        return cost

    monkeypatch.setattr(scoped, "strict_request_reservation", measure)
    chosen = scoped.build_evaluation_request(
        batched.batch_prompt(aspect_payload), aspect_payload
    )
    assert len(measured) == 2
    assert chosen in [(messages, native) for messages, native, _ in measured]
    assert strict_request_reservation(chosen[0], 0, chosen[1]) == min(
        cost for _, _, cost in measured
    )
    assert aspect_payload == before
    aliases = batched.batch_support_aliases(aspect_payload)
    sent = []
    expected = []
    namespaces = set()
    for messages, native, _ in measured:
        wire = json.loads(messages[1]["content"])
        if set(wire["support_passages"]) == set(aspect_payload["support_passages"]):
            selected_aliases = None
            namespaces.add("canonical")
        else:
            assert set(wire["support_passages"]) == set(aliases)
            selected_aliases = aliases
            namespaces.add("alias")
        schema = native["json_schema"]["schema"]
        Draft202012Validator.check_schema(schema)
        groups = assert_own_permissions(schema, aspect_payload, selected_aliases)
        first, second = aspect_payload["assertions"][:2]
        assert allowed_for(aspect_payload, first) == allowed_for(aspect_payload, second)
        assert groups[first["claim_id"]] is not groups[second["claim_id"]]
        reply = complete_reply(aspect_payload, native, selected_aliases)
        validator = Draft202012Validator(schema)
        validator.validate(reply)
        hostile = copy.deepcopy(reply)
        hostile["decisions"][0]["aspect_checks"].append(
            copy.deepcopy(reply["decisions"][1]["aspect_checks"][1])
        )
        assert not validator.is_valid(hostile)
        foreign = reply["decisions"][2]["support_basis"][0]["support_id"]
        for field in ("support_basis", "aspect_checks"):
            hostile = copy.deepcopy(reply)
            if field == "support_basis":
                hostile["decisions"][0][field] = [{"support_id": foreign}]
            else:
                hostile["decisions"][0][field][0]["support_ids"] = [foreign]
            assert not validator.is_valid(hostile)
        assert wire["original_supports"] == aspect_payload["original_supports"]
        expected.append((messages, native, reply))
    assert namespaces == {"canonical", "alias"}

    async def transport(request):
        body = json.loads(request.content)
        messages, native, reply = expected[len(sent)]
        assert body["messages"] == messages
        assert body["response_format"] == native
        assert body["model"] == "lite"
        sent.append(body)
        return httpx.Response(
            200,
            json={
                "id": "controlled-aspect-candidate",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(reply)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, budget, _ = await owner_for(transport)
    try:
        client = ModelBoundClient(owner, "lite", "evaluation")
        for messages, native, reply in expected:
            response = await client.chat(
                messages, response_format=native, max_tokens=16384
            )
            assert json.loads(response.content) == reply
            raw = response.content
            decoded = batched.decode_assessment_response(raw, aspect_payload)
            wire = json.loads(messages[1]["content"])
            restored = (
                batched._restore_batch_support_ids(decoded, aliases)
                if set(wire["support_passages"]) == set(aliases)
                else decoded
            )
            for claim, decision in zip(
                aspect_payload["assertions"], restored.decisions, strict=True
            ):
                assert decision.claim_id == claim["claim_id"]
                assert decision.classification_valid is (
                    decision.correct_kind == claim["kind"]
                )
                allowed = allowed_for(aspect_payload, claim)
                assert [entry.support_id for entry in decision.support_basis] == (
                    allowed[:1]
                )
                assert [check.aspect for check in decision.aspect_checks] == (
                    claim["required_support_aspects"]
                )
                assert all(
                    check.support_ids == allowed[:1] for check in decision.aspect_checks
                )
            assert response.content == raw
            assert json.loads(raw) == reply
        assert len(sent) == 2 and budget.used == 240
        assert not budget.snapshot()["usage_uncertain"]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "missing_contract",
        "missing_identity",
        "wrong_identity",
        "v1_identity",
    ],
)
async def test_bad_response_pair_is_rejected_before_any_sdk_http(aspect_payload, fault):
    """A self-rehashed envelope cannot launder an incomplete or cross-version pair."""
    invalid = copy.deepcopy(aspect_payload)
    envelope = invalid["writer_envelope"]
    if fault == "unknown":
        envelope[PAIR[0]] = "host-bound-evaluation-unknown"
    elif fault == "missing_contract":
        envelope.pop(PAIR[0])
    elif fault == "missing_identity":
        envelope.pop(PAIR[1])
    elif fault == "v1_identity":
        envelope[PAIR[1]] = batched.response_contract_identity()
    else:
        envelope[PAIR[1]] = "0" * 64
    envelope["identity"] = digest(
        {key: value for key, value in envelope.items() if key != "identity"}
    )
    before = copy.deepcopy(invalid)
    sent = []

    async def transport(request):
        sent.append(request)
        raise AssertionError("Invalid permissions reached HTTP.")

    owner, budget, _ = await owner_for(transport)
    try:
        with pytest.raises(ValueError, match="^evaluation_response_contract_invalid$"):
            messages, native = scoped.build_evaluation_request(
                batched.batch_prompt(invalid), invalid
            )
            await ModelBoundClient(owner, "lite", "evaluation").chat(
                messages, response_format=native, max_tokens=16384
            )
        assert not sent and budget.used == 0
        assert budget.snapshot()["requests_sent"] == 0
        assert invalid == before
    finally:
        await owner.close()


class AspectGateway(FlatGateway):
    """Supply both compliant and schema-ignoring replies only inside HTTP Mock."""

    def __init__(self, case, *, fault=None):
        super().__init__(case, declined=fault is not None)
        self.aspect_fault = fault
        self.repair_count = 0
        self.assessments = []
        self.injected = []
        original = self.evaluator.chat

        async def chat(messages, **options):
            response = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            schema = options["response_format"]["json_schema"]["schema"]
            value = json.loads(response.content)
            if all(
                "classification_valid" not in branch["properties"]
                for branch in decision_branches(schema)
            ):
                # The reusable old HTTP fixture recognizes the former definition
                # name. Honor the actual selected provider shape at this boundary.
                for row in value["decisions"]:
                    row.pop("classification_valid", None)
            assert_own_permissions(schema, payload)
            claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
            if fault and not self.injected:
                for row in value["decisions"]:
                    claim = claims[row["claim_id"]]
                    required = claim["required_support_aspects"]
                    if fault == "extra" and "quantities" not in required:
                        extra = copy.deepcopy(row["aspect_checks"][0])
                        extra["aspect"] = "quantities"
                        row["aspect_checks"].append(extra)
                    elif fault == "missing" and len(required) >= 2:
                        row["aspect_checks"].pop()
                    elif fault == "duplicate" and required:
                        row["aspect_checks"].append(
                            copy.deepcopy(row["aspect_checks"][0])
                        )
                    elif (
                        fault in {"unsupported", "unverifiable"}
                        and "On DeiT-B," in claim["claim_text"]
                    ):
                        row.update(
                            status=fault,
                            reason_code=(
                                "insufficient_support"
                                if fault == "unsupported"
                                else "unverifiable"
                            ),
                            explanation="The supplied source says DeiT-S, not DeiT-B.",
                        )
                        condition = next(
                            a
                            for a in row["aspect_checks"]
                            if a["aspect"] == "conditions"
                        )
                        condition.update(
                            supported=False,
                            support_ids=[],
                            explanation="The asserted training condition is not established.",
                        )
                    else:
                        continue
                    self.injected.append(copy.deepcopy(row))
                    break
            validator = Draft202012Validator(schema)
            native_valid = validator.is_valid(value)
            if (
                fault == "extra"
                and self.injected[-1:]
                and self.injected[0] in value["decisions"]
            ):
                assert not native_valid
            else:
                validator.validate(value)
            response.content = json.dumps(value)
            self.assessments.append(
                {
                    "payload": copy.deepcopy(payload),
                    "schema": copy.deepcopy(schema),
                    "value": copy.deepcopy(value),
                    "raw": response.content,
                    "native_valid": native_valid,
                }
            )
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Decline prose edits rather than manufacture a passing negative case."""
        if "targets" in payload:
            self.repair_count += 1
        return super().answer(payload, native_schema)


async def run_aspect_case(case, tmp_path, monkeypatch, *, fault=None):
    """Keep the full frozen planner, writer, SDK, runner and host guards real."""
    for section, text in zip(
        BODY[:2],
        ["EViT reduces computation.", "On DeiT-S, EViT reduces computation."],
        strict=True,
    ):
        case.body[section][0]["sentences"][0].update(
            text=f"{section}: {text}", fact_card_refs=[case.by_eid["a"]]
        )
    if fault in {"unsupported", "unverifiable"}:
        case.body[BODY[1]][0]["sentences"][0][
            "text"
        ] = f"{BODY[1]}: On DeiT-B, EViT reduces computation."
    monkeypatch.setattr(pipeline, "FLAGS", aspect_flags())
    gateway = AspectGateway(case, fault=fault)
    monkeypatch.setattr(pipeline, "ConflictGateway", lambda *_args, **_kwargs: gateway)
    report, _, snapshot = await pipeline.run_conflict_case(case, tmp_path, monkeypatch)
    state, journal, ledger = products(tmp_path)
    assert ledger == snapshot
    assert ledger["requests_sent"] == ledger["call_count"] == len(gateway.sent)
    assert not ledger["usage_uncertain"]
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    envelope = ledger["plan"]["envelope"]
    assert envelope[PAIR[0]] == V2
    assert envelope[PAIR[1]] == batched.response_contract_identity(contract=V2)
    assert batched.selected_response_contract(envelope)
    assert ledger["plan"]["maximum_tokens"] == 4500000
    assert ledger["plan"]["maximum_requests"] == 64
    assert ledger["plan"]["repair_limit"] == 2
    assert sent_with(gateway, "cited_materials") and gateway.assessments
    return report, gateway, state, journal, ledger


def assert_raw_assessments_saved(tmp_path, gateway):
    """Require durable progress to preserve provider bytes and every aspect item."""
    received = [
        request
        for path in (tmp_path / "evaluation").glob("*/progress.json")
        for request in json.loads(path.read_text(encoding="utf-8"))["progress"][
            "requests"
        ]
        if request["phase"] == "assessment"
    ]
    for reply in gateway.assessments:
        matches = [
            request for request in received if request.get("response") == reply["raw"]
        ]
        assert matches and all(r["status"] == "response_received" for r in matches)


@pytest.mark.asyncio
async def test_v2_sdk_and_runner_accept_complete_owned_aspects_without_science_changes(
    case, tmp_path, monkeypatch
):
    """A controlled positive completes independently of the tighter enum shape."""
    report, gateway, _, journal, _ = await run_aspect_case(case, tmp_path, monkeypatch)
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["stages"]["global_check"] == "completed"
    assert details["supported_count"] == details["factual_claim_count"]
    assert details["unsupported_count"] == details["unverified_count"] == 0
    assert report["delivery"]["publishable"] is True
    assert gateway.repair_count == 0 and journal["candidates"] == []
    assert all(reply["native_valid"] for reply in gateway.assessments)
    assert_raw_assessments_saved(tmp_path, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate"])
async def test_host_exact_aspect_set_rejects_schema_ignoring_or_incomplete_provider(
    case, tmp_path, monkeypatch, fault
):
    """Native narrowing never removes the host veto or edits received diagnostics."""
    report, gateway, state, journal, _ = await run_aspect_case(
        case, tmp_path, monkeypatch, fault=fault
    )
    assert gateway.injected, "The fault must reach a real assigned claim."
    details = report["evaluation"]["faithfulness"]["details"]
    assert "support_aspect_coverage_invalid" in json.dumps(details)
    assert report["delivery"]["publishable"] is False
    assert gateway.repair_count == 0 and not sent_with(gateway, "targets")
    assert state["state"]["attempts"] == [] and journal["candidates"] == []
    assert any(
        gateway.injected[0] in reply["value"]["decisions"]
        for reply in gateway.assessments
    )
    assert_raw_assessments_saved(tmp_path, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unsupported", "unverifiable"])
async def test_schema_valid_negative_condition_never_becomes_delivery(
    case, tmp_path, monkeypatch, status
):
    """The source condition stays wrong even with valid aspect and source enums."""
    report, gateway, _, journal, _ = await run_aspect_case(
        case, tmp_path, monkeypatch, fault=status
    )
    assert gateway.injected and all(
        reply["native_valid"] for reply in gateway.assessments
    )
    assert gateway.injected[0]["status"] == status
    assert gateway.injected[0]["support_basis"]
    assert any(not a["supported"] for a in gateway.injected[0]["aspect_checks"])
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"]
    details = initial["faithfulness"]["details"]
    wrong = [row for row in details["assertions"] if "On DeiT-B," in row["claim_text"]]
    assert wrong and all(row["status"] != "supported" for row in wrong)
    assert report["delivery"]["publishable"] is False
    assert not any(candidate["committed"] for candidate in journal["candidates"])
    assert "On DeiT-B," in report["survey"]
    assert_raw_assessments_saved(tmp_path, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_contract", [None, batched.EVALUATION_RESPONSE_CONTRACT])
async def test_correct_v2_pair_cannot_select_a_different_frozen_owner(
    frozen_bundle, owner_contract
):
    """Correct current hashes do not replace the explicit frozen runtime choice."""
    flags = dict(pipeline.FLAGS)
    if owner_contract is not None:
        flags["evaluation_response_contract"] = owner_contract
    plan = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    assert plan["status"] == "admitted"
    envelope = plan["envelope"]
    envelope.update(
        evaluation_response_contract=V2,
        evaluation_response_identity=batched.response_contract_identity(contract=V2),
    )
    for record in (envelope, plan):
        record["identity"] = digest(
            {key: value for key, value in record.items() if key != "identity"}
        )
    before = copy.deepcopy((frozen_bundle, plan))
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="^evaluation_response_frozen_plan_required$"):
        await run_frozen(
            agent,
            frozen_bundle,
            "unstarted-forged-aspect-owner",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **flags,
        )
    assert vars(agent) == {"_wired": False}
    assert (frozen_bundle, plan) == before


@pytest.mark.asyncio
async def test_runner_context_v2_pair_cannot_upgrade_unselected_actual_budget_owner(
    case, tmp_path, monkeypatch
):
    """An external context cannot upgrade the SDK owner's locked response rules."""
    gateway = BoundResponseGateway(case)
    monkeypatch.setattr(pipeline, "ConflictGateway", lambda *_args, **_kwargs: gateway)
    original = LitAgent._build_eval_context
    forged_pairs = []

    def build_context(agent, **kwargs):
        context = original(agent, **kwargs)
        owner = planned_backend(agent._cost_budget)
        assert not batched.selected_response_contract(owner.effective_envelope)
        forged = copy.deepcopy(owner.effective_envelope)
        forged.update(
            evaluation_response_contract=V2,
            evaluation_response_identity=batched.response_contract_identity(
                contract=V2
            ),
        )
        forged["identity"] = digest(
            {key: value for key, value in forged.items() if key != "identity"}
        )
        assert batched.selected_response_contract(forged)
        forged_pairs.append({key: forged[key] for key in PAIR})
        context.update(**forged_pairs[-1], writer_envelope=forged)
        return context

    monkeypatch.setattr(LitAgent, "_build_eval_context", build_context)
    report, _, snapshot = await pipeline.run_conflict_case(case, tmp_path, monkeypatch)
    _, _, ledger = products(tmp_path)
    assert forged_pairs and ledger == snapshot
    assert all(field not in ledger["plan"]["envelope"] for field in PAIR)
    assert report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"]
    assert report["delivery"]["publishable"] is True
    assert gateway.assessment_schemas
    assert all(
        "anyOf" not in schema["properties"]["decisions"]["items"]
        for schema in gateway.assessment_schemas
    )
