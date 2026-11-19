"""Exercise settled native capacity recovery without granting scientific success."""

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from test_global_status_codes import vit_replay
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import PRESET
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine
from litagent.eval.native_capacity import check_inventory_inputs, inventory_frame
from litagent.output_capacity import configured_output_tokens
from litagent.run_limits import configured_run_limits
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, nested_json_size
from litagent.writing_v2 import writer_budget_plan


def configuration():
    """Opt in explicitly; the ordinary profile keeps its existing behavior."""
    config = load_config("config/separate-review-evaluation.yaml")
    config.safety.planned_run_limits.citation_contract = "flat-writer-v2"
    config.eval.native_capacity_recovery = True
    config.observability.enabled = False
    config.llm.base_url = "https://offline.invalid/api/v3"
    return config


def budget_plan(packet, aliases, config, methodology=None):
    """Use the production constructor, including native schemas and all repairs."""
    return writer_budget_plan(
        packet,
        aliases,
        budget_policy="planned-budget-v7",
        methodology=methodology or {},
        capacity=config.capacity,
        run_limits=configured_run_limits(config),
        semantic_engine=configured_engine(config),
        output_tokens=configured_output_tokens(config),
        **PRESET,
    )


def test_inline_summary_bound_covers_longest_cited_and_uncited_examples(case):
    from test_writer_v2 import build_fact_cards, source_selection

    from litagent.format_repair import summary_retry_messages
    from litagent.safety.planned_budget import check_writer_envelope
    from litagent.writing_v2 import (
        BODY,
        RANGE_WRITING_PLAN,
        SCOPE,
        body_units,
        prepare_planned_input,
        provider_response_format,
        request_messages,
        source_packet,
        summary_request_input,
        summary_request_upper_bound,
        validate_body,
    )

    ledger = copy.deepcopy(case.ledger)
    for index in range(8):
        ledger[f"card-{index}"] = {
            **ledger["a"],
            "evidence_id": f"card-{index}",
            "chunk_key": f"section-{index}",
        }
    cards = build_fact_cards(ledger)
    selection = source_selection(cards)
    selection.writing_scope_policy = SCOPE
    packet, aliases = source_packet(
        cards,
        selection,
        case.packet["query"],
        case.packet["requirements"],
        case.packet["language"],
    )
    longest = max(aliases, key=len)
    assert longest == "c10"
    plan = budget_plan(packet, aliases, configuration())
    envelope = plan["envelope"]
    proof = summary_request_upper_bound(packet, aliases, RANGE_WRITING_PLAN, envelope)
    maximum = envelope["body_max_units"]
    assert proof["rows"][-1]["body_units"] == maximum
    for cited in (False, True):
        body = {
            "title": "Valid title",
            **{
                section: [
                    {"sentences": [{"text": "Meaningful prose.", "fact_card_refs": []}]}
                ]
                for section in BODY
            },
        }
        first = body[BODY[0]][0]["sentences"]
        first.extend(copy.deepcopy(first[0]) for _ in range(maximum - len(BODY)))
        if cited:
            body[BODY[-1]][0]["sentences"][0] = {
                "text": f"A final supported premise[C:{longest}]。",
                "fact_card_refs": [longest],
            }
        remaining = envelope["body_json_bytes"] - nested_json_size(body)
        quote_bytes = nested_json_size('a"') - nested_json_size("a")
        for unit in first[:-1]:
            count = min(remaining // quote_bytes, 2000 - len(unit["text"]))
            unit["text"] = unit["text"][:-1] + '"' * count + "."
            remaining -= count * quote_bytes
        first[-1]["text"] = "Meaningful prose" + "x" * remaining + "."
        assert nested_json_size(body) == envelope["body_json_bytes"]
        check_writer_envelope(body, envelope, stage="body")
        validate_body(body, aliases, writing_plan=RANGE_WRITING_PLAN, envelope=envelope)
        system, payload, schema = summary_request_input(
            packet, body_units(body), envelope
        )
        system, payload = prepare_planned_input(
            system, payload, packet, aliases, RANGE_WRITING_PLAN, envelope
        )
        wire = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
        example = json.loads(
            wire[0]["content"]
            .split("JSON encoding example only: ", 1)[1]
            .splitlines()[0]
        )["conclusion"][0]
        assert example["body_unit_refs"] == ([f"u{maximum}"] if cited else ["u1"])
        assert (f"[C:{longest}]" in example["text"]) == cited
        native = provider_response_format(
            "summary", schema, wire_contract="flat-writer-v1"
        )
        output = plan["stages"]["review_revision/summary"]["output_tokens"]
        assert (
            strict_request_reservation(summary_retry_messages(wire), output, native)
            <= proof["rows"][-1]["reservation"]
            <= proof["cap"]
        )


@pytest.mark.parametrize("number", ["03", "04", "05", "06", "07", "08"])
def test_vit_inline_historical_ledger_is_read_only(number):
    from litagent.review_policy import identity
    from litagent.safety.planned_accounting import _envelope

    path = (
        Path("artifacts/acceptance/inline-citations-vit-235d5bf1dba2")
        / f"vit-235d5bf1dba2-inline-reassessment-{number}"
        / "budget-ledger.json"
    )
    if not path.is_file():
        pytest.skip("requires the immutable local ViT ledger")
    original = path.read_bytes()
    snapshot = json.loads(original)
    plan = snapshot["plan"]
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    with pytest.raises(ValueError, match="inline_citation_identity_invalid"):
        PlannedBudget(plan)
    mismatch = copy.deepcopy(plan["envelope"])
    mismatch["run_limits"]["workflow_identity"] = identity(inline=True)
    with pytest.raises(ValueError, match="inline_citation_identity_invalid"):
        _envelope(mismatch)
    assert path.read_bytes() == original


def test_complete_plan_keeps_all_future_rounds_and_independent_quotas(case):
    config = configuration()
    plan = budget_plan(case.packet, case.aliases, config)
    assert plan["status"] == "admitted"
    assert plan["planned_requests"] == 209 <= plan["maximum_requests"] == 226
    assert plan["planned_tokens"] <= plan["maximum_tokens"] == 47420000
    stages = plan["stages"]
    assert stages["capacity/inventory"]["count"] == 18
    assert (
        stages["capacity/repair"]["count"] == stages["capacity/summary"]["count"] == 3
    )
    assert stages["evaluation_revision/inventory"]["count"] == 18
    assert stages["inventory/inventory"]["count"] == 18
    assert stages["reviewer"]["count"] == 4
    validate_accounting_snapshot(PlannedBudget(plan).snapshot(), expected_plan=plan)
    from litagent.eval.native_capacity import validate_bound

    for fault in ("missing", "size", "contract"):
        altered = copy.deepcopy(plan["envelope"])
        if fault == "missing":
            altered["native_inventory_bound"].pop("report_context_json_bytes")
        elif fault == "size":
            altered["native_inventory_bound"]["report_context_json_bytes"] += 1
        else:
            altered["semantic_engine"].pop("inventory_context_contract")
        with pytest.raises(ValueError, match="native_capacity_bound_invalid"):
            validate_bound(altered)
    config.eval.native_capacity_recovery = False
    old = budget_plan(case.packet, case.aliases, config)
    assert "native_inventory_bound" not in old["envelope"]
    assert old["planned_requests"] == 191
    assert "native_capacity_recovery" not in config.eval.model_dump(mode="json")
    assert plan["identity"] != old["identity"]

    from litagent.safety.planned_budget import request_scope

    owner = PlannedBudget(plan)
    # Spend every independently reserved slot at its complete request ceiling.
    # No stage gets to reuse the same input/output reserve or semantic attempt.
    for stage, spec in stages.items():
        for index in range(spec["count"]):
            with request_scope(stage, f"maximum:{stage}:{index}"):
                owner.check_stage_requests(stage, [spec["request_cap"]])
                owner.reserve(spec["request_cap"])
                owner.bind_wire({"stage": stage, "index": index})
                owner.begin_request()
                usage = {
                    "prompt_tokens": spec["request_cap"] - spec["output_tokens"],
                    "completion_tokens": spec["output_tokens"],
                }
                owner.record(usage)
                before = owner.snapshot()
                owner.record(usage)
                assert owner.snapshot() == before
    snapshot = owner.snapshot()
    assert snapshot["requests_sent"] == plan["planned_requests"]
    assert snapshot["used_tokens"] == sum(s["reserved_tokens"] for s in stages.values())
    assert not snapshot["usage_uncertain"] and not snapshot["stop_reason"]
    validate_accounting_snapshot(snapshot, expected_plan=plan)


@pytest.mark.parametrize("text", ["x", '中文😀\\"\n', "\\\\\t\r\u0001"])
def test_inventory_bound_uses_real_requests_and_full_escaped_text(case, text):
    from litagent.eval.ragas_adapter import batch_request

    plan = budget_plan(case.packet, case.aliases, configuration())
    bound = plan["envelope"]["native_inventory_bound"]
    unit = text * ((bound["authored_text_json_bytes"] - 1000) // nested_json_size(text))
    samples = [
        {
            "id": f"u{10000+i}",
            "question": case.packet["query"],
            "response": unit if i == 0 else "",
            "report_context": {
                "id": f"p{10000+i}",
                "text": unit if i == 0 else "",
                "start": 0,
                "end": len(unit) if i == 0 else 0,
            },
        }
        for i in range(bound["batch_size"])
    ]
    assert check_inventory_inputs(samples, bound) <= bound["authored_text_json_bytes"]
    messages, schema, _ = batch_request(
        "inventory", [{"sample": s} for s in samples], inventory_context=True
    )
    measured = strict_request_reservation(
        messages, plan["stages"]["inventory"]["output_tokens"], schema
    )
    assert measured <= bound["request_cap"]
    damaged = copy.deepcopy(samples)
    damaged[0]["response"] = "x" * bound["authored_text_json_bytes"]
    with pytest.raises(ValueError, match="authored_text_exceeded"):
        check_inventory_inputs(damaged, bound)
    damaged = copy.deepcopy(samples)
    damaged[0]["question"] += "changed"
    with pytest.raises(ValueError, match="question_identity"):
        check_inventory_inputs(damaged, bound)
    damaged = copy.deepcopy(samples)
    damaged[0]["report_context"]["text"] += "x" * bound["report_context_json_bytes"]
    with pytest.raises(ValueError, match="report_context_exceeded"):
        check_inventory_inputs(damaged, bound)


def test_parent_stage_routing_does_not_borrow_semantic_repair_slots(case):
    from litagent import review_policy

    plan = budget_plan(case.packet, case.aliases, configuration())
    env = plan["envelope"]
    for parent in (None, "evaluation_revision", "inventory"):
        with review_policy.capacity_partition(parent):
            assert review_policy.stage_name(env, "inventory") == "capacity/inventory"
            assert review_policy.stage_name(env, "repair") == "capacity/repair"
            for phase in ("assessment", "global"):
                assert review_policy.stage_name(env, phase) == (
                    f"{parent}/{phase}" if parent else phase
                )
    assert review_policy.stage_name(env, "inventory") == "inventory"


async def replay(
    tmp_path,
    monkeypatch,
    *,
    mode="declined",
    editorial_number=None,
    config=None,
    capacity_response=None,
):
    """Replace only HTTP transport while running the retained ViT consumers."""
    old, draft, results, *_ = vit_replay()
    config = config or configuration()
    meta = draft["report"]["metadata"]
    doc = meta["writer_document"]
    plan = budget_plan(
        doc["packet"], doc["aliases"], config, meta["review_methodology"]
    )
    root = Path("artifacts/acceptance/inline-citations-vit-235d5bf1dba2")
    monkeypatch.syspath_prepend(str(root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "vit_capacity_replay", root / "offline-reassessment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = json.loads(
        Path(
            "artifacts/acceptance/workbench-vit-386-20261110/runs/235d5bf1dba2.json"
        ).read_bytes()
    )
    prior = archive["nodes"]["worker:adversarial_review"]["output"]
    source = copy.deepcopy(results["adversarial_review"])
    source.update(
        draft=draft["report"]["survey"],
        analysis_plan=prior["analysis_plan"],
        coverage_matrix=prior["coverage_matrix"],
    )
    source["writer_document"]["budget_envelope"] = plan["envelope"]
    from litagent.writing_v2 import RANGE_WRITING_PLAN, compile_document

    doc = source["writer_document"]
    rendered, source["writing_bindings"] = compile_document(
        doc["body"],
        doc["summaries"],
        source["fact_cards"],
        doc["aliases"],
        "zh-CN",
        source["review_methodology"],
        writing_plan=RANGE_WRITING_PLAN,
        envelope=plan["envelope"],
    )
    assert rendered == source["draft"]
    recorded = {
        r["id"]: r
        for r in old["evaluation"]["faithfulness"]["details"]["partial_samples"]
    }
    editorial = []
    initial_evaluation = []
    first_repair = []
    if editorial_number is not None:
        journal = (
            root
            / f"vit-235d5bf1dba2-inline-reassessment-{editorial_number}"
            / "repairs.json.writer-v2.json"
        )
        if not journal.is_file():
            pytest.skip("requires the immutable local ViT writer journal")
        calls = json.loads(journal.read_bytes())["calls"]
        names = ["reviewer:0", "repair_0", "summary_repair_1"]
        if editorial_number == "06":
            names.extend(("reviewer:1", "repair_1", "summary_repair_2"))
            archive = json.loads(
                next((journal.parent / "runs").glob("*.json")).read_bytes()
            )
            initial_evaluation = [
                node
                for node in archive["nodes"].values()
                if node["kind"] == "llm" and node["task_id"] == "evaluation.initial"
            ]
            first_repair = [
                calls[name]["provider_receipt"]
                for name in ("repair_2", "summary_repair_3")
            ]
        editorial = [calls[name]["provider_receipt"] for name in names]

    class Gateway(module.MockGateway):
        capacity_calls = 0
        first_repair_body = None

        async def __call__(self, request):
            wire = json.loads(request.content)
            data = json.loads(wire["input"][-1]["content"])
            schema = wire["text"]["format"]["schema"]
            if editorial or (not initial_evaluation and first_repair):
                from litagent.llm.responses import decode_private_receipt

                queue = editorial if editorial else first_repair
                receipt = queue.pop(0)
                original, _ = decode_private_receipt(receipt)
                assert wire["input"][-1] == original["input"][-1]
                if queue is first_repair and "target_responses" in schema["properties"]:
                    self.first_repair_body = data["body_sha256"]
                self.calls.append("editorial" if queue is editorial else "first_repair")
                return httpx.Response(
                    receipt["status_code"],
                    content=base64.b64decode(receipt["response_body_base64"]),
                    headers=receipt["headers"],
                )
            if initial_evaluation:
                from test_native_responses import envelope

                saved = initial_evaluation.pop(0)
                expected_message = saved["input"]["messages"][-1]
                if "inventory_context_contract" in data:
                    from litagent.eval.ragas_adapter import (
                        INVENTORY_CONTEXT_CONTRACT,
                        report_context,
                    )

                    # Replay historical outputs, but authenticate the added context
                    # and retain an exact comparison of all original input fields.
                    projected = copy.deepcopy(data)
                    assert projected.pop("inventory_context_contract") == (
                        INVENTORY_CONTEXT_CONTRACT
                    )
                    backgrounds = projected.pop("report_contexts")
                    used = set()
                    for job in projected["jobs"].values():
                        key = job.pop("report_context_ref")
                        start = job.pop("answer_start")
                        end = job.pop("answer_end")
                        report_context(
                            {
                                "response": job["answer"],
                                "report_context": {
                                    "id": key,
                                    "text": backgrounds[key],
                                    "start": start,
                                    "end": end,
                                },
                            }
                        )
                        used.add(key)
                    assert used == set(backgrounds)
                    assert wire["input"][-1]["role"] == expected_message["role"]
                    assert projected == json.loads(expected_message["content"])
                else:
                    assert wire["input"][-1] == expected_message
                usage = saved["usage"]
                response = envelope(saved["output"], model=wire["model"])
                response["usage"].update(
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    output_tokens_details={"reasoning_tokens": 0},
                )
                self.calls.append("initial_evaluation")
                return httpx.Response(200, json=response)
            if editorial_number == "06" and "target_responses" in schema["properties"]:
                assert data["previous_local_failure"]["code"] == (
                    "writer_inline_citation_without_prose"
                )
                assert data["body_sha256"] == self.first_repair_body
            if "deleted_sources" in schema["properties"]:
                self.capacity_calls += 1
                (tmp_path / "capacity-request.json").write_text(
                    json.dumps(wire, ensure_ascii=False, indent=2)
                )
                answer = {
                    "outcome": "declined",
                    "explanation": "Frozen material cannot be safely shortened in this transport fixture.",
                    "patches": [],
                    "deleted_sources": [],
                }
                if mode in ("candidate", "persistent"):
                    target = data["targets"]["r1"]
                    original = data["current_body"]["introduction"][0]
                    texts = [unit[0] for unit in original]
                    # Remove an organizational restatement, keeping the full
                    # research question and every cited scientific premise.
                    texts[0] = texts[0].split("这里不把")[0]
                    if mode == "persistent":
                        texts[0] += f"本节讨论范围见引言（第{self.capacity_calls}稿）。"
                    answer = {
                        "outcome": "patched",
                        "explanation": "Remove a redundant scope restatement; preserve all cited content.",
                        "patches": [
                            {"text": text, "source_unit_ids": [uid]}
                            for text, uid in zip(
                                texts, target["original_unit_ids"], strict=True
                            )
                        ],
                        "deleted_sources": [],
                    }
                from test_native_responses import envelope

                reply = envelope(json.dumps(answer), model=wire["model"])
                reply["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
                if capacity_response is not None:
                    capacity_response(self, data, reply)
                self.calls.append(
                    "capacity_declined"
                    if answer["outcome"] == "declined"
                    else "capacity_patch"
                )
                return httpx.Response(200, json=reply)
            if "abstract" in schema["properties"]:
                from test_native_responses import envelope

                answer = {
                    section: [
                        {
                            "text": unit["text"],
                            "body_unit_refs": unit["body_unit_refs"],
                            "paragraph_start": index == 0,
                        }
                        for paragraph in doc["summaries"][section]
                        for index, unit in enumerate(paragraph["sentences"])
                    ]
                    for section in ("abstract", "conclusion")
                }
                self.calls.append("summary")
                reply = envelope(
                    json.dumps(answer, ensure_ascii=False), model=wire["model"]
                )
                reply["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
                return httpx.Response(200, json=reply)
            response = await super().__call__(request)
            if "jobs" in data and "context_pool" not in data:
                if self.capacity_calls and mode == "candidate":
                    return response
                answer = {
                    key: {"statements": copy.deepcopy(recorded[key]["statements"])}
                    for key in data["jobs"]
                }
                for key in ("u1", "u2", "u3"):
                    if key in answer and editorial_number is None:
                        answer[key]["statements"][0] = "x" * 2394
                reply = response.json()
                reply["output"][0]["content"][0]["text"] = json.dumps(
                    answer, ensure_ascii=False
                )
                if mode == "unknown_usage":
                    reply.pop("usage")
                return httpx.Response(200, json=reply)
            return response

    gateway = Gateway()
    monkeypatch.setenv("LLM_API_KEY", "offline-only")
    monkeypatch.setattr(module, "RUN_ID", "vit-native-capacity")
    monkeypatch.setattr(
        module,
        "load_inputs",
        lambda: (draft["query"], source, results["extract"], config, plan),
    )
    monkeypatch.setattr(module, "MockGateway", lambda: gateway)
    await module.execute(tmp_path, mock=True)
    report = json.loads(
        (tmp_path / "runs/vit-native-capacity-offline.json").read_bytes()
    )["report"]
    return report, gateway, plan


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "number, reason, stage, charge",
    [
        ("04", "writer_inline_citation_position_invalid", "summary", 11968),
        ("05", "writer_inline_citation_unknown", "summary_compilation", 11998),
        ("07", "writer_inline_citation_unknown", "summary_compilation", 12176),
    ],
)
async def test_vit_settled_summary_rejection_completes_retained_evaluation(
    tmp_path, monkeypatch, number, reason, stage, charge
):
    """Replay all three real editorial HTTP bodies before evaluating the original."""
    r = Path("artifacts/acceptance/inline-citations-vit-235d5bf1dba2") / (
        f"vit-235d5bf1dba2-inline-reassessment-{number}"
    )
    journal = r / "repairs.json.writer-v2.json"
    if not journal.is_file():
        pytest.skip("requires the immutable local ViT writer journal")
    archived = {
        p: p.read_bytes()
        for p in (journal, r / "budget-ledger.json", r / "result.json")
    }
    report, gateway, plan = await replay(tmp_path, monkeypatch, editorial_number=number)
    assert gateway.calls[:3] == ["editorial"] * 3
    assert all(
        phase in gateway.calls[3:] for phase in ("inventory", "assessment", "global")
    )
    meta = report["metadata"]
    detail = report["evaluation"]["faithfulness"]["details"]
    assert detail["stages"]["global_check"] == "completed"
    assert detail["coverage"]["complete"]
    assert detail["unverified_count"] == 0
    assert not report["delivery"]["publishable"]
    original = json.loads(
        (r.parent / "runs/vit-235d5bf1dba2-inline-draft.json").read_bytes()
    )["report"]
    assert original["survey"] == report["survey"]
    baseline = original["metadata"]["writing_bindings"]
    current = meta["writing_bindings"]
    assert current["report_sha256"] == baseline["report_sha256"]
    assert current["citation_scopes"] == baseline["citation_scopes"]
    assert current["units"] == baseline["units"]
    assert detail["source_report_sha256"] == baseline["report_sha256"]
    [edit] = meta["editorial_history"]
    assert edit["status"] == "candidate_rejected"
    assert edit["reason_code"] == reason
    assert edit["summary_failure"]["stage"] == stage
    assert edit["summary_failure"]["call"] == "summary_repair_1"
    if number == "05":
        [diagnostic] = edit["summary_failure"]["diagnostics"]
        assert diagnostic["path"] == ["conclusion", 0, 0]
        assert diagnostic["reference"] == "c9"
    if number == "07":
        decision = edit["summary_rejection"]
        assert decision["action"] == "evaluate_retained"
        assert decision["retained_report_sha256"] == baseline["report_sha256"]
        errors = {d["reference"]: d for d in decision["diagnostics"]}
        assert set(errors) == {"c29", "c48"}
        assert errors["c29"]["in_frozen_pool"] is True
        assert errors["c48"]["in_frozen_pool"] is False
        assert errors["c29"]["path"] == errors["c48"]["path"] == ["conclusion", 1, 0]
    assert edit["source_report_sha256"] == baseline["report_sha256"]
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    summary = [
        e
        for e in snapshot["entries"].values()
        if e["stage"] == "review_revision/summary"
    ]
    assert len(summary) == 1 and summary[0]["actual_tokens"] == charge
    review_attempts = [
        a
        for a in meta["targeted_repair_state"]["attempts"]
        if a["category"] == "review_revision"
    ]
    assert len(review_attempts) == 1 and review_attempts[0]["status"] == "completed"
    saved = json.loads((tmp_path / "repairs.json.writer-v2.json").read_bytes())
    assert (
        saved["calls"]["summary_repair_1"]["raw"]
        == json.loads(archived[journal])["calls"]["summary_repair_1"]["raw"]
    )
    assert not saved.get("editorial_candidates")
    assert all(p.read_bytes() == raw for p, raw in archived.items())


@pytest.mark.asyncio
async def test_vit06_settled_summary_rejection_uses_next_evaluation_revision(
    tmp_path, monkeypatch
):
    report, gateway, plan = await replay(tmp_path, monkeypatch, editorial_number="06")
    assert gateway.calls == (
        ["editorial"] * 6
        + ["initial_evaluation"] * 12
        + ["first_repair"] * 2
        + ["repair_declined"]
    )
    metadata = report["metadata"]
    initial = metadata["evaluation_history"]["initial_evaluation"]["faithfulness"][
        "details"
    ]
    assert (
        initial["source_report_sha256"] == metadata["writing_bindings"]["report_sha256"]
    )
    assert (
        initial["coverage"]["complete"]
        and initial["stages"]["global_check"] == "completed"
    )
    assert [
        initial[key]
        for key in ("supported_count", "unsupported_count", "unverified_count")
    ] == [376, 85, 0]
    first, second = metadata["evaluation_history"]["attempts"]
    assert first["reason_code"] == "writer_inline_citation_without_prose"
    assert {
        k: first["local_diagnostics"][0][k] for k in ("code", "path", "offset")
    } == {
        "code": "writer_inline_citation_without_prose",
        "path": ["abstract", 0, 0],
        "offset": 237,
    }
    assert len(first["local_diagnostics"]) > 1
    assert first["candidate"] is None and first["candidate_evaluation"] is None
    assert second["attempted"] and not second["committed"]
    assert not report["delivery"]["publishable"]
    assert metadata["review_workflow"]["allowances"]["evaluation_revision"]["used"] == 2
    saved = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(saved, expected_plan=plan)
    summary_charges = [
        e["actual_tokens"]
        for e in saved["entries"].values()
        if e["stage"] == "evaluation_revision/summary"
    ]
    assert summary_charges == [12283]


@pytest.mark.asyncio
async def test_complete_overflow_enters_capacity_controller_and_decline_stays_draft(
    tmp_path, monkeypatch
):
    report, gateway, plan = await replay(tmp_path, monkeypatch)
    details = report["evaluation"]["faithfulness"]["details"]
    assert (
        details["execution_error"] == "evaluation_native_capacity_refinement_required"
    )
    assert len(details["assertions"]) == 505
    assert report["evaluation"]["faithfulness"]["score"] is None
    assert gateway.calls.count("capacity_declined") == 1
    assert "assessment" not in gateway.calls and "global" not in gateway.calls
    assert not report["delivery"]["publishable"]
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert [a["category"] for a in attempts] == ["capacity"]
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert not snapshot["stop_reason"] and not snapshot["usage_uncertain"]
    assert all(
        e["stage"] in {"reviewer", "inventory", "capacity/repair"}
        for e in snapshot["entries"].values()
    )
    from litagent.eval.ragas_repair import recovery_decision
    from litagent.runner import LitAgent
    from litagent.safety.planned_budget import request_scope

    _, draft, results, *_ = vit_replay()
    results["adversarial_review"].update(
        report["metadata"], final_draft=report["survey"]
    )
    context = LitAgent(configuration())._build_eval_context(
        query=draft["query"], survey=report["survey"], results=results
    )
    owner = PlannedBudget(plan)
    owner.restore(snapshot)
    assert recovery_decision(report, context, owner)["action"] == "capacity"
    for fault in ("sha", "count", "statements", "usage", "rows", "exhausted"):
        changed = copy.deepcopy(report)
        d = changed["evaluation"]["faithfulness"]["details"]
        if fault == "sha":
            changed["survey"] += "changed"
        elif fault == "count":
            d["stages"]["capacity"]["actual_assertions"] -= 1
        elif fault == "statements":
            d["partial_samples"][0]["statements"].clear()
        elif fault == "usage":
            d["stages"]["requests"][0]["usage"]["usage_available"] = False
        elif fault == "rows":
            d["stages"]["requests"].append(copy.deepcopy(d["stages"]["requests"][0]))
        else:
            state = changed["metadata"]["targeted_repair_state"]
            for index in (1, 2):
                state["attempts"].append(
                    {**copy.deepcopy(state["attempts"][0]), "index": index}
                )
        decision = recovery_decision(changed, context, owner)
        assert decision["action"] == "stop", (fault, decision)
    with request_scope("capacity/repair", "unknown-result"):
        ceiling = plan["stages"]["capacity/repair"]["request_cap"]
        owner.reserve(ceiling)
        owner.begin_request()
        owner.charge_unknown_usage(ceiling)
    blocked = owner.snapshot()
    assert recovery_decision(report, context, owner)["action"] == "stop"
    restored = PlannedBudget(plan)
    restored.restore(blocked)
    assert restored.snapshot() == blocked
    assert recovery_decision(report, context, restored)["action"] == "stop"
    assert restored.snapshot() == blocked


@pytest.mark.asyncio
async def test_capacity_candidate_gets_full_assessment_and_scientific_failures_stay_local(
    tmp_path, monkeypatch
):
    report, gateway, plan = await replay(tmp_path, monkeypatch, mode="candidate")
    assert gateway.calls.count("capacity_patch") == 1
    assert gateway.calls.count("summary") == 1
    assert "assessment" in gateway.calls and "global" in gateway.calls
    journal = json.loads((tmp_path / "repairs.json.writer-v2.json").read_bytes())
    candidates = journal["candidates"]
    assert len(candidates) == 1 and not candidates[0]["committed"]
    details = candidates[0]["evaluation"]["faithfulness"]["details"]
    assert details["stages"]["global_check"] == "completed"
    assert details["unverified_count"] == 0 and details["unsupported_count"] > 0
    assert not report["delivery"]["publishable"]
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    stages = [e["stage"] for e in snapshot["entries"].values()]
    assert "capacity/inventory" in stages
    assert "assessment" in stages and "global" in stages
    assert not any(s.startswith("evaluation_revision/") for s in stages)
    assert report["metadata"]["evidence_rewrite"]["reason_code"] == (
        "native_capacity_candidate_quality_failed"
    )
    from litagent.eval.automatic import _funded_automatic_capacity

    owner = PlannedBudget(plan)
    owner.restore(snapshot)
    candidate = candidates[0]
    current = {
        "survey": candidate["survey"],
        "evaluation": candidate["evaluation"],
        "metadata": {
            **report["metadata"],
            "writing_bindings": candidate["bindings"],
            "writer_document": candidate["writer_document"],
        },
    }
    # Funding proves settled provenance, not scientific success.
    funded = _funded_automatic_capacity(
        current, owner, run_id="vit-native-capacity-offline"
    )
    assert funded["source_report_sha256"] == candidate["report_sha256"]
    changed = copy.deepcopy(current)
    changed["metadata"]["targeted_repair_state"]["attempts"][0]["native_capacity"][
        "budget_partition"
    ] = "evaluation_revision"
    with pytest.raises(ValueError, match="automatic_capacity_binding_invalid"):
        _funded_automatic_capacity(changed, owner, run_id="vit-native-capacity-offline")


@pytest.mark.asyncio
async def test_persistent_overflow_uses_at_most_three_capacity_attempts(
    tmp_path, monkeypatch
):
    report, gateway, plan = await replay(tmp_path, monkeypatch, mode="persistent")
    assert gateway.calls.count("capacity_patch") == 3
    assert gateway.calls.count("summary") == 3
    assert "assessment" not in gateway.calls and "global" not in gateway.calls
    state = report["metadata"]["targeted_repair_state"]
    assert len(state["attempts"]) == 3
    assert all(a["category"] == "capacity" for a in state["attempts"])
    assert not report["delivery"]["publishable"]
    assert (
        report["metadata"]["recovery_decision"]["reason_code"]
        == "native_capacity_attempts_exhausted"
    )
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert (
        sum(e["stage"] == "capacity/inventory" for e in snapshot["entries"].values())
        == 18
    )
    journal = json.loads((tmp_path / "repairs.json.writer-v2.json").read_bytes())
    feedback = journal["deferred_prior_feedback"]
    assert len(feedback) == 2
    assert all(row["policy"] == "native-capacity-recovery-v1" for row in feedback)
    assert all(
        row["capacity"]["actual"] > row["capacity"]["maximum"] for row in feedback
    )
    from litagent.eval.native_capacity import retry_feedback
    from litagent.writing_v2 import repair_feedback_directory

    original = report["metadata"]
    prior = journal["candidates"][0]
    failure = feedback[0]["previous_failure"]
    from litagent.capacity_repair import capacity_targets

    targets = capacity_targets(
        original["writer_document"]["body"],
        report["evaluation"]["faithfulness"]["details"]["stages"]["capacity"],
        maximum=plan["envelope"]["body_max_units"],
    )
    directory = repair_feedback_directory(
        original["writer_document"]["packet"],
        report["survey"],
        original["writer_document"],
        original["writing_bindings"],
        prior,
        failure,
        targets,
    )
    assert (
        retry_feedback(failure, prior, directory)["capacity"] == feedback[0]["capacity"]
    )
    changed = copy.deepcopy(prior)
    changed["survey"] += "changed"
    with pytest.raises(ValueError, match="native_capacity_retry_identity_invalid"):
        retry_feedback(failure, changed, directory)


@pytest.mark.asyncio
async def test_unknown_usage_stops_before_capacity_or_support_requests(
    tmp_path, monkeypatch
):
    report, gateway, _ = await replay(tmp_path, monkeypatch, mode="unknown_usage")
    assert gateway.capacity_calls == 0
    assert "assessment" not in gateway.calls and "global" not in gateway.calls
    assert not report["delivery"]["publishable"]
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    assert snapshot["usage_uncertain"]
