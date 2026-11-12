"""Repair located native failures without discarding sources or protected prose."""

import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from litagent.quality import (
    GroundedWritingError,
    _body_reference_path,
    _native_unit,
    apply_native_repairs,
    grounded_messages,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
)
from litagent.safety.budget import strict_request_reservation


@pytest.fixture
def received():
    """Use complete public sources, received Pro prose and ten original failures."""
    return json.loads(
        Path("tests/fixtures/native-complete-failure-real.json").read_text()
    )


def render(fixture, candidate):
    """Compile with the exact live permissions, including sentence citation scope."""
    return render_grounded_report_traced(
        candidate,
        fixture["cards"],
        "academic_v2",
        "zh-CN",
        evidence_selection=fixture["selection"],
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=True,
    )


def request(fixture):
    """Exercise the production first-writing request and bounded repair producer."""
    with pytest.raises(GroundedWritingError) as failed:
        render(fixture, fixture["candidate"])
    assert failed.value.diagnostics == fixture["diagnostics"]
    messages, _ = prepare_grounded_request(
        grounded_messages(
            fixture["cards"],
            "academic_v2",
            "zh-CN",
            fixture["query"],
            coverage=fixture["coverage_requirements"],
            evidence_selection=fixture["selection"],
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        )
    )
    return prepare_native_repair_request(
        messages,
        fixture["candidate"],
        failed.value.diagnostics,
        cards=fixture["cards"],
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=fixture["selection"],
    )


def protocol_patches(fixture, binding):
    """Create format fixtures, without claiming the complete report is scientific gold."""
    document = json.loads(fixture["candidate"])
    patches = {}
    summary_keys = []
    for key, target in binding["targets"].items():
        if target.get("unit_type") == "summary":
            summary_keys.append(key)
            continue
        original = target.get("original") or _native_unit(document, target["path"])
        allowed = target["allowed_fact_card_ids"]
        ids = []
        for cid in original.get("fact_card_ids", []):
            source = fixture["cards"][cid]
            matches = [
                aid
                for aid in allowed
                if all(
                    fixture["cards"][aid][field] == source[field]
                    for field in ("paper_id", "source_quote")
                )
            ]
            if not matches:
                break
            ids.append(matches[0])
        else:
            if ids:
                patch = {**original, "fact_card_ids": list(dict.fromkeys(ids))}
                patches[key] = patch
                _native_unit(document, target["path"]).update(patch)
                continue
        notices = [n for n in target["allowed_scope_notice_ids"] if n != "none"]
        if (
            original["kind"] != "scientific"
            and notices
            and not target.get("must_remain_scientific")
        ):
            patch = {
                "text": "",
                "kind": "organization",
                "fact_card_ids": [],
                "scope_notice_id": notices[0],
            }
        else:
            cid = allowed[0]
            quote = fixture["cards"][cid]["source_quote"]
            text = re.split(r"(?<=[.!?])\s+", quote)[0]
            assert (
                text
                == "The goal of few-shot learning is to recognize new visual concepts with just a few amount of labeled samples in each class."
            )
            text = "少样本学习的目标是利用每个类别少量标注样本来识别新的视觉概念。"
            patch = {
                "text": text,
                "kind": "scientific",
                "fact_card_ids": [cid],
                "scope_notice_id": "none",
            }
        patches[key] = patch
        _native_unit(document, target["path"]).clear()
        _native_unit(document, target["path"]).update(patch)
    for index, key in enumerate(summary_keys):
        target = binding["targets"][key]
        uid = target["allowed_body_unit_ids"][index]
        body = _native_unit(
            document, _body_reference_path(binding["body_unit_catalog"][uid])
        )
        patches[key] = {"text": body["text"], "body_unit_ids": [uid]}
    return {"can_repair": True, "patches": patches}


def test_received_ten_failures_have_nine_exact_targets_and_passive_dependency(received):
    messages, schema, binding = request(received)
    assert binding["version"] == "complete-native-unit-repair-v5"
    assert len(binding["targets"]) == 9
    summaries = [
        t for t in binding["targets"].values() if t.get("unit_type") == "summary"
    ]
    assert len(summaries) == 2
    assert all(t["section_key"] == "conclusion" for t in summaries)
    packet = json.loads(messages[1]["content"])
    original = json.loads(received["messages"][1]["content"])
    assert packet["sources"] == original["sources"]
    assert [r["finding_candidate"] for r in packet["analysis_plan"]["sources"]] == [
        r["finding_candidate"] for r in original["analysis_plan"]["sources"]
    ]
    original_rows = {
        row["fact_card_id"]: row for row in original["analysis_plan"]["sources"]
    }
    for row in packet["analysis_plan"]["sources"]:
        cid = row["fact_card_id"]
        assert (
            packet["cards"][cid]["unverified_fields"]
            == original_rows[cid]["candidate_fields"]
        )
        assert (
            packet["sources"][row["source_ref"]]["paper_id"]
            == original_rows[cid]["paper_id"]
        )
    assert strict_request_reservation(messages, 20000, schema) <= 117964
    assert "or edit summary selectors." not in messages[0]["content"]
    catalog = binding["body_unit_catalog"]
    for ref in catalog.values():
        body = _native_unit(
            json.loads(received["candidate"]), _body_reference_path(ref)
        )
        assert body["kind"] == "scientific"
    assert not any("abstract" in t["path"] for t in binding["targets"].values())


def test_complete_patches_compile_and_preserve_every_non_target_native_node(received):
    _, schema, binding = request(received)
    response = protocol_patches(received, binding)
    Draft202012Validator(schema["json_schema"]["schema"]).validate(response)
    candidate, proof = apply_native_repairs(
        received["candidate"], json.dumps(response), binding, cards=received["cards"]
    )
    draft = render(received, candidate)
    assert "[E:" in draft
    original, updated = json.loads(received["candidate"]), json.loads(candidate)
    for target in binding["targets"].values():
        for document in (original, updated):
            unit = _native_unit(document, target["path"])
            unit.clear()
            unit["authorized"] = True
    assert original == updated
    assert len(proof["modified_paths"]) == 9


@pytest.mark.parametrize(
    "damage",
    [
        "unknown",
        "duplicate",
        "empty",
        "boolean",
        "direct_ref",
        "organization",
        "tampered_sha",
        "changed_source",
        "extra_patch",
    ],
)
def test_summary_repair_cannot_add_sources_or_reach_unapproved_units(received, damage):
    _, _, binding = request(received)
    response = protocol_patches(received, binding)
    key = next(
        k for k, t in binding["targets"].items() if t.get("unit_type") == "summary"
    )
    patch = response["patches"][key]
    if damage == "unknown":
        patch["body_unit_ids"] = ["missing"]
    elif damage == "duplicate":
        patch["body_unit_ids"] *= 2
    elif damage == "empty":
        patch["text"] = ""
    elif damage == "boolean":
        patch["body_unit_ids"] = [True]
    elif damage == "direct_ref":
        patch["body_references"] = [
            binding["body_unit_catalog"][patch["body_unit_ids"][0]]
        ]
    elif damage == "organization":
        patch.update(
            kind="organization", fact_card_ids=[], scope_notice_id="review_scope"
        )
    elif damage == "tampered_sha":
        received["candidate"] += " "
    elif damage == "changed_source":
        next(iter(received["cards"].values()))["source_quote"] += " changed"
    elif damage == "extra_patch":
        response["patches"]["unapproved"] = patch
    with pytest.raises(ValueError):
        apply_native_repairs(
            received["candidate"],
            json.dumps(response),
            binding,
            cards=received["cards"],
        )


def test_forged_or_omitted_native_diagnostics_cannot_authorize_repairs(received):
    messages, _, _ = request(received)
    for diagnostics in (
        received["diagnostics"][:-1],
        [{**received["diagnostics"][0], "candidate_text": "forged"}],
    ):
        with pytest.raises(ValueError, match="diagnostic_invalid"):
            prepare_native_repair_request(
                messages,
                received["candidate"],
                diagnostics,
                cards=received["cards"],
                structure="academic_v2",
                language="zh-CN",
                evidence_selection=received["selection"],
            )


def test_complete_repair_budget_still_blocks_oversized_diagnostic_feedback(received):
    """Keep full sources and maximum output; excessive feedback cannot escape context."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litagent.quality import check_grounded_request_budget

    messages, schema, _ = request(received)
    packet = json.loads(messages[1]["content"])
    packet["previous_local_failure"] = {"explanation": "Known failure. " * 6000}
    messages[1]["content"] = json.dumps(
        packet, ensure_ascii=False, separators=(",", ":")
    )
    llm = SimpleNamespace(chat=AsyncMock())
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_grounded_request_budget(llm, messages, schema)
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_actual_sdk_synthesis_repairs_real_mixed_failure_then_enters_review(
    received, tmp_path
):
    """Mock only HTTP, retaining source selection, compiler, ledger and Reviewer."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx
    from openai import AsyncOpenAI

    from litagent.agents.adversarial import AdversarialReviewWorker
    from litagent.agents.reviewer import ReviewerWorker
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.llm.client import OpenAICompatibleClient
    from litagent.llm.routing import ModelBoundClient
    from litagent.orchestrator.task_graph import SubTask
    from litagent.safety.budget import CostBudget

    wire = []

    def transport(req):
        payload = json.loads(req.content)
        wire.append(payload)
        i = len(wire) - 1
        if i == 0:
            packet = json.loads(payload["messages"][1]["content"])
            assert packet["summary_array_wire_contract"] == "summary-array-items-v1"
            sections = payload["response_format"]["json_schema"]["schema"][
                "properties"
            ]["sections"]["properties"]
            for section_key in ("abstract", "conclusion"):
                assert set(sections[section_key]["properties"]["blocks"]) == {
                    "type",
                    "items",
                }
            from litagent.native_wire import array_from_native

            content = json.dumps(
                array_from_native(json.loads(received["candidate"])), ensure_ascii=False
            )
        elif i == 1:
            packet = json.loads(payload["messages"][1]["content"])
            assert packet["repair_contract"] == "complete-native-unit-repair-v5"
            content = json.dumps(
                protocol_patches(
                    received,
                    {
                        "targets": packet["repair_targets"],
                        "body_unit_catalog": packet["body_unit_catalog"],
                    },
                ),
                ensure_ascii=False,
            )
        else:
            assert i == 2
            content = json.dumps(
                {
                    "score": 0.9,
                    "verdict": "accept",
                    "evidence_compliance": {"passed": True},
                }
            )
        return httpx.Response(
            200,
            json={
                "id": f"offline-{i}",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                },
            },
        )

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1", "lite", api_key="offline", cost_budget=budget
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    pro, lite = ModelBoundClient(owner, "pro", "generation"), ModelBoundClient(
        owner, "lite", "evaluation"
    )
    path = tmp_path / "repairs.json"
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(
                return_value=EvidenceSelection.from_dict(received["selection"])
            )
        ),
        run_id_provider=lambda: "native-real",
        repair_state_path_provider=lambda: path,
        thinking_mode="disabled",
    )
    worker = AdversarialReviewWorker(pro, synthesis, ReviewerWorker(lite), max_rounds=1)
    task = SubTask(
        "adversarial_review",
        "write",
        "adversarial_review",
        input_data={
            "query": received["query"],
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "report_language": "zh-CN",
            "review_structure": "academic_v2",
            "verification": {"judge": False, "jev": False},
        },
    )
    try:
        result = await worker.execute(task)
        assert [p["model"] for p in wire] == ["pro", "pro", "lite"]
        assert result["writing_bindings"]["report_sha256"]
        state = json.loads(path.read_text())["state"]
        assert (
            len(state["attempts"]) == 1
            and state["attempts"][0]["status"] == "completed"
        )
        assert budget.used == 3300
    finally:
        await owner.close()
