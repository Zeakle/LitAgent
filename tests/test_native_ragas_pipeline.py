"""Run native RAGAS through the real SDK, workers, rewrites and automatic delivery."""

import copy
import json

import httpx
import pytest
import test_structured_repair_pipeline as structured
import test_writer_source_view_pipeline as writer_view
from test_native_ragas_evaluation import SemanticTransport, verdict_pairs
from test_native_responses import envelope
from test_writer_source_view_pipeline import run_scientific_case
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import load_config
from litagent.eval.engine_policy import (
    ANCHORED_GLOBAL_REVIEW,
    CONTRACT,
    KEYED_GLOBAL_ANCHORS,
    KEYED_GLOBAL_LOCATED_OMISSIONS,
    KEYED_GLOBAL_QUOTED_OMISSIONS,
    configured_engine,
    engine_descriptor,
)
from litagent.eval.ragas_global_wire import UNIT_CONTRACT, unpack_global_input
from litagent.output_capacity import configured_output_tokens
from litagent.run_limits import configured_run_limits
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_flat import _canonical_task_payload
from litagent.writing_v2 import BODY, CLASSIFICATION_GUIDANCE, EVIDENCE_SCOPE_GUIDANCE


class NativeGateway(structured.LongScientificGateway):
    """Replace provider HTTP only, including official RAGAS typed output payloads."""

    def __init__(self, case, failures, **kwargs):
        """Keep the prior writer/reviewer fixtures and the real native evaluator."""
        super().__init__(case, failures, **kwargs)
        self.semantic = SemanticTransport()
        self.global_calls = 0
        self.capacity_calls = 0

    def answer(self, payload, native_schema):
        mode = getattr(self, "recovery_case", None)
        if mode in {"combined_last", "extended"}:
            canonical = _canonical_task_payload(payload)
            if canonical.get("cause") == "capacity":
                self.capacity_calls += 1
                patches = []
                for target in canonical["targets"].values():
                    section, pi, start = target["path"]
                    units = canonical["current_body"][section][pi]["sentences"]
                    for i, uid in enumerate(target["original_unit_ids"]):
                        unit = units[start + i]
                        text = unit["text"].rstrip()
                        if (
                            mode == "extended"
                            and self.capacity_calls < 3
                            and section == BODY[-1]
                        ):
                            text += " " * (1200 - self.capacity_calls)
                        patches.append(
                            {
                                **unit,
                                "text": text,
                                "source_unit_ids": [uid],
                            }
                        )
                return {
                    "outcome": "patched",
                    "explanation": "Preserve all claims and remove trailing whitespace.",
                    "patches": patches,
                    "deleted_sources": [],
                }
        value = super().answer(payload, native_schema)
        if mode in {"combined_last", "extended"}:
            if payload.get("requested_body_section") == BODY[-1]:
                value["units"][0]["text"] += " " * 1200
        if getattr(self, "recovery_case", None) in {"inventory", "persistent"}:
            if "issues" in native_schema["properties"]:
                return {"issues": [], "explanation": "No separate reviewer concern."}
        return value

    async def __call__(self, request):
        """Return actual provider envelopes after validating native request schemas."""
        body = json.loads(request.content)
        messages = body.get("input", body.get("messages"))
        payload = json.loads(messages[1]["content"])
        mode = getattr(self, "recovery_case", None)
        if "jobs" not in payload and not (
            "source_units" in payload
            and payload.get("body_analysis_contract") == "body-analysis-v1"
            and "unit_inventory" not in payload
        ):
            return await super().__call__(request)
        self.sent.append(
            {
                "request": body,
                "request_bytes": request.content,
                "payload": copy.deepcopy(payload),
            }
        )
        response = await self.semantic.chat(messages)
        raw = json.loads(response.content)
        if "jobs" in payload and "context_pool" not in payload:
            for value in raw.values():
                value["statements"] *= getattr(self, "statements_per_unit", 1)
        if "context_pool" in payload:
            for key, value in raw.items():
                for statement, verdict in verdict_pairs(payload["jobs"][key], value):
                    if (
                        "FLOPs by" in statement
                        and mode not in {"inventory", "persistent"}
                        and (mode != "extended" or self.global_calls >= 3)
                    ):
                        verdict.update(
                            verdict=0,
                            reason="No cited source reports this FLOPs result.",
                        )
        if "source_units" in payload:
            self.global_calls += 1
            if mode and (
                self.global_calls == 1
                or mode == "persistent"
                or (mode == "extended" and self.global_calls <= 3)
            ):
                canonical = (
                    unpack_global_input(payload) if "owners" in payload else payload
                )
                unit = next(
                    unit
                    for unit in canonical["source_units"]
                    if "FLOPs by" in unit["text"]
                )
                assert unit["extracted_claim_ids"]
                location = (
                    {
                        "anchor_ids": [
                            (
                                unit["coverage_key"]
                                if canonical.get("global_unit_response_contract")
                                in (
                                    KEYED_GLOBAL_ANCHORS,
                                    KEYED_GLOBAL_QUOTED_OMISSIONS,
                                    KEYED_GLOBAL_LOCATED_OMISSIONS,
                                )
                                else unit["source_unit_id"]
                            )
                        ]
                    }
                    if canonical.get("global_findings_contract")
                    == ANCHORED_GLOBAL_REVIEW
                    else {"source_unit_id": unit["source_unit_id"], "claim_ids": []}
                )
                raw["issues"] = [
                    {
                        **location,
                        "code": "missing_assertion",
                        "explanation": (
                            unit["text"].split("[E:", 1)[0].strip()
                            if canonical.get("global_unit_response_contract")
                            == KEYED_GLOBAL_QUOTED_OMISSIONS
                            else "The FLOPs result is missing from the inventory."
                        ),
                    }
                ]
                if mode == "persistent" or (
                    canonical.get("global_unit_response_contract")
                    in (
                        KEYED_GLOBAL_ANCHORS,
                        KEYED_GLOBAL_QUOTED_OMISSIONS,
                        KEYED_GLOBAL_LOCATED_OMISSIONS,
                    )
                    and mode == "combined"
                ):
                    if "coverage_key" in unit:
                        raw["units"][unit["coverage_key"]]["complete"] = False
                    else:
                        for item in raw["units"]:
                            if item["source_unit_id"] == unit["source_unit_id"]:
                                item["complete"] = False
        from jsonschema import Draft202012Validator

        Draft202012Validator(body["text"]["format"]["schema"]).validate(raw)
        result = envelope(json.dumps(raw), model=body["model"])
        result["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
        return httpx.Response(200, json=result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failures,workbench,statements,closeout,uplift",
    [
        (0, False, 1, False, False),
        (2, False, 1, False, False),
        (0, True, 1, False, False),
        (0, False, 25, False, False),
        (0, False, 1, True, False),
        (0, False, 1, True, True),
        (2, False, 1, True, True),
    ],
)
async def test_native_ragas_real_consumers_and_two_shared_repairs(
    case, tmp_path, monkeypatch, failures, workbench, statements, closeout, uplift
):
    """A candidate is committed only after a complete official RAGAS reevaluation."""
    import litagent.frozen_writing as frozen

    cfg = load_config(
        "config/operational-plus20.yaml" if uplift else "config/expanded-capacity.yaml"
    )
    cfg.eval = cfg.eval.model_copy(update={"semantic_engine": CONTRACT})
    actual = frozen.plan_frozen_budget
    if closeout:
        from litagent.closeout import PROFILE, profile_flags
        from litagent.eval.results import CONTRACT as RESULT_CONTRACT

        monkeypatch.setattr(
            writer_view,
            "source_flags",
            lambda *args: {
                **profile_flags(PROFILE),
                "evaluation_result_contract": RESULT_CONTRACT,
            },
        )
    if workbench:
        from dataclasses import replace

        import litagent.writing_v2 as writing
        from litagent.closeout import PRESET

        actual_synthesis = writing.synthesize

        async def server_synthesis(worker, task, *args):
            worker._frozen_writer_plan_provider = None
            task = replace(
                task,
                input_data={
                    key: value
                    for key, value in task.input_data.items()
                    if key not in PRESET
                    and not key.endswith("_identity")
                    and key != "round_resource_contract"
                },
            )
            return await actual_synthesis(worker, task, *args)

        monkeypatch.setattr(writing, "synthesize", server_synthesis)

    def plan(*args, **kwargs):
        kwargs.setdefault("semantic_engine", configured_engine(cfg))
        kwargs.setdefault("run_limits", configured_run_limits(cfg))
        if uplift:
            kwargs.setdefault("capacity", cfg.capacity)
        return actual(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(NativeGateway, "statements_per_unit", statements, raising=False)
    monkeypatch.setattr(structured, "LongScientificGateway", NativeGateway)
    result = await run_scientific_case(
        case, tmp_path, monkeypatch, failures=failures, config=cfg
    )
    (tmp_path / "native-result.json").write_text(json.dumps(result.report))
    saved = result.ledger["plan"]
    assert saved["envelope"]["semantic_engine"] == configured_engine(cfg)
    assert saved["stages"]["inventory"]["count"] == 5 * (1 + saved["repair_limit"])
    assert saved["maximum_requests"] == (77 if uplift else 64)
    assert saved["repair_limit"] in ((2, 3) if uplift else (2,))
    metrics = result.report["evaluation"]
    assert metrics["faithfulness"]["details"]["implementation"] == CONTRACT
    assert result.report["delivery"]["publishable"] is (
        failures == 0
    ), result.report.get("metadata", {}).get("evidence_rewrite")
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == (
        1 if failures == 0 else 2
    )
    assert any("jobs" in row["payload"] for row in result.gateway.sent)
    assert result.gateway.repair_count == (1 if failures == 0 else 2)
    assert sum("context_pool" in row["payload"] for row in result.gateway.sent) == (
        2 if failures == 0 else 3
    )
    if statements == 25:
        assert len(metrics["faithfulness"]["details"]["assertions"]) == 175
        assert result.report["delivery"]["publishable"] is True
    assert any("targets" in row["payload"] for row in result.gateway.sent)
    assert not any("cited_materials" in row["payload"] for row in result.gateway.sent)
    if uplift:
        from litagent.eval.ragas_global_wire import CONTRACT as GLOBAL_WIRE

        assert sum(
            row["payload"].get("global_input_contract") == GLOBAL_WIRE
            for row in result.gateway.sent
        ) == (2 if failures == 0 else 3)
    validate_accounting_snapshot(result.ledger, expected_plan=saved)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "combined",
        "inventory",
        "persistent",
        "combined_last",
        "extended",
        "format_agent",
        "format_agent_unit",
    ],
)
async def test_native_recovery_runs_full_sdk_reassessment(
    case, tmp_path, monkeypatch, mode
):
    """Route omissions without assigning them to the prose editor or reusing a wire."""
    import litagent.frozen_writing as frozen
    from litagent.closeout import PROFILE, profile_flags
    from litagent.eval.results import CONTRACT as RESULT_CONTRACT

    unit_view = mode == "format_agent_unit"
    if unit_view:
        mode = "format_agent"
    config_file = {
        "combined_last": "expanded-capacity",
        "extended": "extended-repair",
        "format_agent": "agent-format-repair",
    }.get(mode, "operational-plus20")
    cfg = load_config(f"config/{config_file}.yaml")
    if mode == "combined_last":
        cfg.capacity = CapacityConfig(body_json_bytes=1600)
    if mode in {"extended", "format_agent"}:
        cfg.capacity = cfg.capacity.model_copy(update={"body_json_bytes": 1600})
    cfg.eval = cfg.eval.model_copy(update={"semantic_engine": CONTRACT})
    if unit_view:
        cfg.eval = cfg.eval.model_copy(update={"native_global_input": UNIT_CONTRACT})
    actual = frozen.plan_frozen_budget

    def plan(*args, **kwargs):
        kwargs.setdefault("semantic_engine", configured_engine(cfg))
        kwargs.setdefault("run_limits", configured_run_limits(cfg))
        kwargs.setdefault("capacity", cfg.capacity)
        return actual(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(
        writer_view,
        "source_flags",
        lambda *args: {
            **profile_flags(PROFILE),
            "evaluation_result_contract": RESULT_CONTRACT,
        },
    )

    class FormattingGateway(NativeGateway):
        """Inject formatting defects at the provider response boundary only."""

        async def __call__(self, request):
            """Cover chapter, summary and repair responses in the full run."""
            body = json.loads(request.content)
            payload = json.loads(body["input"][1]["content"])
            if "anchors" in payload:
                raw = payload["raw"]
                if raw.startswith("```json\n"):
                    edits = [(0, 8), (len(raw) - 4, len(raw))]
                elif raw.endswith("}}"):
                    edits = [(len(raw) - 1, len(raw))]
                else:
                    assert raw.endswith(",}")
                    edits = [(len(raw) - 2, len(raw) - 1)]
                self.sent.append({"request": body, "payload": payload})
                result = envelope(
                    json.dumps(
                        {
                            "outcome": "patch",
                            "edits": [
                                {"start": a, "end": b, "replacement": ""}
                                for a, b in edits
                            ],
                        }
                    ),
                    model=body["model"],
                )
                result["usage"]["output_tokens_details"]["reasoning_tokens"] = 0
                return httpx.Response(200, json=result)
            response = await super().__call__(request)
            wire = response.json()
            if "output" not in wire:
                return response
            part = wire["output"][-1]["content"][0]
            raw = part["text"]
            value = json.loads(raw)
            payload = json.loads(json.loads(request.content)["input"][1]["content"])
            if "requested_body_section" in payload and "units" in value:
                part["text"] = raw + "}"
            elif {"abstract", "conclusion"} == set(value):
                part["text"] = "```json\n" + raw + "\n```"
            elif "patches" in value:
                part["text"] = raw[:-1] + ",}"
            else:
                return response
            return httpx.Response(200, json=wire)

    gateway = (
        FormattingGateway if mode in {"extended", "format_agent"} else NativeGateway
    )
    monkeypatch.setattr(
        gateway,
        "recovery_case",
        "extended" if mode == "format_agent" else mode,
        raising=False,
    )
    monkeypatch.setattr(structured, "LongScientificGateway", gateway)
    result = await run_scientific_case(
        case,
        tmp_path,
        monkeypatch,
        config=cfg,
        failures=4 if mode in {"extended", "format_agent"} else 0,
    )
    meta = result.report["metadata"]
    history = meta["evaluation_history"]
    initial = history["initial_evaluation"]["faithfulness"]
    assert initial["score"] is None
    assert initial["details"]["inventory_check"]["status"] == "failed"
    if mode in {"extended", "format_agent"}:
        attempts = meta["targeted_repair_state"]["attempts"]
        assert [row["category"] for row in attempts] == [
            *(["capacity"] * 3),
            *(["inventory"] * 3),
            *(["prose"] * 5),
        ]
        assert result.gateway.global_calls == 9
        assert result.gateway.repair_count == 5
        assert result.report["delivery"]["publishable"] is True
        format_status = meta["writer_format"]
        # Initial capacity fixes precede the first summary. Only the five
        # later prose edits require summary regeneration in this fixture.
        assert format_status["recovered_count"] == 19
        assert {row["stage"] for row in format_status["calls"]} == {
            "synthesis",
            "summary",
            "repair",
        }
        assert {row["status"] for row in format_status["calls"]} == {"recovered"}
        assert len(result.gateway.sent) == result.ledger["requests_sent"]
        if mode == "format_agent":
            guidance = CLASSIFICATION_GUIDANCE
            if unit_view:
                guidance += "\n" + EVIDENCE_SCOPE_GUIDANCE
            for payload_key in ("requested_body_section", "body_units"):
                prompts = [
                    row["request"]["input"][0]["content"]
                    for row in result.gateway.sent
                    if payload_key in row["payload"] and "targets" not in row["payload"]
                ]
                assert prompts and all(CLASSIFICATION_GUIDANCE in p for p in prompts)
                assert all(EVIDENCE_SCOPE_GUIDANCE not in p for p in prompts)
            repairs = [
                row["payload"]
                for row in result.gateway.sent
                if "targets" in row["payload"]
            ]
            assert repairs and all(
                p["writing_envelope"]["classification_guidance"] == guidance
                for p in repairs
            )
            assert format_status["requests_used"] == 19
            assert format_status["actual_tokens"] > 0
            assert format_status["unknown_tokens"] == 0
            assert format_status["requests_remaining"] == 47
            if unit_view:
                sent = [
                    row["payload"]
                    for row in result.gateway.sent
                    if row["payload"].get("global_input_contract") == UNIT_CONTRACT
                ]
                assert len(sent) == 9
                for payload in sent:
                    canonical = unpack_global_input(payload)
                    assert len(payload["claim_order"]) == len(canonical["assertions"])
                    assert all("claims" in unit for unit in payload["source_units"])
                plan = result.ledger["plan"]
                assert plan["envelope"]["semantic_engine"]["global_input_contract"] == (
                    UNIT_CONTRACT
                )
                assert all(
                    entry["reserved_tokens"] <= plan["context_reservation_limit"]
                    for entry in result.ledger["entries"].values()
                    if entry["stage"] == "global"
                )
        assert history["recovery"]["remaining_by_category"] == dict(
            capacity=0, inventory=0, prose=0
        )
        validate_accounting_snapshot(result.ledger, expected_plan=result.ledger["plan"])
        return
    assert result.gateway.global_calls == 2
    combined = mode in {"combined", "combined_last"}
    assert result.gateway.repair_count == int(combined)
    assert history["recovery"]["action"] == ("combined" if combined else "inventory")
    assert history["recovery"]["reassessment"] == "completed"
    assert result.report["delivery"]["publishable"] is (mode != "persistent")
    if not combined:
        assert (
            history["source_report_sha256"]
            == initial["details"]["source_report_sha256"]
        )
    else:
        assert "FLOPs by" not in result.report["survey"]
        assert (
            history["attempts"][0]["recovery_decision"]["budget"]["status"]
            == "admitted"
        )
    attempts = meta["targeted_repair_state"]["attempts"]
    assert len(attempts) == (2 if mode == "combined_last" else 1)
    if mode == "combined_last":
        assert attempts[0]["cause"] == "capacity"
        assert history["recovery"]["remaining_attempts"] == 0
        assert history["attempts"][0]["recovery_decision"]["remaining_attempts"] == 1
    assert (
        len(
            {
                key.split(":")[0]
                for key, row in result.ledger["entries"].items()
                if row["stage"] == "inventory"
            }
        )
        == 2
    )
    validate_accounting_snapshot(result.ledger, expected_plan=result.ledger["plan"])
