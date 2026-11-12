"""Exercise host-derived edit positions through real frozen writer consumers."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_capacity_repair import OverflowInventoryGateway, run_capacity_case
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import (
    capacity_targets,
    edit_schema,
    normalize_edit_map,
    original_unit_directory,
)
from litagent.config import AppConfig
from litagent.frozen_writing import plan_frozen_budget
from litagent.quality import digest
from litagent.writing_flat import parse
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    WriterError,
    check_source_map_response,
    provider_response_format,
    schema_for,
)
from scripts import run_frozen_writing as campaign

CONTRACT = "capacity-source-map-v3"
FIXTURE = Path(__file__).parent / "fixtures/source_unit_map_044.json"


def select_source_policy(monkeypatch):
    """Select the trusted policy without replacing real planning or execution."""
    import litagent.frozen_writing as frozen

    real_plan, real_run = frozen.plan_frozen_budget, frozen.run_frozen
    flags = {**SELECTION, "repair_mapping_contract": CONTRACT}

    def plan(*args, **kwargs):
        for key, value in flags.items():
            kwargs.setdefault(key, value)
        return real_plan(*args, **kwargs)

    async def run(*args, **kwargs):
        for key, value in flags.items():
            kwargs.setdefault(key, value)
        assert kwargs["expected_budget_plan"] is not None
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(frozen, "run_frozen", run)


def products(tmp_path):
    """Read actual persisted attempts, writer responses and accounting."""
    path = tmp_path / "repair.json"
    return (
        json.loads(path.read_text()),
        json.loads(Path(str(path) + ".writer-v2.json").read_text()),
        json.loads(Path(str(path) + ".budget-ledger.json").read_text()),
    )


def sent_with(gateway, key):
    """Filter the unmodified actual HTTP requests recorded by the fixture."""
    return [entry for entry in gateway.sent if key in entry["payload"]]


def actual_start(entry):
    """Retain actual SDK message bytes when checking the historical cost view."""
    body = entry["request"]
    return {"messages": copy.deepcopy(body.get("input", body.get("messages")))}


def test_sanitized044_preserves_the_old_global_index_failure():
    """Keep original addresses rejected without treating a projection as gold."""
    original = FIXTURE.read_bytes()
    fixture = json.loads(original)
    assert fixture["science_gold_standard"] is False
    assert fixture["contains_original_scientific_text"] is False
    assert fixture["provenance"]["projection_is_original_raw_response"] is False
    assert fixture["provenance"]["original_response_sha256"] == (
        "96ee16062001e90fba87673e7ef90737a7dd474cdd5fd22d4ed1b464ec25d073"
    )
    targets = capacity_targets(fixture["body"], {}, maximum=66)
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=True
    )
    raw = fixture["raw_response"]
    provider = provider_response_format(
        "repair", schema, wire_contract="flat-writer-v1"
    )
    Draft202012Validator(provider["json_schema"]["schema"]).validate(raw)
    parsed = parse(
        json.dumps(raw),
        schema,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
    )
    assert parsed["canonical"]["edit_records"] == raw["edit_records"]
    directory = original_unit_directory(targets)
    owners = {uid: tid for tid, ids in directory.items() for uid in ids}
    counts = {
        target: sum(p["target_id"] == target for p in raw["patches"])
        for target in targets
    }
    invalid = [
        row
        for row in raw["edit_records"]
        if any(
            index >= counts[owners[row["unit_id"]]]
            for index in row["replacement_indexes"]
        )
    ]
    cross = [
        row
        for row in raw["edit_records"]
        if any(
            0 <= index < len(raw["patches"])
            and raw["patches"][index]["target_id"] != owners[row["unit_id"]]
            for index in row["replacement_indexes"]
        )
    ]
    assert len(targets) == len(raw["patches"]) == 39
    assert sum(map(len, directory.values())) == len(raw["edit_records"]) == 45
    assert len(invalid) == 39 and len(cross) == 29
    assert sum(not row["replacement_indexes"] for row in raw["edit_records"]) == 5
    with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
        normalize_edit_map(parsed["canonical"], targets)
    assert FIXTURE.read_bytes() == original


@pytest.mark.parametrize("policy", [None, "capacity-edit-map-v2", CONTRACT])
def test_actual_prepare_validate_binds_only_the_frozen_selector(
    case, tmp_path, monkeypatch, policy
):
    """Exercise preparation and validation with mocked local inputs, not consumers."""
    from scripts.run_stage_model_quality import MODELS, ORIGIN

    monkeypatch.chdir(tmp_path)
    archives = tmp_path / "artifacts/runs"
    archives.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (archives / (run_id + ".json")).write_text(json.dumps({"source": run_id}))
    baseline = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = Path("src/litagent/test_frozen_source.py")
    source.parent.mkdir(parents=True)
    source.write_text('"""Frozen local preparation fixture; no model execution."""\n')
    bundle = {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "methodology": {},
        "extractions": [],
    }

    def local_bundle(path):
        """Read actual fixture archive bytes while retaining the real frozen plan."""
        return {
            **copy.deepcopy(bundle),
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(campaign, "read_bundle", local_bundle)
    monkeypatch.setattr(campaign, "source_paths", lambda: [source])
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    config = AppConfig(agent={}, logging={})
    config.llm.base_url = ORIGIN
    config.llm.model = MODELS["extraction"]
    for role, model in MODELS.items():
        setattr(config.llm, role + "_model", model)
    config.llm.max_tokens = 20000
    config.safety.max_cost_tokens = 500000
    for component in (
        config.planner,
        config.extractor,
        config.eval,
        config.adversarial,
    ):
        component.thinking_mode = "disabled"
    config.adversarial.review_max_tokens = 4096
    config.eval.max_tokens = 16384
    root = tmp_path / "only-offline-prepared-fixture"
    selector = {} if policy is None else {"repair_mapping_contract": policy}
    campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        **SELECTION,
        **selector,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert campaign.validate(root, config) == manifest
    assert campaign.expected_limits(manifest) == manifest["limits"]
    assert "repair_mapping_contract" not in manifest["options"]
    for container in (manifest, manifest["writing_parameters"]):
        assert ("repair_mapping_contract" in container) is (policy is not None)
    assert not any(
        (root / name).exists()
        for name in ("state.json", "offline-gates.json", "launch-intent.json")
    )
    for row in manifest["cases"]:
        plan = manifest["budget_plans"][row["case_id"]]
        assert plan == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            **SELECTION,
            **selector,
        )
        assert plan["envelope"]["repair_mapping_contract"] == (
            policy or "capacity-edit-map-v2"
        )
    if policy != CONTRACT:
        return
    original = copy.deepcopy(manifest)
    manifest["writing_parameters"].pop("repair_mapping_contract")
    manifest["identity"] = campaign.digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen_writing_parameters_conflict"):
        campaign.validate(root, config)
    manifest = copy.deepcopy(original)
    manifest.pop("repair_mapping_contract")
    manifest["writing_parameters"].pop("repair_mapping_contract")
    manifest["identity"] = campaign.digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        campaign.validate(root, config)


class SourceMapGateway(OverflowInventoryGateway):
    """Generate independent source declarations only at the HTTP boundary."""

    def __init__(self, case, *, mode="valid", unknown=False):
        super().__init__(case)
        self.mode = mode
        self.unknown = unknown
        self.capacity_responses = []
        self.declined = mode in {"wrong_paper", "missing_core"}
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "cited_materials" in payload and mode == "wrong_paper":
                claims = {c["claim_id"]: c for c in payload["assertions"]}
                for decision in value["decisions"]:
                    claim = claims[decision["claim_id"]]
                    if "Evo-ViT" not in claim["claim_text"]:
                        continue
                    assert claim["evidence"]
                    decision.update(
                        status="unsupported",
                        reason_code="attribution_mismatch",
                        support_basis=[],
                        explanation="The cited EViT source does not support the Evo-ViT attribution.",
                    )
                    for aspect in decision["aspect_checks"]:
                        aspect.update(supported=False, support_ids=[])
            if "coverage" in value and mode == "missing_core":
                assert not any(
                    "DeiT-S" in c["claim_text"] for c in payload["assertions"]
                )
                value["coverage"][0].update(
                    answered=False,
                    claim_ids=[],
                    explanation="The required DeiT-S condition was removed.",
                )
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat

    def answer(self, payload, native_schema):
        """Copy host source IDs; never ask the model fixture to invent indexes."""
        if payload.get("cause") != "capacity":
            value = super().answer(payload, native_schema)
            if (
                self.mode == "missing_core"
                and payload.get("requested_body_section") == "methods"
            ):
                value["units"].append(
                    {
                        "text": "On DeiT-S, EViT reduces computation.",
                        "fact_card_refs": [self.case.by_eid["a"]],
                        "paragraph_start": False,
                    }
                )
            return value
        patches, deleted = [], []
        for tid, target in payload["targets"].items():
            section, pi, start = target["path"]
            units = payload["current_body"][section][pi]["sentences"]
            ids = target["original_unit_ids"]
            if self.mode.startswith("split_merge") and section == "methods":
                patches.append(
                    {
                        "target_id": tid,
                        **copy.deepcopy(units[start]),
                        "source_unit_ids": list(ids),
                    }
                )
                continue
            for ordinal, uid in enumerate(ids):
                index = start + ordinal
                if section == "methods" and index >= 1:
                    deleted.append(
                        {
                            "unit_id": uid,
                            "reason": "Remove this explicitly selected redundant unit or controlled core-condition regression.",
                        }
                    )
                    continue
                if self.mode.startswith("split_merge") and section == "introduction":
                    patches.extend(
                        [
                            {
                                "target_id": tid,
                                "text": "Relation networks learn a comparison function.",
                                "fact_card_refs": [self.case.by_eid["b"]],
                                "source_unit_ids": [uid],
                            },
                            {
                                "target_id": tid,
                                "text": "On DeiT-S, EViT reduces computation.",
                                "fact_card_refs": [self.case.by_eid["a"]],
                                "source_unit_ids": [uid],
                            },
                        ]
                    )
                    continue
                patch = {
                    "target_id": tid,
                    **copy.deepcopy(units[index]),
                    "source_unit_ids": [uid],
                }
                if self.mode == "wrong_paper" and section == "methods":
                    patch.update(
                        text="Evo-ViT learns the comparison function.",
                        fact_card_refs=[self.case.by_eid["a"]],
                    )
                patches.append(patch)
        if self.mode == "split_merge_interleaved":
            first_target = patches[0]["target_id"]
            same = [patch for patch in patches if patch["target_id"] == first_target]
            others = [patch for patch in patches if patch["target_id"] != first_target]
            assert len(same) > 1 and others
            patches = [same[0], *others[:1], *same[1:], *others[1:]]
        value = {
            "outcome": "patched",
            "explanation": "Independent source relations; host computes replacement positions.",
            "patches": patches,
            "deleted_sources": deleted,
        }
        self.capacity_responses.append(copy.deepcopy(value))
        return value

    async def __call__(self, request):
        """Drop usage only after receiving the controlled actual provider reply."""
        body = json.loads(request.content)
        messages = body.get("input", body.get("messages"))
        payload = json.loads(messages[1]["content"])
        response = await super().__call__(request)
        if self.unknown and payload.get("cause") == "capacity":
            data = response.json()
            data.pop("usage", None)
            return httpx.Response(200, json=data)
        return response


async def run_source_case(case, tmp_path, monkeypatch, *, mode="valid", unknown=False):
    """Keep the actual frozen provider, graph, report, evaluation and transaction."""
    if mode.startswith("split_merge"):
        unit = case.body["introduction"][0]["sentences"][0]
        unit.update(
            text="Relation networks learn a comparison function; on DeiT-S, EViT reduces computation.",
            fact_card_refs=[case.by_eid["b"], case.by_eid["a"]],
        )
    if mode == "missing_core":
        case.packet["requirements"][0][
            "text"
        ] = "Explain the EViT setting on DeiT-S and limitations."
    select_source_policy(monkeypatch)
    gateway = SourceMapGateway(case, mode=mode, unknown=unknown)
    report, _ = await run_capacity_case(
        case, tmp_path, monkeypatch, gateway=gateway, allow_uncertain=unknown
    )
    return report, gateway


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "split_merge", "split_merge_interleaved"])
async def test_real_frozen_chain_derives_positions_refreshes_summary_and_commits(
    case, tmp_path, monkeypatch, mode
):
    """A legitimate map still traverses every existing scientific delivery gate."""
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, mode=mode)
    state, journal, budget = products(tmp_path)
    assert report["delivery"]["publishable"] is True
    details = report["evaluation"]["faithfulness"]["details"]
    assert (
        details["coverage"]["complete"]
        and details["stages"]["global_check"] == "completed"
    )
    assert details["supported_count"] == details["factual_claim_count"] > 0
    assert details["unsupported_count"] == details["unverified_count"] == 0
    assert gateway.inventory_calls == 2 and len(gateway.capacity_responses) == 1
    assert len(sent_with(gateway, "body_units")) == 2
    assert any("report_layout" in row["payload"] for row in gateway.sent)
    assert len(state["state"]["attempts"]) == 1
    call = journal["calls"]["repair_0"]
    raw = json.loads(call["raw"])
    assert set(raw) == {"outcome", "explanation", "patches", "deleted_sources"}
    assert all(
        "source_unit_ids" in patch and "replacement_index" not in patch
        for patch in raw["patches"]
    )
    assert "edit_records" not in raw
    assert call["repair_mapping_contract"] == CONTRACT
    assert call["source_relation_bindings"]
    assert call["source_relations_sha256"] and call["source_relation_bindings_sha256"]
    parsed = parse(
        call["raw"],
        call["reference_schema"],
        reference_contract=SELECTION["reference_contract"],
        transport_revision=SELECTION["transport_revision"],
    )
    check_source_map_response(budget["plan"]["envelope"], call, parsed)
    for name in ("canonical", "received_canonical"):
        assert all(
            set(unit) == {"text", "fact_card_refs"}
            for patch in parsed[name]["patches"]
            for unit in patch["sentences"]
        )
    expected, groups = [], {}
    for patch in raw["patches"]:
        groups.setdefault(patch["target_id"], []).append(patch)
    for tid, group in groups.items():
        for index, patch in enumerate(group):
            expected.append(
                {
                    "target_id": tid,
                    "replacement_index": index,
                    "source_unit_ids": patch["source_unit_ids"],
                }
            )
    assert parsed["canonical"]["source_relations"] == expected
    assert sum(row["replacement_index"] == 0 for row in expected) >= 2
    campaign._historical_call_binding(call, budget["plan"]["envelope"])
    if call["source_relation_bindings"]["deleted_sources"]:
        damaged = copy.deepcopy(call)
        damaged["source_relation_bindings"]["deleted_sources"][0]["canonical_path"][
            1
        ] = False
        damaged["source_relation_bindings_sha256"] = digest(
            damaged["source_relation_bindings"]
        )
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, budget["plan"]["envelope"])
    if mode == "split_merge_interleaved":
        assert any(
            row["wire_ordinal"] != index
            for index, row in enumerate(call["source_relation_bindings"]["relations"])
        )
    assert budget["requests_sent"] == budget["call_count"] == len(gateway.sent)
    assert all(entry["status"] == "settled" for entry in budget["entries"].values())
    assert budget["usage_uncertain"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wrong_paper", "missing_core"])
async def test_valid_relationships_never_commit_wrong_science_or_missing_core(
    case, tmp_path, monkeypatch, mode
):
    """Scientific and coverage failures remain independent of a legal lineage map."""
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, mode=mode)
    _, journal, budget = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert journal["candidates"] and not any(
        item.get("committed") for item in journal["candidates"]
    )
    candidate = journal["candidates"][0]
    details = candidate["evaluation"]["faithfulness"]["details"]
    assert (
        details["coverage"]["complete"]
        and details["stages"]["global_check"] == "completed"
    )
    if mode == "wrong_paper":
        assert "Evo-ViT" in candidate["survey"] and "Evo-ViT" not in report["survey"]
        assert details["unsupported_count"] > 0
        attempts = report["metadata"]["targeted_repair_state"]["attempts"]
        assert len(attempts) == 2
        assert len(sent_with(gateway, "targets")) == 2
    else:
        assert "DeiT-S" not in candidate["survey"]
        assert any(not row["answered"] for row in details["core_coverage"])
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) <= 2
    assert budget["requests_sent"] == len(gateway.sent)
    assert all(entry["status"] == "settled" for entry in budget["entries"].values())


@pytest.mark.asyncio
async def test_saved_source_metadata_is_recomputed_from_actual_raw_before_reuse(
    case, tmp_path, monkeypatch
):
    """Fresh parsing refuses metadata changes without sending another request."""
    _, gateway = await run_source_case(
        case, tmp_path, monkeypatch, mode="split_merge_interleaved"
    )
    _, journal, budget = products(tmp_path)
    original = copy.deepcopy(journal["calls"]["repair_0"])
    parsed = parse(
        original["raw"],
        original["reference_schema"],
        reference_contract=SELECTION["reference_contract"],
        transport_revision=SELECTION["transport_revision"],
    )
    envelope = budget["plan"]["envelope"]
    check_source_map_response(envelope, original, parsed)
    campaign._historical_call_binding(original, envelope)
    count = len(gateway.sent)
    for field in (
        "repair_mapping_contract",
        "source_map_identity",
        "source_relation_bindings_sha256",
        "source_relations_sha256",
    ):
        damaged = copy.deepcopy(original)
        damaged[field] = "0" * 64
        with pytest.raises(WriterError, match="writer_source_map_receipt_invalid"):
            check_source_map_response(envelope, damaged, parsed)
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
        damaged = copy.deepcopy(original)
        damaged.pop(field)
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["source_relation_bindings"] = {}
    with pytest.raises(WriterError, match="writer_source_map_receipt_invalid"):
        check_source_map_response(envelope, damaged, parsed)
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["source_relation_bindings"]["canonical_map_sha256"] = "0" * 64
    damaged["source_relation_bindings_sha256"] = digest(
        damaged["source_relation_bindings"]
    )
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["reference_canonical"]["source_relations"][0]["replacement_index"] += 1
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    for field in ("wire_ordinal", "replacement_index"):
        damaged = copy.deepcopy(original)
        damaged["source_relation_bindings"]["relations"][0][field] = False
        damaged["source_relation_bindings_sha256"] = digest(
            damaged["source_relation_bindings"]
        )
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["reference_canonical"]["source_relations"][0]["replacement_index"] = False
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    binding = damaged["source_relation_bindings"]["relations"][0]["source_ids"][0]
    binding["wire_path"][1] = False
    damaged["source_relation_bindings_sha256"] = digest(
        damaged["source_relation_bindings"]
    )
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["source_relation_bindings"]["relations"][0]["canonical_path"][1] = False
    damaged["source_relation_bindings_sha256"] = digest(
        damaged["source_relation_bindings"]
    )
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    first, second = damaged["source_relation_bindings"]["relations"][:2]
    left, right = first["source_ids"][0], second["source_ids"][0]
    assert left["value_sha256"] == right["value_sha256"]
    assert (left["start"], left["end"]) != (right["start"], right["end"])
    for field in ("start", "end", "raw_sha256", "value_sha256"):
        left[field] = right[field]
    damaged["source_relation_bindings_sha256"] = digest(
        damaged["source_relation_bindings"]
    )
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    damaged = copy.deepcopy(original)
    damaged["source_relation_bindings"]["relations"][0]["target_id"]["start"] += 1
    damaged["source_relation_bindings_sha256"] = digest(
        damaged["source_relation_bindings"]
    )
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    source_fields = {
        "repair_mapping_contract",
        "source_map_identity",
        "source_relations_sha256",
        "source_relation_bindings_sha256",
        "source_relation_bindings",
    }
    damaged = copy.deepcopy(original)
    for name in ("reference_canonical", "received_canonical"):
        damaged[name].pop("source_relations")
        damaged[name].pop("deleted_sources")
    for field in source_fields:
        damaged.pop(field)
    with pytest.raises(
        ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
    ):
        campaign._historical_call_binding(damaged, envelope)
    ordinary = [
        call
        for name, call in journal["calls"].items()
        if name.startswith("body_") or name.startswith("summary_")
    ]
    assert ordinary
    for call in ordinary:
        assert not source_fields.intersection(call)
        campaign._historical_call_binding(call, envelope)
        damaged = copy.deepcopy(call)
        damaged["source_map_identity"] = envelope["source_map_identity"]
        with pytest.raises(
            ValueError, match="frozen_accounting_mixed_source_map_call_invalid"
        ):
            campaign._historical_call_binding(damaged, envelope)
    starts = [row for row in gateway.sent if "writing_envelope" in row["payload"]]
    assert starts
    for entry in starts:
        campaign._historical_writer_envelope(actual_start(entry), envelope)
    entry = sent_with(gateway, "targets")[0]
    for field in (
        "repair_mapping_contract",
        "source_map_identity",
        "source_map_json_bytes",
    ):
        for remove in (False, True):
            damaged = actual_start(entry)
            payload = json.loads(damaged["messages"][1]["content"])
            if remove:
                payload["writing_envelope"].pop(field)
            else:
                payload["writing_envelope"][field] = "wrong"
            damaged["messages"][1]["content"] = json.dumps(payload)
            with pytest.raises(
                ValueError,
                match="frozen_accounting_mixed_writer_source_map_envelope_invalid",
            ):
                campaign._historical_writer_envelope(damaged, envelope)
    assert journal["calls"]["repair_0"] == original and len(gateway.sent) == count


@pytest.mark.asyncio
async def test_unknown_map_response_keeps_sent_reservation_and_does_not_retry(
    case, tmp_path, monkeypatch
):
    """An unmeasured reply cannot release its escrow or consume a third attempt."""
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, unknown=True)
    _, journal, budget = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert (
        budget["usage_uncertain"] is True
        and budget["stop_reason"] == "strict_usage_unknown"
    )
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert (
        report["metadata"]["targeted_repair_state"]["attempts"][0]["cause"]
        == "capacity"
    )
    assert len(sent_with(gateway, "targets")) == 1 and gateway.inventory_calls == 1
    unknown = [
        entry
        for entry in budget["entries"].values()
        if entry["sent"] and entry["status"] != "settled"
    ]
    assert len(unknown) == 1
    assert unknown[0]["actual_tokens"] is None and unknown[0]["reserved_tokens"] > 0
    assert budget["requests_sent"] == len(gateway.sent)
    call = journal["calls"]["repair_0"]
    for field in (
        "repair_mapping_contract",
        "source_map_identity",
        "source_relations_sha256",
        "source_relation_bindings_sha256",
        "source_relation_bindings",
    ):
        assert field not in call
    campaign._historical_call_binding(call, budget["plan"]["envelope"])
