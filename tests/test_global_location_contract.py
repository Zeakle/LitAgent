"""Keep host locations independent of model explanations without clearing findings."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from jsonschema import Draft202012Validator, ValidationError
from test_native_ragas_evaluation import SemanticTransport, native_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.engine_policy import (
    ANCHORED_GLOBAL_REVIEW,
    ARCHIVED_QUOTED_OMISSION_ENGINES,
    KEYED_ASSESSMENT,
    KEYED_GLOBAL_LOCATED_OMISSIONS,
    KEYED_GLOBAL_QUOTED_OMISSIONS,
    configured_engine,
    engine_descriptor,
    validate_engine,
)
from litagent.eval.ragas_global_wire import UNIT_CONTRACT, unpack_global_input
from litagent.eval.ragas_native import (
    canonical_assertions,
    global_review_request,
    parse_native_global_response,
    source_samples,
)
from litagent.report_render import build_report_source_units
from litagent.review_cycles import improvement
from litagent.runner import LitAgent
from litagent.safety.budget import strict_request_reservation


@pytest_asyncio.fixture
async def located_case(native_case):
    """Use the actual request, schema and response consumers."""
    survey, context = native_case
    _, units, _, samples = source_samples(survey, context)
    for sample in samples:
        sample["statements"] = [sample["response"]]
        sample["verdicts"] = [{"verdict": 0, "reason": "Not supported"}]
    assertions, _ = canonical_assertions(samples)
    cfg = load_config("config/separate-review-evaluation.yaml")
    payload, messages, fmt = global_review_request(
        survey,
        context,
        cfg.eval,
        units,
        assertions,
        anchored=True,
        unit_view=True,
        unit_response_contract=KEYED_GLOBAL_LOCATED_OMISSIONS,
    )
    response = await SemanticTransport().chat(messages)
    raw = json.loads(response.content)
    raw["issues"] = [
        {
            "code": "missing_assertion",
            "anchor_ids": ["u1"],
            "explanation": "The unit's labeled-example premise appears unrepresented.",
        }
    ]
    return payload, messages, fmt, raw


@pytest.mark.asyncio
async def test_explanation_is_not_location_and_omission_remains_negative(located_case):
    payload, messages, fmt, raw = located_case
    assert unpack_global_input(json.loads(messages[1]["content"])) == payload
    assert "this is not a verbatim quote" in json.dumps(fmt)
    assert "unsupported and translated equivalents" in messages[0]["content"]
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(raw)
    progress = {}
    parsed = parse_native_global_response(json.dumps(raw), payload, progress)
    unit = payload["source_units"][0]
    (issue,) = parsed.issues
    assert issue.owner == "inventory" and issue.code == "missing_assertion"
    assert issue.source_unit_id == unit["source_unit_id"]
    assert issue.explanation == raw["issues"][0]["explanation"]
    assert progress["omission_locations"] == [
        {
            "issue_index": 0,
            "coverage_key": "u1",
            "source_unit_id": unit["source_unit_id"],
            "source_quote": unit["text"],
            "extracted_claim_ids": unit["extracted_claim_ids"],
        }
    ]
    assert all(c["status"] == "unsupported" for c in payload["assertions"])
    # The same content at another position does not acquire this finding's location.
    raw["issues"][0]["anchor_ids"] = [payload["source_units"][-1]["coverage_key"]]
    remapped = parse_native_global_response(json.dumps(raw), payload, {})
    assert (
        remapped.issues[0].source_unit_id
        == payload["source_units"][-1]["source_unit_id"]
    )
    assert remapped.issues[0].source_unit_id != issue.source_unit_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keys,reason",
    [
        (["u999"], "unknown_or_wrong_role"),
        (["claim_0"], "unknown_or_wrong_role"),
        (["u1", "u1"], "duplicate_key"),
        (["u1", "u2"], "omission_requires_one_unit"),
        ([1], "unknown_or_wrong_role"),
    ],
)
async def test_bad_locations_are_not_repaired_or_guessed(located_case, keys, reason):
    payload, _, fmt, raw = located_case
    raw["issues"][0]["anchor_ids"] = keys
    with pytest.raises(ValidationError):
        Draft202012Validator(fmt["json_schema"]["schema"]).validate(raw)
    progress = {}
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_native_global_response(json.dumps(raw), payload, progress)
    assert progress["global_validation_error"]["reason"] == reason


@pytest.mark.asyncio
async def test_missing_coverage_and_invalid_json_remain_errors(located_case):
    payload, _, _, raw = located_case
    del raw["units"]["u2"]
    with pytest.raises(ValueError, match="evaluation_global_unit_coverage_invalid"):
        parse_native_global_response(json.dumps(raw), payload, {})
    with pytest.raises(ValueError):
        parse_native_global_response('{"units":', payload, {})


@pytest.mark.parametrize("input_contract", [None, UNIT_CONTRACT])
@pytest.mark.parametrize("assessment", [None, KEYED_ASSESSMENT])
def test_old_quoted_descriptor_is_read_only(input_contract, assessment):
    old = engine_descriptor(
        global_findings_contract=ANCHORED_GLOBAL_REVIEW,
        global_input_contract=input_contract,
        global_unit_response_contract=KEYED_GLOBAL_QUOTED_OMISSIONS,
        assessment_response_contract=assessment,
    )
    old["identity"] = ARCHIVED_QUOTED_OMISSION_ENGINES[input_contract, assessment]
    assert validate_engine(old, archived=True) == old
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(old)
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(
            {**old, "global_unit_response_contract": KEYED_GLOBAL_LOCATED_OMISSIONS},
            archived=True,
        )
    assert (
        configured_engine(load_config("config/agent-format-repair.yaml"))[
            "global_unit_response_contract"
        ]
        == KEYED_GLOBAL_QUOTED_OMISSIONS
    )
    assert (
        configured_engine(load_config("config/separate-review-evaluation.yaml"))[
            "global_unit_response_contract"
        ]
        == KEYED_GLOBAL_LOCATED_OMISSIONS
    )


def _archive_409():
    root = Path("artifacts/acceptance/workbench-few-shot-409-20261111")
    path = root / "runs/d0d2a15706dd.json"
    if not path.exists():
        pytest.skip("Private immutable 409 is not distributed")
    original = path.read_bytes()
    assert (
        hashlib.sha256(original).hexdigest()
        == "5460eba151d0eed5e5dceb08b835cd39b9af8644baf7e944e210fe0d4e6db6ca"
    )
    return root, path, original, json.loads(original)


def test_409_original_rejected_new_request_keeps_same_omission_and_nli():
    root, path, original, archive = _archive_409()
    nodes = []
    for node in archive["nodes"].values():
        messages = (node.get("input") or {}).get("messages") or []
        if (
            node.get("kind") == "llm"
            and messages
            and messages[0]["content"].startswith("Review the complete frozen report")
        ):
            nodes.append(node)
    node = nodes[-1]
    old = unpack_global_input(json.loads(node["input"]["messages"][1]["content"]))
    response = node["output"]
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        parse_native_global_response(response, old, {})
    journal = json.loads(
        (
            root / "runtime/targeted-repairs/d0d2a15706dd.json.writer-v2.json"
        ).read_bytes()
    )
    survey = journal["candidates"][-1]["survey"]
    directory = {u.unit_id: u for u in build_report_source_units(survey)}
    cfg = load_config("config/separate-review-evaluation.yaml").eval
    payload, messages, fmt = global_review_request(
        survey,
        old,
        cfg,
        [directory[u["source_unit_id"]] for u in old["source_units"]],
        old["assertions"],
        anchored=True,
        unit_view=True,
        unit_response_contract=KEYED_GLOBAL_LOCATED_OMISSIONS,
    )
    assert payload["assertions"] == old["assertions"]
    assert payload["source_units"] == old["source_units"]
    assert strict_request_reservation(messages, cfg.max_tokens, fmt) <= int(
        cfg.context_tokens * (1 - cfg.strict_context_margin)
    )
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(json.loads(response))
    progress = {}
    result = parse_native_global_response(response, payload, progress)
    assert len(result.issues) == 1 and result.issues[0].owner == "inventory"
    (location,) = progress["omission_locations"]
    assert location["coverage_key"] == "u7"
    assert "claim_39" in location["extracted_claim_ids"]
    assert "claim_39" in {
        c["claim_id"] for c in payload["assertions"] if c["status"] == "unsupported"
    }
    assert sum(c["status"] == "supported" for c in payload["assertions"]) == 355
    assert sum(c["status"] == "unsupported" for c in payload["assertions"]) == 22
    assert any(
        not u.complete and u.source_unit_id == location["source_unit_id"]
        for u in result.units
    )
    assert archive["report"]["delivery"]["publishable"] is False
    assert path.read_bytes() == original


def test_409_science_regression_cannot_promote_a_working_draft():
    root, path, original, archive = _archive_409()
    journal = json.loads(
        (
            root / "runtime/targeted-repairs/d0d2a15706dd.json.writer-v2.json"
        ).read_bytes()
    )
    for index, record in enumerate(journal["candidates"]):
        candidate = SimpleNamespace(
            candidate=record["survey"],
            candidate_evaluation=record["evaluation"],
            candidate_bindings=record["bindings"],
            candidate_metadata={
                "writer_document": record["writer_document"],
                "targeted_repair_state": {
                    "attempts": [{"provenance": record["repair_provenance"]}]
                },
            },
        )
        decision = improvement(
            archive["report"], candidate, LitAgent._passed_metric_regressed
        )
        assert decision["promote"] is False
        if index == 0:
            # Fewer problem groups alone is not improvement: new failing positions
            # must prevent promotion even though 15 groups became 11.
            assert decision["reason"] == "no_bound_improvement"
            assert decision["previous_problem_groups"] == 15
            assert decision["candidate_problem_groups"] == 11
        else:
            assert decision["reason"] == "candidate_not_independently_verified"
        assert not record["committed"]
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_located_omission_reaches_combined_repair_and_complete_reevaluation(
    case, tmp_path, monkeypatch
):
    """Only mock provider transport, including the erroneous omission opinion."""
    from test_separate_review_pipeline import SeparateGateway, execute

    gateway = SeparateGateway(case, reviews=0)
    gateway.recovery_case = "combined"
    report = await execute(case, tmp_path, monkeypatch, gateway)
    history = report["metadata"]["evaluation_history"]
    first = history["initial_evaluation"]["faithfulness"]
    assert first["score"] is None
    assert first["details"]["inventory_check"]["status"] == "failed"
    assert first["details"]["stages"]["global_check"] == "completed"
    assert first["details"]["stages"]["omission_locations"]
    (attempt,) = history["attempts"]
    assert attempt["recovery_decision"]["action"] == "combined"
    assert attempt["committed"]
    assert gateway.repair_count == 1 and gateway.global_calls == 2
    assert report["evaluation"]["faithfulness"]["score"] == 1
    assert report["delivery"]["publishable"] is True
