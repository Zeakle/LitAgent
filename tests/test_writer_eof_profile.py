"""Bind EOF compilation to an explicit frozen profile and safe public receipts."""

import copy
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    body_section_schema,
    normalize_eof_response,
)
from scripts import run_frozen_writing as campaign


def test_profile_is_explicit_and_preserves_legacy_roles(configured):
    """The default and prior re-emission profiles retain their exact parameters."""
    before = copy.deepcopy(configured.model_dump())
    assert campaign.profile_parameters(campaign.ROLE_PROFILE) == {
        "native_reasoning_effort": "low",
        "native_reviewer_thinking_mode": "disabled",
        "native_provider_api": "responses",
        "native_reemission_contract": "intact-unit-reemission-v2",
    }
    assert campaign.profile_parameters(campaign.EOF_PROFILE) == {
        "native_reasoning_effort": "low",
        "native_reviewer_thinking_mode": "disabled",
        "native_provider_api": "responses",
        "native_syntax_policy": EOF_CONTAINER_POLICY,
    }
    config = campaign.profile_config(configured, campaign.EOF_PROFILE)
    campaign.check_profile(config, campaign.EOF_PROFILE)
    description = campaign.profile_description(campaign.EOF_PROFILE, {})
    assert description["syntax_policy"] == EOF_CONTAINER_POLICY
    assert description["reviewer_thinking_mode"] == "disabled"
    assert "reemission_contract" not in description
    assert "syntax_policy" not in campaign.profile_description(
        campaign.ROLE_PROFILE, {}
    )
    assert before == configured.model_dump()


def test_frozen_identity_rejects_policy_drift_even_if_rehashed(manifest):
    """Policy selection is part of the server contract, not arbitrary caller data."""
    root, config = manifest
    target = root.with_name("eof-identity")
    config = campaign.profile_config(config, campaign.EOF_PROFILE)
    campaign.prepare(target, config, writing_profile=campaign.EOF_PROFILE)
    value = campaign.validate(target, config)
    assert value["writing_parameters"]["native_syntax_policy"] == EOF_CONTAINER_POLICY
    value["writing_parameters"].pop("native_syntax_policy")
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    atomic_json(target / "manifest.json", value)
    with pytest.raises(ValueError, match="parameters_conflict"):
        campaign.validate(target, config)


def receipt(case):
    """Create a real parser receipt rather than a fabricated accepted score."""
    schema = body_section_schema("taxonomy", case.aliases)
    raw = json.dumps({"taxonomy": case.body["taxonomy"]})[:-2]
    _, normalized = normalize_eof_response(raw, schema)
    normalized.update(provider_schema_sha256="a" * 64, input_sha256="b" * 64)
    return {
        "syntax_policy": EOF_CONTAINER_POLICY,
        "calls": {
            "body:taxonomy": {
                "raw": raw,
                "normalized_raw": raw + normalized["suffix"],
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "syntax_normalization": normalized,
                **{
                    k: normalized[k]
                    for k in ("schema_sha256", "provider_schema_sha256", "input_sha256")
                },
            }
        },
    }


def test_exports_share_syntax_receipts_without_scientific_promotion(manifest, case):
    """All four formats keep scientific status unavailable for this parser fixture."""
    root, config = manifest
    target = root.with_name("eof-export")
    config = campaign.profile_config(config, campaign.EOF_PROFILE)
    campaign.prepare(target, config, writing_profile=campaign.EOF_PROFILE)
    value = campaign.validate(target, config)
    rid = "eof-export-few-shot-metric"
    path = Path("artifacts/targeted-repairs", rid + ".json.writer-v2.json")
    journal = receipt(case)
    atomic_json(path, journal)
    state = {
        "identity": value["identity"],
        "status": "offline_fixture",
        "cases": [
            {
                "case_id": "few-shot-metric",
                "run_id": rid,
                "status": "blocked",
                "generated": False,
                "publishable": False,
                "evaluation": "not_run",
            }
        ],
    }
    public = campaign.export(target, state, value)
    row = public["cases"][0]
    assert row["syntax_normalization_count"] == 1
    assert row["syntax_normalizations"][0]["scientific_assessment"] == "not_verified"
    assert not row["publishable"] and public["passed"] == 0
    out = Path("benchmarks/survey/results") / target.name
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    csv_rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert csv_rows[0]["syntax_normalization_count"] == "1"
    assert "Host syntax normalization" in (out / "benchmark.md").read_text()
    assert journal["calls"]["body:taxonomy"]["raw"] not in html
    assert public["cases"][1]["syntax_normalization_count"] is None


@pytest.mark.parametrize(
    "fault", ["raw", "candidate", "hash", "schema", "policy", "semantic", "call"]
)
def test_corrupt_receipt_is_unavailable_without_borrowing_other_calls(
    case, tmp_path, monkeypatch, fault
):
    """No raw content or unvalidated hashes may leak through the public projection."""
    monkeypatch.chdir(tmp_path)
    journal = receipt(case)
    entry = journal["calls"]["body:taxonomy"]
    if fault in ("raw", "candidate"):
        entry["raw" if fault == "raw" else "normalized_raw"] += "tampered"
    elif fault == "call":
        journal["calls"]["reviewer"] = journal["calls"].pop("body:taxonomy")
    elif fault == "schema":
        entry["schema_sha256"] = "bad"
    else:
        entry["syntax_normalization"][
            {"hash": "raw_sha256", "policy": "policy", "semantic": "semantic_sha256"}[
                fault
            ]
        ] = "PRIVATE_SOURCE_TEXT"
    atomic_json(Path("artifacts/targeted-repairs/r.json.writer-v2.json"), journal)
    assert campaign.syntax_normalizations("r", EOF_CONTAINER_POLICY) is None
    assert campaign.syntax_normalizations("../r", EOF_CONTAINER_POLICY) is None
    assert campaign.syntax_normalizations("r", None) is None


def test_trace_keeps_only_identity_and_never_invents_a_model_call():
    """The real redaction and LangFuse consumer handle the new metadata event."""
    tracer = LangFuseTracer("", "", "")
    tracer._root = MagicMock()
    safe = {
        "contract_version": EOF_CONTAINER_POLICY,
        "phase": "synthesis",
        "input_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "count": 2,
        "status": "syntax_only_not_verified",
        "binding": {"raw_sha256": "c" * 64},
    }
    hook = RedactingTraceHook(
        lambda event, data: tracer._handle(event, data), payload_mode="metadata_only"
    )
    hook(
        "writer.syntax_normalized",
        {
            **safe,
            "raw": "PRIVATE_TEXT",
            "normalized_raw": "PRIVATE_TEXT",
            "authorization": "Bearer secret",
        },
    )
    metadata = tracer._root.update.call_args.kwargs["metadata"]["writing_syntax"]
    assert metadata == safe
    assert "PRIVATE_TEXT" not in json.dumps(metadata)
    assert not tracer._root.start_observation.called
