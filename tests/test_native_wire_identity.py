"""Bind the explicit v4 wire contract without reinterpreting historical runs."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.report_audit import atomic_json
from litagent.workbench import RunOptions, VerificationOptions
from scripts import run_frozen_writing as campaign


def writer_options(policy="legacy"):
    """Choose the same strict options without optional paid services."""
    return RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v2",
        verification=VerificationOptions(judge=False, jev=False),
        budget_policy=policy,
    )


@pytest.mark.parametrize(
    "policy,expected",
    [
        (
            None,
            "a66dbe38a4c2354825f756e5e219b6f8139716683617cfa8e6afd00b7b4ee75b",
        ),
        (
            "legacy",
            "0f732d09f533f7715ac269161ddfbb63ed35b41de0db4e04ec7844ff85b0c42d",
        ),
        (
            "planned-budget-v2",
            "2d0fe0ded03587cd2506a9a60c36258b0134d4dd6e36b8be7eac1c5ce05d4cd3",
        ),
        (
            "planned-budget-v3",
            "796619d7a78eb40935576cc97ed89601a2a459533659fadc6075a0588c61336d",
        ),
    ],
)
def test_pre_wire_option_fingerprints_are_exact_goldens(policy, expected):
    """Keep identities captured before the approved v4 wire implementation."""
    options = RunOptions() if policy is None else writer_options(policy)
    assert options.fingerprint() == expected
    assert "writer_wire_contract" not in options.model_dump()


def test_only_v4_options_bind_the_central_wire_contract(monkeypatch):
    """A transport change invalidates v4 without mutating old request defaults."""
    import litagent.writing_v2 as writer

    options = {
        policy: writer_options(policy)
        for policy in (
            "legacy",
            "planned-budget-v2",
            "planned-budget-v3",
            "planned-budget-v4",
        )
    }
    original = {policy: value.fingerprint() for policy, value in options.items()}
    assert original["planned-budget-v4"] != (
        "ff25cf024d6864d0269011305384c314dad0587f5ba36d5ffd1dfe109578fe22"
    )
    monkeypatch.setattr(writer, "NATIVE_WIRE_CONTRACT", "future-wire-contract-test")
    changed = {policy: value.fingerprint() for policy, value in options.items()}
    for policy in ("legacy", "planned-budget-v2", "planned-budget-v3"):
        assert changed[policy] == original[policy]
    assert changed["planned-budget-v4"] != original["planned-budget-v4"]


def test_citation_style_remains_outside_new_wire_generation_identity():
    """Use the actual request model while preserving the API's options boundary."""
    from litagent.api import SurveyRequest

    fingerprints = []
    for style in ("ieee", "apa7", "evidence"):
        request = SurveyRequest(
            query="Offline citation identity question",
            citation_style=style,
            **writer_options("planned-budget-v4").model_dump(),
        )
        options = RunOptions(**request.model_dump(include=set(RunOptions.model_fields)))
        assert "classification_review" not in SurveyRequest.model_fields
        assert not options.classification_review
        fingerprints.append(options.fingerprint())
    assert len(set(fingerprints)) == 1


@pytest.fixture
def native_manifest(manifest, case, monkeypatch):
    """Run real preparation with public fixture evidence and no provider calls."""
    root, config = manifest
    original_read = campaign.read_bundle

    def read(path):
        return {
            **original_read(path),
            "selection": case.selection.to_dict(),
            "fact_cards_sha256": digest(case.cards),
            "requirements": copy.deepcopy(case.packet["requirements"]),
            "methodology": {},
            "extractions": [],
        }

    monkeypatch.setattr(campaign, "read_bundle", read)
    config = campaign.profile_config(config, campaign.BUDGETED_PROFILE)
    target = root.with_name("wire-v4-offline")
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.BUDGETED_PROFILE,
        writing_plan=campaign.RANGE_WRITING_PLAN,
    )
    return target, config


def test_prepared_v4_manifest_and_all_exports_use_frozen_wire_tag(native_manifest):
    """One frozen field drives metadata without relabeling the old prompt."""
    from litagent.writing_native import CONTRACT
    from litagent.writing_v2 import PROMPT_CONTRACT

    root, config = native_manifest
    value = campaign.validate(root, config)
    assert value["writer_wire_contract"] == CONTRACT == "required-head-wire-v1"
    assert value["writer_prompt_contract"] == PROMPT_CONTRACT
    manifest_bytes = (root / "manifest.json").read_bytes()
    public = campaign.export(
        root,
        {"identity": value["identity"], "status": "offline_fixture", "cases": []},
        value,
    )
    assert (root / "manifest.json").read_bytes() == manifest_bytes
    assert public["writer_wire_contract"] == CONTRACT
    assert public["writing_profile"]["writer_wire_contract"] == CONTRACT
    out = Path("benchmarks/survey/results") / root.name
    with (out / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and all(
        row["writer_wire_contract"] == CONTRACT for row in rows
    )
    assert "Writer wire: " + CONTRACT in (out / "benchmark.md").read_text()
    html = (out / "benchmark.html").read_text()
    embedded = html.split('<script id="data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == public


def test_rehashed_missing_or_changed_wire_cannot_resume_old_v4(
    native_manifest, monkeypatch
):
    """Reject before recomputing paid-plan inputs and never rewrite a manifest."""
    import litagent.writing_v2 as writer

    root, config = native_manifest
    original = json.loads((root / "manifest.json").read_text())

    def forbidden(*args, **kwargs):
        raise AssertionError("Wire drift must fail before budget recomputation")

    monkeypatch.setattr(campaign, "expected_limits", forbidden)
    for wire in (None, "", "required-head-wire-v0", "unknown"):
        value = copy.deepcopy(original)
        if wire is None:
            value.pop("writer_wire_contract")
        else:
            value["writer_wire_contract"] = wire
        value["identity"] = campaign.digest(
            {key: item for key, item in value.items() if key != "identity"}
        )
        atomic_json(root / "manifest.json", value)
        before = (root / "manifest.json").read_bytes()
        with pytest.raises(ValueError, match="frozen_writer_wire_contract_conflict"):
            campaign.validate(root, config)
        assert (root / "manifest.json").read_bytes() == before
        assert not (root / "state.json").exists()
    atomic_json(root / "manifest.json", original)
    monkeypatch.setattr(writer, "NATIVE_WIRE_CONTRACT", "future-wire-test")
    with pytest.raises(ValueError, match="frozen_writer_wire_contract_conflict"):
        campaign.validate(root, config)


def test_old_profile_exports_do_not_infer_a_wire_contract(manifest):
    """An old manifest remains read-only and explicitly lacks the new metadata."""
    root, config = manifest
    value = campaign.validate(root, config)
    assert "writer_wire_contract" not in value
    before = (root / "manifest.json").read_bytes()
    public = campaign.export(
        root,
        {"identity": value["identity"], "status": "offline_fixture", "cases": []},
        value,
    )
    assert "writer_wire_contract" not in public
    assert "writer_wire_contract" not in public["writing_profile"]
    assert all("writer_wire_contract" not in row for row in public["cases"])
    assert (root / "manifest.json").read_bytes() == before


def test_non_v4_profile_cannot_claim_the_new_wire(manifest):
    """Even a valid outer digest cannot activate the codec in an older profile."""
    from litagent.writing_native import CONTRACT

    root, config = manifest
    value = json.loads((root / "manifest.json").read_text())
    value["writer_wire_contract"] = CONTRACT
    value["identity"] = campaign.digest(
        {key: item for key, item in value.items() if key != "identity"}
    )
    atomic_json(root / "manifest.json", value)
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="frozen_writer_wire_contract_conflict"):
        campaign.validate(root, config)
    assert (root / "manifest.json").read_bytes() == before


def test_source_freeze_inventory_includes_the_new_helper():
    """Untracked implementation helpers cannot disappear from a frozen run."""
    assert Path("src/litagent/writing_native.py") in campaign.source_paths()


def create_historical_parent(tmp_path):
    """Create a verified known-usage prefix without today's source inventory."""
    history = {
        "model_requests": 7,
        "model_known_tokens": 11,
        "model_unknown_conservative_tokens": 17,
    }
    root = tmp_path / "old-parent"
    manifest = {
        "version": campaign.VERSION,
        "history": history,
        "source_hashes": {"src/litagent/old_snapshot.py": "a" * 64},
    }
    manifest["identity"] = campaign.digest(manifest)
    row = {
        "run_id": "old-parent-case",
        "requests": 1,
        "known_tokens": 5,
        "unknown_hold": 0,
        "usage_complete": True,
    }
    state = {"identity": manifest["identity"], "status": "stopped", "cases": [row]}
    archive = {
        "run_id": row["run_id"],
        "events": [
            {"event": "llm.start", "data": {"operation_id": "old-operation"}},
            {
                "event": "llm.complete",
                "data": {
                    "operation_id": "old-operation",
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "usage_available": True,
                },
            },
        ],
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "state.json", state)
    archive_path = Path("artifacts/runs") / (row["run_id"] + ".json")
    atomic_json(archive_path, archive)
    return root, history, archive_path


def test_history_accounting_does_not_apply_new_source_inventory(tmp_path, monkeypatch):
    """Adding a helper neither rewrites old prefixes nor resets their spend."""
    monkeypatch.chdir(tmp_path)
    root, history, archive = create_historical_parent(tmp_path)
    paths = [root / "manifest.json", root / "state.json", archive]
    originals = {path: path.read_bytes() for path in paths}

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical accounting must not validate today's source")

    monkeypatch.setattr(campaign, "source_paths", forbidden)
    monkeypatch.setattr(campaign, "validate", forbidden)
    updated, receipts = campaign.accounting_history(history, [root])
    assert history == {
        "model_requests": 7,
        "model_known_tokens": 11,
        "model_unknown_conservative_tokens": 17,
    }
    assert updated == {**history, "model_requests": 8, "model_known_tokens": 16}
    assert (
        receipts[0]["archives"][str(archive)]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_history_compatibility_keeps_usage_binding_strict(tmp_path, monkeypatch):
    """A source-inventory change cannot waive mismatched historical usage."""
    monkeypatch.chdir(tmp_path)
    root, history, archive = create_historical_parent(tmp_path)
    value = json.loads(archive.read_text())
    value["events"][1]["data"]["total_tokens"] = 6
    atomic_json(archive, value)
    paths = [root / "manifest.json", root / "state.json", archive]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="frozen_accounting_archive_usage_invalid"):
        campaign.accounting_history(history, [root])
    assert all(path.read_bytes() == content for path, content in before.items())
