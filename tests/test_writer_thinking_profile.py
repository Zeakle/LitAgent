"""Freeze explicit thinking campaigns without loosening the old disabled baseline."""

import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
from model_config_helpers import load_previous_default_config as load_config

from litagent.report_audit import atomic_json
from scripts import run_frozen_writing as campaign
from scripts.run_stage_model_quality import MODELS, ORIGIN, check_models

BASELINE = "disabled_v1"
THINKING = "writer_reviewer_thinking_v1"


@pytest.fixture
def configured():
    """Make provider and reservation assertions independent of a developer's env."""
    config = load_config()
    config.llm.base_url, config.llm.model = ORIGIN, MODELS["extraction"]
    for role, model in MODELS.items():
        setattr(config.llm, role + "_model", model)
    config.llm.max_tokens = 20000
    config.safety.max_cost_tokens = 500000
    for section in (config.planner, config.extractor, config.eval, config.adversarial):
        section.thinking_mode = "disabled"
    config.adversarial.review_max_tokens = 4096
    config.eval.max_tokens = 16384
    return config


@pytest.mark.parametrize(
    "version", [None, "grounded-writer-native-intent-v2", "unknown"]
)
def test_manifest_rejects_rehashed_prompt_version_drift(manifest, version):
    """Source preparation cannot omit or relabel the actual writer prompt."""
    from litagent.report_audit import digest
    from litagent.writing_v2 import PROMPT_CONTRACT

    root, config = manifest
    value = campaign.validate(root, config)
    assert value["writer_prompt_contract"] == PROMPT_CONTRACT
    if version is None:
        value.pop("writer_prompt_contract")
    else:
        value["writer_prompt_contract"] = version
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    atomic_json(root / "manifest.json", value)
    with pytest.raises(ValueError, match="identity_conflict"):
        campaign.validate(root, config)


@pytest.mark.parametrize("historical", [False, True])
def test_prompt_export_uses_frozen_metadata_and_never_infers_history(
    manifest, historical
):
    """All formats share the stored version; an absent historical value is unknown."""
    from litagent.writing_v2 import PROMPT_CONTRACT

    root, config = manifest
    value = campaign.validate(root, config)
    if historical:
        value.pop("writer_prompt_contract")
    state = {"identity": value["identity"], "status": "offline_fixture", "cases": []}
    public = campaign.export(root, state, value)
    expected = None if historical else PROMPT_CONTRACT
    assert public["writer_prompt_contract"] == expected
    out = Path("benchmarks/survey/results") / root.name
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert all(r["writer_prompt_contract"] == (expected or "") for r in rows)
    assert (
        "Writer prompt: " + (expected or "unavailable")
        in (out / "benchmark.md").read_text()
    )
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )


def test_old_model_gate_stays_disabled_and_new_profile_is_explicit(configured):
    """The new mode cannot silently alter the baseline or shared app config."""
    check_models(configured)
    before = configured.model_dump()
    thinking = campaign.profile_config(configured, THINKING)
    assert configured.model_dump() == before
    assert thinking.adversarial.thinking_mode == "enabled"
    with pytest.raises(ValueError, match="model_configuration_invalid"):
        check_models(thinking)
    campaign.check_profile(thinking, THINKING)
    with pytest.raises(ValueError, match="profile"):
        campaign.check_profile(thinking, BASELINE)
    assert campaign.profile_config(configured, BASELINE).model_dump() == before


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "mode",
        "evaluation",
        "planner",
        "model",
        "budget",
        "output",
        "review_output",
        "eval_output",
    ],
)
def test_profile_rejects_all_other_drift_without_correcting_the_caller(
    configured, fault
):
    """Only the named mode change is allowed; no wider budget or model role slips in."""
    config = campaign.profile_config(configured, THINKING)
    profile = THINKING
    if fault == "unknown":
        profile = "arbitrary-url-or-mode"
    if fault == "mode":
        config.adversarial.thinking_mode = "provider_default"
    if fault == "evaluation":
        config.eval.thinking_mode = "enabled"
    if fault == "planner":
        config.planner.thinking_mode = "enabled"
    if fault == "model":
        config.llm.generation_model = "different"
    if fault == "budget":
        config.safety.max_cost_tokens = 500001
    if fault == "output":
        config.llm.max_tokens = 20001
    if fault == "review_output":
        config.adversarial.review_max_tokens = 4097
    if fault == "eval_output":
        config.eval.max_tokens = 16385
    before = config.model_dump()
    with pytest.raises(ValueError):
        campaign.check_profile(config, profile)
    assert config.model_dump() == before


@pytest.fixture
def manifest(tmp_path, monkeypatch, configured):
    """Use real prepare/validate while substituting only the archive parser and Git."""
    monkeypatch.chdir(tmp_path)
    source = Path("source.py")
    source.write_text("# Frozen test source\n")
    monkeypatch.setattr(campaign, "source_paths", lambda: [source])
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **kw: "commit\n"
    )
    for _, run in campaign.CASES:
        atomic_json(Path("artifacts/runs") / (run + ".json"), {"run": run})

    def read(path):
        return {
            "query": "Offline public fixture question",
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection": {"selected_items": []},
        }

    monkeypatch.setattr(campaign, "read_bundle", read)
    atomic_json(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ),
        {
            "cumulative": {
                "model_requests": 749,
                "model_known_tokens": 4176901,
                "model_unknown_conservative_tokens": 1030414,
            }
        },
    )
    config = campaign.profile_config(configured, THINKING)
    root = tmp_path / "thinking-suite"
    campaign.prepare(root, config, writing_profile=THINKING)
    return root, config


def test_named_profile_is_bound_to_manifest_and_preserves_fixed_scope(manifest):
    """The explicit mode and unchanged three-case cap are part of one frozen identity."""
    root, config = manifest
    value = campaign.validate(root, config)
    assert value["writing_profile"] == THINKING
    assert value["config"]["effective"]["adversarial"]["thinking_mode"] == "enabled"
    assert value["limits"] == campaign.LIMITS
    assert [r["case_id"] for r in value["cases"]] == [c for c, _ in campaign.CASES]
    assert value["history"]["model_requests"] == 749
    assert value["history"]["model_unknown_conservative_tokens"] == 1030414
    assert not (root / "state.json").exists()
    with pytest.raises(ValueError, match="already_exists"):
        campaign.prepare(root, config, writing_profile=THINKING)


@pytest.mark.parametrize("profile", [None, BASELINE, "unreviewed"])
def test_changing_profile_cannot_reinterpret_enabled_manifest(manifest, profile):
    """Even a freshly recomputed digest cannot turn the mode into another contract."""
    root, config = manifest
    value = json.loads((root / "manifest.json").read_text())
    if profile is None:
        value.pop("writing_profile")
    else:
        value["writing_profile"] = profile
    value["identity"] = campaign.digest(
        {k: v for k, v in value.items() if k != "identity"}
    )
    atomic_json(root / "manifest.json", value)
    with pytest.raises(ValueError, match="profile"):
        campaign.validate(root, config)


def test_live_cli_loads_frozen_mode_without_changing_default_config(
    manifest, configured, monkeypatch, capsys
):
    """Live restores an already frozen mode, never an implicit new attempt."""
    root, _ = manifest
    before = configured.model_dump()
    monkeypatch.setattr(sys, "argv", ["campaign", "--root", str(root), "--live"])
    monkeypatch.setattr(campaign, "load_config", lambda: configured)

    async def execute(path, config):
        campaign.validate(path, config)
        assert config.adversarial.thinking_mode == "enabled"
        assert config.eval.thinking_mode == "disabled"
        return {"status": "completed", "cases": []}

    monkeypatch.setattr(campaign, "execute", execute)
    campaign.main()
    assert configured.model_dump() == before
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_live_cli_cannot_override_frozen_profile(tmp_path, monkeypatch):
    """Reject conflicting execution flags before loading config or any transport."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign", "--root", str(tmp_path), "--live", "--writing-profile", THINKING],
    )

    def forbidden():
        raise AssertionError("must not even load mutable config")

    monkeypatch.setattr(campaign, "load_config", forbidden)
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2


def test_public_exports_show_real_mode_and_keep_three_planned_cases(manifest):
    """All presentations use one mode descriptor, without leaking config or sources."""
    root, config = manifest
    frozen = campaign.validate(root, config)
    state = {"identity": frozen["identity"], "status": "prepared", "cases": []}
    before = copy.deepcopy(state)
    public = campaign.export(root, state, frozen)
    assert state == before
    descriptor = public["writing_profile"]
    assert descriptor["id"] == THINKING
    assert (
        descriptor["generation_thinking_mode"]
        == descriptor["reviewer_thinking_mode"]
        == "enabled"
    )
    assert descriptor["evaluation_thinking_mode"] == "disabled"
    assert descriptor["output_limit_parameter"] == "max_completion_tokens"
    assert (
        descriptor["generation_output_tokens"] == 20000
        and descriptor["reviewer_output_tokens"] == 4096
    )
    out = Path("benchmarks/survey/results") / root.name
    page = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            page.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    assert THINKING in (out / "benchmark.md").read_text()
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert len(rows) == 3 and all(r["writing_profile"] == THINKING for r in rows)
    assert all(
        r["reviewer_thinking_mode"] == "enabled"
        and r["generation_output_tokens"] == "20000"
        and r["output_limit_parameter"] == "max_completion_tokens"
        for r in rows
    )
    assert "writing_profile:d.writing_profile" in page
    assert public["planned"] == public["not_run"] == 3 and public["passed"] == 0
    assert "source_run_id" not in json.dumps(public) and "api_key" not in json.dumps(
        public
    )


def test_low_profile_binds_internal_parameters_without_changing_app_config(manifest):
    """A fixed internal parameter changes campaign identity, not old config defaults."""
    root, config = manifest
    low = "writer_reviewer_low_thinking_v1"
    original = campaign.validate(root, config)
    target = root.with_name("low-suite")
    effective = campaign.profile_config(config, low)
    assert campaign.profile_parameters(low) == {"native_reasoning_effort": "low"}
    campaign.prepare(target, effective, writing_profile=low)
    frozen = campaign.validate(target, effective)
    assert frozen["writing_parameters"] == {"native_reasoning_effort": "low"}
    assert frozen["config"] == original["config"]
    assert frozen["cases"] == original["cases"]
    assert frozen["identity"] != original["identity"]
    state = {"status": "prepared", "cases": []}
    public = campaign.export(target, state, frozen)
    assert public["writing_profile"]["reasoning_effort"] == "low"
    assert all(r["reasoning_effort"] == "low" for r in public["cases"])


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"native_reasoning_effort": "high"},
        {"native_reasoning_effort": "low", "budget": 1000000},
    ],
)
def test_low_profile_rejects_missing_or_changed_frozen_parameters(manifest, parameters):
    """Digest recomputation must not grant new server parameters or wider budgets."""
    root, config = manifest
    low = "writer_reviewer_low_thinking_v1"
    target = root.with_name("low-suite")
    effective = campaign.profile_config(config, low)
    campaign.prepare(target, effective, writing_profile=low)
    frozen = json.loads((target / "manifest.json").read_text())
    if parameters is None:
        frozen.pop("writing_parameters")
    else:
        frozen["writing_parameters"] = parameters
    frozen["identity"] = campaign.digest(
        {k: v for k, v in frozen.items() if k != "identity"}
    )
    atomic_json(target / "manifest.json", frozen)
    with pytest.raises(ValueError, match="parameters"):
        campaign.validate(target, effective)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_detail", [True, False])
async def test_driver_keeps_optional_reasoning_separate_from_primary_cost(
    manifest, monkeypatch, complete_detail
):
    """Exercise the actual driver, recorder and exporters at the external run boundary."""
    from litagent.safety.budget import CostBudget
    from litagent.writing_v2 import WriterError

    root, config = manifest
    profile = "writer_reviewer_low_thinking_v1"
    cfg = campaign.profile_config(config, profile)
    cfg.observability.enabled = False
    target = root.with_name("low-driver")
    campaign.prepare(target, cfg, writing_profile=profile)

    async def external_run(agent, bundle, run_id, *, native_reasoning_effort=None):
        assert native_reasoning_effort == "low"
        budget = CostBudget()
        budget.configure_strict(halt_on_unknown_usage=True)
        agent._cost_budget = budget
        for i in range(2):
            budget.begin_request()
            details = (
                {
                    "reasoning_tokens": 70 if i == 0 else 0,
                    "reasoning_usage_available": True,
                }
                if i == 0 or complete_detail
                else {}
            )
            usage = {
                "prompt_tokens": 20,
                "completion_tokens": 100,
                "usage_available": True,
                **details,
            }
            budget.record(usage)
            data = {
                "operation_id": str(i),
                "model": "pro",
                "messages": [],
                "max_tokens": 20000,
                "output_limit_parameter": "max_completion_tokens",
                "reasoning_effort": "low",
            }
            agent._trace_hook("llm.start", data)
            agent._trace_hook(
                "llm.complete",
                {
                    **data,
                    **usage,
                    "total_tokens": 120,
                    "content": "",
                    "finish_reason": "length",
                },
            )
        raise WriterError("writer_output_truncated")

    monkeypatch.setattr(campaign, "run_frozen", external_run)
    state = await campaign.execute(target, cfg)
    assert state["status"] == "stopped" and len(state["cases"]) == 1
    row = state["cases"][0]
    assert (
        row["usage_complete"] is True
        and row["known_tokens"] == 240
        and row["unknown_hold"] == 0
    )
    assert row["reasoning_usage_complete"] is complete_detail
    assert row["reasoning_tokens"] == (70 if complete_detail else None)
    assert row["reasoning_usage_available_requests"] == (2 if complete_detail else 1)
    out = Path("benchmarks/survey/results") / target.name
    public = json.loads((out / "benchmark.json").read_text())
    assert (
        public["new_usage"]["known_tokens"] == 240
        and public["passed"] == 0
        and public["not_run"] == 2
    )
    csv_rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert csv_rows[0]["reasoning_tokens"] == ("70" if complete_detail else "")
    assert "Details complete" in (out / "benchmark.md").read_text()
