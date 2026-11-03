"""Keep run-budget presentation safe, explicit, and independent of report scores."""

import copy
import hashlib
import json
from types import MappingProxyType

import pytest

from litagent.config import AgentConfig, AppConfig, LoggingConfig, ObservabilityConfig
from litagent.demo import _snapshot_budget_state, project_flow_demo_view
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.observability.recorder import ArchiveRepository
from litagent.workbench import RunOptions

FIELDS = {
    "plan_identity",
    "known_tokens",
    "requests",
    "usage_uncertain",
    "stop_reason",
    "unknown_hold",
}


@pytest.fixture
def budget_state():
    """Retain the real six-field producer shape with portable 038 amounts."""
    return {
        "plan_identity": "a" * 64,
        "known_tokens": 218506,
        "requests": 14,
        "usage_uncertain": True,
        "stop_reason": "strict_usage_unknown",
        "unknown_hold": 78205,
    }


@pytest.fixture
def archive(budget_state):
    """Provide an unverified readable report without a private archive dependency."""
    survey = "# Retained report\n\nScientific verification remains incomplete."
    return {
        "version": 1,
        "run_id": "budget-run",
        "query": "portable budget question",
        "status": "failed",
        "events": [],
        "nodes": {},
        "graph": {},
        "usage": {"total_tokens": 218506},
        "budget_state": budget_state,
        "budget_plan": {"contract": "planned-budget-v7", "identity": "a" * 64},
        "report": {
            "survey": survey,
            "quality": {"status": "unverified"},
            "delivery": {
                "status": "blocked",
                "publishable": False,
                "reason_codes": ["quality_faithfulness_unverified"],
            },
            "evaluation": {
                "faithfulness": {
                    "score": None,
                    "passed": False,
                    "skipped": True,
                    "details": {
                        "evaluation_version": "survey-evaluation-v3",
                        "skipped_reason": "evaluation_contract_invalid",
                        "source_report_sha256": hashlib.sha256(
                            survey.encode()
                        ).hexdigest(),
                    },
                }
            },
            "metadata": {
                "report_language": "en",
                "token_usage": {"total_tokens": 999999},
            },
        },
    }


def assert_unavailable(value, availability="invalid"):
    """Require an explicit unavailable snapshot with no invented accounting."""
    assert set(value) == FIELDS | {
        "contract",
        "scope",
        "run_id",
        "availability",
        "reason",
    }
    assert value["contract"] == "run-budget-state-v1" and value["scope"] == "run"
    assert value["availability"] == availability
    assert value["reason"] == (
        "not_recorded" if availability == "missing" else "invalid_record"
    )
    assert all(value[key] is None for key in FIELDS)


def test_available_snapshot_is_a_safe_copy_of_actual_producer_fields(budget_state):
    """Preserve unknown hold separately from known usage and sent requests."""
    before = copy.deepcopy(budget_state)
    value = _snapshot_budget_state(MappingProxyType(budget_state), run_id="budget-run")
    assert value == {
        "contract": "run-budget-state-v1",
        "scope": "run",
        "run_id": "budget-run",
        "availability": "available",
        "reason": None,
        **budget_state,
    }
    value["known_tokens"] = 1
    assert budget_state == before


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_missing_any_producer_field_is_not_zero_usage(budget_state, field):
    """Partial snapshots cannot silently claim complete or zero accounting."""
    budget_state.pop(field)
    assert_unavailable(_snapshot_budget_state(budget_state, run_id="budget-run"))


@pytest.mark.parametrize("value", [None, {}, [], "state", 0, True])
def test_absent_and_malformed_snapshots_are_distinct(value):
    """Only an absent snapshot is unrecorded; provided malformed values are invalid."""
    assert_unavailable(
        _snapshot_budget_state(value, run_id="budget-run"),
        "missing" if value is None else "invalid",
    )


@pytest.mark.parametrize("field", ["known_tokens", "requests", "unknown_hold"])
@pytest.mark.parametrize(
    "value",
    [True, False, -1, 1.5, "1", None, float("nan"), float("inf"), 2**53],
)
def test_numeric_fields_reject_coercion_and_unsafe_browser_values(
    budget_state, field, value
):
    """Reject bools, non-integers, negative amounts, and browser precision loss."""
    budget_state[field] = value
    before = json.dumps(budget_state, sort_keys=True)
    assert_unavailable(_snapshot_budget_state(budget_state, run_id="budget-run"))
    assert json.dumps(budget_state, sort_keys=True) == before


@pytest.mark.parametrize("field", ["known_tokens", "requests", "unknown_hold"])
@pytest.mark.parametrize("value", [0, 2**53 - 1])
def test_numeric_fields_preserve_recorded_safe_integer_boundaries(
    budget_state, field, value
):
    """Present genuine zero or safe maximum values without converting their type."""
    budget_state[field] = value
    result = _snapshot_budget_state(budget_state, run_id="budget-run")
    assert result["availability"] == "available" and result[field] == value
    assert type(result[field]) is int


@pytest.mark.parametrize("value", [0, 1, None, "false", [], {}])
def test_uncertainty_requires_a_real_boolean(budget_state, value):
    """An integer or truthy container must not imply complete accounting."""
    budget_state["usage_uncertain"] = value
    assert_unavailable(_snapshot_budget_state(budget_state, run_id="budget-run"))


@pytest.mark.parametrize(
    "value", [None, "A" * 64, "a" * 63, "a" * 65, "g" * 64, True, [], {}]
)
def test_plan_identity_requires_exact_lowercase_sha(budget_state, value):
    """Reject invalid identities without failing the report projection."""
    budget_state["plan_identity"] = value
    assert_unavailable(_snapshot_budget_state(budget_state, run_id="budget-run"))


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "strict_usage_unknown",
        "strict_worker_failed",
        "planned_usage_bound_violation",
        "planned_ledger_persistence_failed",
    ],
)
def test_only_real_producer_stop_codes_are_preserved(budget_state, reason):
    """Keep each existing machine code without interpreting it as a science score."""
    budget_state["stop_reason"] = reason
    value = _snapshot_budget_state(budget_state, run_id="budget-run")
    assert value["availability"] == "available" and value["stop_reason"] == reason


@pytest.mark.parametrize("reason", ["private-provider-error", "", 0, True, [], {}])
def test_arbitrary_stop_values_are_not_exposed(budget_state, reason):
    """Return only a fixed invalid code instead of unsafe provider text."""
    budget_state["stop_reason"] = reason
    value = _snapshot_budget_state(budget_state, run_id="budget-run")
    assert_unavailable(value)
    assert "private-provider-error" not in json.dumps(value)


@pytest.mark.parametrize("run_id", [None, "", " ", " budget-run", True, 1, []])
def test_recorded_snapshot_requires_a_real_run_owner(budget_state, run_id):
    """A recorded amount cannot be reused as an ownerless run snapshot."""
    value = _snapshot_budget_state(budget_state, run_id=run_id)
    assert_unavailable(value)
    assert value["run_id"] is None


@pytest.mark.parametrize("version", range(2, 8))
def test_recorded_supported_plan_conflict_is_invalid(budget_state, version):
    """An accounting snapshot cannot borrow a different displayed plan identity."""
    assert_unavailable(
        _snapshot_budget_state(
            budget_state,
            run_id="budget-run",
            budget_plan={
                "contract": f"planned-budget-v{version}",
                "identity": "b" * 64,
            },
        )
    )


@pytest.mark.parametrize(
    "plan",
    [
        None,
        {},
        {"contract": "legacy", "identity": "b" * 64},
        {"contract": "planned-budget-v7"},
    ],
)
def test_safe_snapshot_does_not_require_a_comparable_plan(budget_state, plan):
    """An absent initial plan does not erase a genuine run accounting record."""
    assert (
        _snapshot_budget_state(budget_state, run_id="budget-run", budget_plan=plan)[
            "availability"
        ]
        == "available"
    )


def test_unknown_hold_cannot_claim_certain_usage(budget_state):
    """Preserve uncertainty even when no unknown amount was recorded yet."""
    budget_state["usage_uncertain"] = False
    assert_unavailable(_snapshot_budget_state(budget_state, run_id="budget-run"))
    budget_state.update(usage_uncertain=True, unknown_hold=0)
    value = _snapshot_budget_state(budget_state, run_id="budget-run")
    assert value["availability"] == "available" and value["usage_uncertain"] is True
    budget_state.update(usage_uncertain=False, stop_reason=None)
    value = _snapshot_budget_state(budget_state, run_id="budget-run")
    assert value["availability"] == "available" and value["unknown_hold"] == 0


@pytest.mark.parametrize("extra", ["secret", "messages", "entries"])
def test_additional_raw_fields_never_escape_the_projection(archive, extra):
    """Keep provider payloads outside the new browser budget contract."""
    archive["budget_state"][extra] = "private-budget-secret"
    before = copy.deepcopy(archive)
    view = project_flow_demo_view(archive, source="archive")
    assert_unavailable(view["budget_state"])
    assert "private-budget-secret" not in json.dumps(view)
    assert archive == before
    assert view["report"]["survey"] == archive["report"]["survey"]


def test_archive_projection_preserves_038_semantics_and_original_state(archive):
    """Adding a run budget must not add unknown hold to known usage or scores."""
    before = copy.deepcopy(archive)
    view = project_flow_demo_view(archive, source="archive")
    assert (
        view["budget_state"]["known_tokens"]
        == view["summary"]["total_tokens"]
        == 218506
    )
    assert view["budget_state"]["unknown_hold"] == 78205
    assert view["budget_state"]["requests"] == 14
    assert view["budget_state"]["usage_uncertain"] is True
    assert view["budget_state"]["stop_reason"] == "strict_usage_unknown"
    assert view["report"]["survey"] == archive["report"]["survey"]
    assert view["quality"] == archive["report"]["quality"]
    assert view["delivery"] == archive["report"]["delivery"]
    assert view["evaluation"] == archive["report"]["evaluation"]
    assert archive == before


def test_legacy_projection_does_not_infer_zero_from_known_tokens(archive):
    """A readable old report can have known tokens without a budget snapshot."""
    archive.pop("budget_state")
    view = project_flow_demo_view(archive, source="archive")
    assert_unavailable(view["budget_state"], "missing")
    assert view["summary"]["total_tokens"] == 218506
    assert view["report"]["survey"] == archive["report"]["survey"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["available", "missing", "invalid"])
async def test_interactive_get_view_bridges_actual_archive_projection(
    tmp_path, archive, mode
):
    """Use real SQLite and archive repositories without starting any agent."""
    if mode == "missing":
        archive.pop("budget_state")
    elif mode == "invalid":
        archive["budget_state"]["secret"] = "private-budget-secret"
    repository = SqliteReviewRepository(tmp_path / "interactive")
    await repository.initialize()
    await repository.create_run("budget-run", "question", RunOptions())
    await repository.save_report_version("budget-run", 1, archive["report"])
    await repository.mark_terminal("budget-run", "failed", result=archive["report"])
    archives = ArchiveRepository(tmp_path / "archives")
    path = archives.save(archive)
    before = path.read_bytes()

    def forbidden_agent(*args):
        """Fail if a read-only projection tries to initialize runtime work."""
        raise AssertionError("No agent is permitted during get_view")

    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
        agent_factory=forbidden_agent,
        archive_repository=archives,
    )
    expected = project_flow_demo_view(archive, source="archive")["budget_state"]
    for style in ("numbered", "apa7"):
        view = await service.get_view("budget-run", citation_style=style)
        assert view["budget_state"] == expected
        assert view["budget_state"]["availability"] == mode
        assert view["report"] == archive["report"]
        assert view["quality"] == archive["report"]["quality"]
        assert view["delivery"] == archive["report"]["delivery"]
        assert view["evaluation"] == archive["report"]["evaluation"]
        assert "private-budget-secret" not in json.dumps(view)
    version = await service.get_version("budget-run", 1)
    assert "budget_state" not in version
    assert version["report"] == archive["report"]
    assert (
        version["content_sha256"]
        == hashlib.sha256(archive["report"]["survey"].encode()).hexdigest()
    )
    assert path.read_bytes() == before and not service._tasks


@pytest.mark.asyncio
async def test_interactive_without_archive_does_not_guess_snapshot(tmp_path, archive):
    """Existing report metadata and token counts are not budget evidence."""
    repository = SqliteReviewRepository(tmp_path / "interactive")
    await repository.initialize()
    await repository.create_run("budget-run", "question", RunOptions())
    await repository.mark_terminal("budget-run", "failed", result=archive["report"])
    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
        archive_repository=ArchiveRepository(tmp_path / "empty"),
    )
    view = await service.get_view("budget-run")
    assert view["budget_state"] is None
    assert view["summary"]["total_tokens"] == 999999
    assert view["report"] == archive["report"] and not service._tasks
