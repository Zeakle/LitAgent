"""Phase 14.0 release-baseline and runtime-contract tests."""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from litagent.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    LoggingConfig,
    MCPServerConfig,
    MemoryConfig,
)
from litagent.memory.working import WorkingMemory
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.runner import Infra, LitAgent

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    model: str = "model-a",
    redis_url: str = "redis://user:redis-secret@localhost:6380/0",
    pg_url: str = "postgresql://user:pg-secret@localhost:5433/litagent",
    mcp_secret: str = "mcp-secret-a",
) -> AppConfig:
    return AppConfig(
        agent=AgentConfig(),
        logging=LoggingConfig(),
        llm=LLMConfig(model=model),
        memory=MemoryConfig(redis_url=redis_url, pg_url=pg_url),
        mcp_servers={
            "papers": MCPServerConfig(
                transport="streamable_http",
                url=(
                    f"https://user:{mcp_secret}@mcp.example.test/"
                    f"rpc?token={mcp_secret}"
                ),
                headers={"Authorization": f"Bearer {mcp_secret}"},
                env={"API_TOKEN": mcp_secret},
            )
        },
    )


def _report_payload() -> dict:
    return {
        "survey": "Evidence-grounded survey.",
        "metadata": {"query": "few-shot learning"},
        "review_history": [],
        "graph_data": {},
        "partial": False,
        "evaluation": {"faithfulness": {"score": 1.0, "passed": True}},
        "quality": {
            "status": "passed",
            "failed_metrics": [],
            "unverified_metrics": [],
        },
        "delivery": {"status": "ready", "publishable": True, "reason_codes": []},
    }


def test_config_summary_is_stable_and_excludes_connection_secrets():
    from litagent.contracts import build_config_summary

    first = build_config_summary(_config())
    second = build_config_summary(
        _config(
            redis_url="redis://other:changed@redis.internal:6380/1",
            pg_url="postgresql://other:changed@pg.internal:5433/other",
            mcp_secret="mcp-secret-b",
        )
    )

    # Credentials and deployment endpoints do not alter behavioral identity.
    assert first["fingerprint"] == second["fingerprint"]
    serialized = json.dumps(first, sort_keys=True)
    for secret in ("redis-secret", "pg-secret", "mcp-secret-a", "Bearer"):
        assert secret not in serialized


def test_config_summary_changes_when_behavioral_model_changes():
    from litagent.contracts import build_config_summary

    first = build_config_summary(_config(model="model-a"))
    second = build_config_summary(_config(model="model-b"))

    assert first["schema_version"] == 1
    assert first["fingerprint"].startswith("sha256:")
    assert first["fingerprint"] != second["fingerprint"]


def test_survey_result_contract_adds_stable_top_level_identity():
    from litagent.contracts import normalize_survey_result

    result = normalize_survey_result(
        _report_payload(),
        config_fingerprint="sha256:" + ("a" * 64),
    )

    assert set(result) == {
        "schema_version",
        "config_fingerprint",
        "survey",
        "metadata",
        "review_history",
        "graph_data",
        "partial",
        "evaluation",
        "quality",
        "delivery",
    }
    assert result["schema_version"] == 1
    assert result["config_fingerprint"] == "sha256:" + ("a" * 64)


def test_api_report_reuses_shared_survey_result_fields():
    from litagent.api import SurveyReport
    from litagent.contracts import SurveyResult

    shared_fields = set(SurveyResult.model_fields)
    assert shared_fields <= set(SurveyReport.model_fields)
    assert set(SurveyReport.model_fields) == shared_fields | {"task_id"}


def test_runner_report_metadata_uses_canonical_config_summary():
    from litagent.contracts import build_config_summary

    config = _config()
    agent = LitAgent(config)
    report = agent._extract_report(
        {
            "adversarial_review": {
                "final_draft": "Survey.",
                "total_rounds": 1,
                "final_score": 1.0,
                "accepted": True,
            }
        },
        "few-shot learning",
    )

    assert (
        report["metadata"]["config_summary"]
        == build_config_summary(config)["effective"]
    )


def test_artifact_and_report_share_config_fingerprint(tmp_path):
    from litagent.contracts import build_config_summary, normalize_survey_result

    summary = build_config_summary(_config())
    recorder = RunRecorder("phase14-run", "query", ArchiveRepository(tmp_path))
    recorder.set_config_summary(summary)
    report = normalize_survey_result(
        _report_payload(),
        config_fingerprint=summary["fingerprint"],
    )

    artifact = recorder.finalize(report)

    assert artifact["config_fingerprint"] == report["config_fingerprint"]
    assert artifact["config"]["fingerprint"] == report["config_fingerprint"]
    assert artifact["status"] == "completed"


def test_artifact_rejects_mismatched_report_fingerprint(tmp_path):
    from litagent.contracts import build_config_summary, normalize_survey_result

    summary = build_config_summary(_config())
    recorder = RunRecorder("phase14-mismatch", "query", ArchiveRepository(tmp_path))
    recorder.set_config_summary(summary)
    report = normalize_survey_result(
        _report_payload(),
        config_fingerprint="sha256:" + ("b" * 64),
    )

    artifact = recorder.finalize(report)

    assert artifact["status"] == "failed"
    assert artifact["error"] == "config_fingerprint_mismatch"


@pytest.mark.asyncio
async def test_infra_closes_each_owned_resource_once_and_is_idempotent():
    working = MagicMock()
    working.close = AsyncMock()
    qdrant = MagicMock()
    qdrant.close = AsyncMock()
    pg_pool = MagicMock()
    pg_pool.close = AsyncMock()
    infra = Infra(
        _working_memory=working,
        _qdrant_client=qdrant,
        _pg_pool=pg_pool,
    )

    await infra.close()
    await infra.close()

    working.close.assert_awaited_once()
    qdrant.close.assert_awaited_once()
    pg_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_wiring_failure_closes_already_connected_infra():
    config = _config()
    agent = LitAgent(config)
    working = MagicMock()
    working.close = AsyncMock()
    infra = Infra(_working_memory=working)

    with (
        patch("litagent.runner.OpenAICompatibleClient", return_value=MagicMock()),
        patch.object(agent, "_connect_infra", new=AsyncMock(return_value=infra)),
        patch("litagent.runner.SkillManager", side_effect=RuntimeError("late failure")),
    ):
        with pytest.raises(RuntimeError, match="late failure"):
            await agent._wire()

    working.close.assert_awaited_once()
    assert agent._wired is False


@pytest.mark.asyncio
async def test_postgres_initialization_failure_closes_created_pool():
    config = _config()
    agent = LitAgent(config)
    pool = MagicMock()
    pool.close = AsyncMock()

    with (
        patch(
            "litagent.runner.WorkingMemory.connect",
            new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
        ),
        patch(
            "qdrant_client.AsyncQdrantClient",
            side_effect=RuntimeError("qdrant unavailable"),
        ),
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
        patch(
            "litagent.runner.ProceduralMemory.ensure_tables",
            new=AsyncMock(side_effect=RuntimeError("schema failed")),
        ),
    ):
        infra = await agent._connect_infra(config)

    pool.close.assert_awaited_once()
    assert infra._pg_pool is None


@pytest.mark.asyncio
async def test_working_memory_connect_failure_closes_created_client():
    config = _config()
    redis = MagicMock()
    redis.ping = AsyncMock(side_effect=RuntimeError("ping failed"))
    redis.aclose = AsyncMock()

    with patch("litagent.memory.working.Redis.from_url", return_value=redis):
        with pytest.raises(RuntimeError, match="ping failed"):
            await WorkingMemory.connect(config.memory)

    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_working_memory_rollback_preserves_original_connect_error():
    config = _config()
    redis = MagicMock()
    redis.ping = AsyncMock(side_effect=RuntimeError("ping failed"))
    redis.aclose = AsyncMock(side_effect=RuntimeError("close failed"))

    with patch("litagent.memory.working.Redis.from_url", return_value=redis):
        with pytest.raises(RuntimeError, match="ping failed"):
            await WorkingMemory.connect(config.memory)

    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_wiring_cancellation_closes_infra_created_before_transfer():
    config = _config()
    agent = LitAgent(config)
    working = MagicMock()
    working.close = AsyncMock()

    async def cancel_connect(_config, *, infra):
        infra._working_memory = working
        raise asyncio.CancelledError

    with (
        patch("litagent.runner.OpenAICompatibleClient", return_value=MagicMock()),
        patch.object(agent, "_connect_infra", side_effect=cancel_connect),
    ):
        with pytest.raises(asyncio.CancelledError):
            await agent._wire()

    working.close.assert_awaited_once()
    assert agent._wired is False


def test_failed_artifact_closes_unfinished_execution_nodes(tmp_path):
    recorder = RunRecorder("phase14-failed", "query", ArchiveRepository(tmp_path))
    recorder(
        "llm.start",
        {
            "operation_id": "llm-pending",
            "task_id": "synthesis",
            "messages": [{"role": "user", "content": "query"}],
        },
    )

    artifact = recorder.finalize(error="run_failed")

    node = artifact["nodes"]["llm:llm-pending"]
    assert artifact["status"] == "failed"
    assert node["status"] == "cancelled"
    assert node["error"] == "terminal_event_missing"


@pytest.mark.asyncio
async def test_mcp_partial_connect_failure_disconnects_bridge():
    config = _config()
    agent = LitAgent(config)
    bridge = MagicMock()
    bridge.connect_all = AsyncMock(side_effect=RuntimeError("partial MCP failure"))
    bridge.disconnect_all = AsyncMock()

    with (
        patch("litagent.runner.OpenAICompatibleClient", return_value=MagicMock()),
        patch.object(agent, "_connect_infra", new=AsyncMock(return_value=Infra())),
        patch.object(agent, "_create_reranker_async", new=AsyncMock(return_value=None)),
        patch("litagent.runner.MCPBridge", return_value=bridge),
    ):
        await agent._wire()

    bridge.disconnect_all.assert_awaited_once()
    assert agent._mcp_bridge is None
    await agent.cleanup()


def test_flow_demo_assets_are_loaded_from_the_installed_package():
    import litagent
    from litagent.api import FLOW_DEMO_STATIC, FLOW_DEMO_STATIC_ROOT

    package_root = Path(litagent.__file__).resolve().parent
    assert FLOW_DEMO_STATIC_ROOT == package_root / "static"
    assert FLOW_DEMO_STATIC == package_root / "static" / "flow-demo" / "index.html"
    assert FLOW_DEMO_STATIC.is_file()
    assert (FLOW_DEMO_STATIC.parent / "app.js").is_file()
    assert (FLOW_DEMO_STATIC.parent / "styles.css").is_file()


def test_distribution_declares_runtime_dependency_and_package_data():
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = pyproject["project"]["dependencies"]
    assert any(item.startswith("python-dotenv") for item in dependencies)
    package_data = pyproject["tool"]["setuptools"]["package-data"]["litagent"]
    assert "skills/*/SKILL.md" in package_data
    assert "static/flow-demo/*" in package_data


def test_compose_and_gitignore_define_durable_local_storage():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    qdrant_volumes = compose["services"]["qdrant"]["volumes"]

    assert "qdrant_data:/qdrant/storage" in qdrant_volumes
    assert "qdrant_data" in compose["volumes"]
    assert "pgdata" in compose["volumes"]

    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for path in (
        "artifacts/corpus/raw/",
        "artifacts/corpus/parsed/",
        "artifacts/corpus/quarantine/",
        "artifacts/qdrant-migration/",
        "artifacts/benchmarks/runtime/",
    ):
        assert path in ignored
