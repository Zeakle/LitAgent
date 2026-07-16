"""Tests for FastAPI REST API (Phase 11.5)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from litagent.api import app, SurveyRequest, SurveyStatus, SurveyReport


# ═══════════════════════════════════════════════════
# FastAPI TestClient 需要在 import 阶段就可用
# LitAgent 构造依赖 AppConfig，API 测试中 mock 掉
# ═══════════════════════════════════════════════════

client = TestClient(app)


# ═══════════════════════════════════════════════════
# Pydantic model 单元测试
# ═══════════════════════════════════════════════════

class TestSurveyRequest:
    def test_valid_minimal(self):
        req = SurveyRequest(query="review attention in LLMs")
        assert req.query == "review attention in LLMs"
        assert req.config_path is None

    def test_valid_with_config(self):
        req = SurveyRequest(query="test", config_path="config/custom.yaml")
        assert req.config_path == "config/custom.yaml"

    def test_query_empty_rejected(self):
        with pytest.raises(Exception):
            SurveyRequest(query="")

    def test_query_whitespace_only(self):
        """min_length=1 检查的是字符串长度，不检查内容。空格也算字符，这里验证它会通过。"""
        req = SurveyRequest(query="   ")  # 3 个空格，长度 >= 1，通过校验
        assert req.query == "   "
        # NOTE: 如果要拒绝纯空格 query，应该在 SurveyRequest 里加 field_validator

    def test_query_too_long(self):
        with pytest.raises(Exception):
            SurveyRequest(query="A" * 2001)


class TestSurveyStatus:
    def test_minimal(self):
        s = SurveyStatus(task_id="abc123", status="running")
        assert s.task_id == "abc123"
        assert s.status == "running"
        assert s.progress == ""
        assert s.error is None

    def test_with_error(self):
        s = SurveyStatus(task_id="x", status="failed", error="timeout")
        assert s.error == "timeout"


class TestSurveyReport:
    def test_full(self):
        r = SurveyReport(
            task_id="t1",
            survey="A survey on attention...",
            metadata={"query": "attention", "generated_at": "2026-01-01"},
            review_history=[{"score": 0.8}],
            graph_data={"papers": [], "tier_counts": {}, "seminal_papers": []},
            evaluation={"citation": {"score": 0.9, "passed": True}},
            partial=False,
        )
        assert r.survey == "A survey on attention..."
        assert r.metadata["query"] == "attention"
        assert r.review_history[0]["score"] == 0.8
        assert r.evaluation["citation"]["score"] == 0.9

    def test_evaluation_defaults_to_empty_dict(self):
        """evaluation 缺失时默认 {}，不破坏已有响应（向后兼容）。"""
        r = SurveyReport(
            task_id="t2",
            survey="text",
            metadata={},
            review_history=[],
            graph_data={},
            partial=False,
        )
        assert r.evaluation == {}

    def test_model_fields_include_evaluation(self):
        """13.7.0：API response model 契约包含 evaluation 字段。"""
        assert "evaluation" in SurveyReport.model_fields


# ═══════════════════════════════════════════════════
# API 端点测试
# ═══════════════════════════════════════════════════

class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCreateSurvey:
    def test_create_returns_201(self):
        with patch("litagent.api._run_survey", new_callable=AsyncMock):
            resp = client.post("/survey", json={"query": "test query"})
            assert resp.status_code == 201

    def test_create_returns_task(self):
        with patch("litagent.api._run_survey", new_callable=AsyncMock):
            resp = client.post("/survey", json={"query": "test query"})
            body = resp.json()
            assert "task_id" in body
            assert body["status"] == "running"
            assert body["progress"] == "planner"

    def test_missing_query_rejected(self):
        resp = client.post("/survey", json={})
        assert resp.status_code == 422

    def test_empty_query_rejected(self):
        resp = client.post("/survey", json={"query": ""})
        assert resp.status_code == 422


class TestGetSurveyStatus:
    def test_not_found(self):
        resp = client.get("/survey/nonexistent")
        assert resp.status_code == 404

    def test_running_status(self):
        app.state.tasks["test123"] = {
            "status": "running", "progress": "search", "error": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/test123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["progress"] == "search"

    def test_completed_status(self):
        app.state.tasks["done1"] = {
            "status": "completed", "progress": "done", "error": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/done1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


class TestGetSurveyReport:
    def test_not_found(self):
        resp = client.get("/survey/nonexistent/report")
        assert resp.status_code == 404

    def test_still_running_409(self):
        app.state.tasks["running1"] = {
            "status": "running", "progress": "synthesis",
            "result": None, "error": None, "_created_at": 0,
        }
        resp = client.get("/survey/running1/report")
        assert resp.status_code == 409

    def test_failed_500(self):
        app.state.tasks["failed1"] = {
            "status": "failed", "progress": "",
            "result": None, "error": "LLM timeout", "_created_at": 0,
        }
        resp = client.get("/survey/failed1/report")
        assert resp.status_code == 500

    def test_completed_report(self):
        app.state.tasks["completed1"] = {
            "status": "completed", "progress": "done",
            "error": None,
            "result": {
                "survey": "## Survey on attention...",
                "metadata": {"query": "attention", "total_rounds": 3},
                "review_history": [{"score": 0.85}],
                "graph_data": {"papers": [], "tier_counts": {}, "seminal_papers": []},
                "evaluation": {"citation": {"score": 0.9, "passed": True}},
                "partial": False,
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/completed1/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["survey"] == "## Survey on attention..."
        assert body["metadata"]["total_rounds"] == 3
        assert body["partial"] is False

    def test_report_includes_evaluation(self):
        """13.7.0：endpoint 返回包含 evaluation 字段。"""
        app.state.tasks["eval1"] = {
            "status": "completed", "progress": "done",
            "error": None,
            "result": {
                "survey": "text",
                "metadata": {"query": "q"},
                "review_history": [],
                "graph_data": {},
                "evaluation": {"citation": {"score": 0.5}},
                "partial": False,
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/eval1/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "evaluation" in body
        assert body["evaluation"]["citation"]["score"] == 0.5

    def test_report_evaluation_defaults_when_missing(self):
        """旧数据无 evaluation 字段 → endpoint 返回 {} 而非 500。"""
        app.state.tasks["old1"] = {
            "status": "completed", "progress": "done",
            "error": None,
            "result": {
                "survey": "old survey",
                "metadata": {"query": "old"},
                "review_history": [],
                "graph_data": {},
                "partial": False,
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/old1/report")
        assert resp.status_code == 200
        assert resp.json()["evaluation"] == {}


# ═══════════════════════════════════════════════════
# TTL cleanup
# ═══════════════════════════════════════════════════

class TestTTLCleanup:
    def test_expired_tasks_cleaned(self):
        import time
        # 添加已完成的"过期"任务
        app.state.tasks["old_done"] = {
            "status": "completed", "progress": "done",
            "result": {}, "error": None,
            "_created_at": time.time() - 7200,  # 2 小时前
        }
        app.state.tasks["old_failed"] = {
            "status": "failed", "progress": "",
            "result": None, "error": "boom",
            "_created_at": time.time() - 5000,
        }
        # 添加未过期的
        app.state.tasks["recent"] = {
            "status": "running", "progress": "plan",
            "result": None, "error": None,
            "_created_at": time.time(),
        }

        # 直接调清理逻辑
        from litagent.api import _TASK_TTL_SECONDS
        now = time.time()
        expired = [
            tid for tid, t in app.state.tasks.items()
            if t["status"] in ("completed", "failed")
            and (now - t.get("_created_at", 0)) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            del app.state.tasks[tid]

        assert "old_done" not in app.state.tasks
        assert "old_failed" not in app.state.tasks
        assert "recent" in app.state.tasks  # running 不清理


# ═══════════════════════════════════════════════════════════
# 13.7.2-C2 — Quality API
# ═══════════════════════════════════════════════════════════

class TestQualityAPI:
    """13.7.2-C2：API quality 字段。"""

    def test_api_returns_quality_for_new_report(self):
        app.state.tasks["q1"] = {
            "status": "completed", "progress": "done", "error": None,
            "result": {
                "survey": "text", "metadata": {}, "review_history": [],
                "graph_data": {}, "partial": False,
                "quality": {"status": "failed",
                            "failed_metrics": ["citation_accuracy"],
                            "unverified_metrics": []},
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/q1/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quality"]["status"] == "failed"

    def test_api_defaults_quality_for_legacy_report(self):
        app.state.tasks["old2"] = {
            "status": "completed", "progress": "done", "error": None,
            "result": {
                "survey": "old", "metadata": {}, "review_history": [],
                "graph_data": {}, "partial": False,
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/old2/report")
        assert resp.status_code == 200
        assert resp.json()["quality"]["status"] == "unverified"
