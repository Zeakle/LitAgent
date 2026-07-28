"""Tests for survey API models, endpoints, and delivery state."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from litagent.api import app, SurveyRequest, SurveyStatus, SurveyReport
from litagent.observability.recorder import ArchiveRepository

client = TestClient(app)


class TestSurveyRequest:
    """Tests survey request validation."""

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
        """Whitespace-only queries retain the current model semantics."""
        req = SurveyRequest(query="   ")
        assert req.query == "   "

    def test_query_too_long(self):
        with pytest.raises(Exception):
            SurveyRequest(query="A" * 2001)


class TestSurveyStatus:
    """Tests survey status serialization."""

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
    """Tests survey report serialization."""

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
        """Missing evaluation data defaults to an empty mapping."""
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
        """The report schema exposes evaluation data."""
        assert "evaluation" in SurveyReport.model_fields


class TestHealth:
    """Tests the health endpoint."""

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestFlowDemo:
    """Tests live and archived flow-demo endpoints."""

    def test_serves_replay_page(self):
        response = client.get("/flow-demo")

        assert response.status_code == 200
        assert "Actual DAG" in response.text

    def test_starts_fixed_live_demo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app.state, "flow_repository", ArchiveRepository(tmp_path))
        with patch("litagent.api._run_flow_demo", new_callable=AsyncMock):
            response = client.post("/flow-demo/runs")
        assert response.status_code == 201
        assert response.json()["query"] == "few-shot learning in computer vision"

    def test_reads_persisted_archive_and_downloads_it(self, tmp_path, monkeypatch):
        repository = ArchiveRepository(tmp_path)
        repository.save(
            {
                "run_id": "flow-1",
                "query": "few-shot",
                "completed_at": "2026-07-18T00:00:00+00:00",
            }
        )
        monkeypatch.setattr(app.state, "flow_repository", repository)
        monkeypatch.setattr(app.state, "flow_runs", {})
        monkeypatch.setattr(app.state, "flow_archive_index", {})

        assert client.get("/flow-demo/runs").json()[0]["run_id"] == "flow-1"
        assert client.get("/flow-demo/runs/flow-1").json()["query"] == "few-shot"
        download = client.get("/flow-demo/runs/flow-1/download")
        assert download.status_code == 200
        assert download.json()["run_id"] == "flow-1"


class TestCreateSurvey:
    """Tests survey creation."""

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
    """Tests survey status retrieval."""

    def test_not_found(self):
        resp = client.get("/survey/nonexistent")
        assert resp.status_code == 404

    def test_running_status(self):
        app.state.tasks["test123"] = {
            "status": "running",
            "progress": "search",
            "error": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/test123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["progress"] == "search"

    def test_completed_status(self):
        app.state.tasks["done1"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/done1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


class TestGetSurveyReport:
    """Tests survey report retrieval."""

    def test_not_found(self):
        resp = client.get("/survey/nonexistent/report")
        assert resp.status_code == 404

    def test_still_running_409(self):
        app.state.tasks["running1"] = {
            "status": "running",
            "progress": "synthesis",
            "result": None,
            "error": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/running1/report")
        assert resp.status_code == 409

    def test_failed_500(self):
        app.state.tasks["failed1"] = {
            "status": "failed",
            "progress": "",
            "result": None,
            "error": "LLM timeout",
            "_created_at": 0,
        }
        resp = client.get("/survey/failed1/report")
        assert resp.status_code == 500

    def test_completed_report(self):
        app.state.tasks["completed1"] = {
            "status": "completed",
            "progress": "done",
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
        """Completed reports include evaluation data."""
        app.state.tasks["eval1"] = {
            "status": "completed",
            "progress": "done",
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
        """Legacy reports default missing evaluation data."""
        app.state.tasks["old1"] = {
            "status": "completed",
            "progress": "done",
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


class TestTTLCleanup:
    """Tests expiration cleanup for in-memory survey tasks."""

    def test_expired_tasks_cleaned(self):
        import time

        app.state.tasks["old_done"] = {
            "status": "completed",
            "progress": "done",
            "result": {},
            "error": None,
            "_created_at": time.time() - 7200,
        }
        app.state.tasks["old_failed"] = {
            "status": "failed",
            "progress": "",
            "result": None,
            "error": "boom",
            "_created_at": time.time() - 5000,
        }

        app.state.tasks["recent"] = {
            "status": "running",
            "progress": "plan",
            "result": None,
            "error": None,
            "_created_at": time.time(),
        }

        from litagent.api import _TASK_TTL_SECONDS

        now = time.time()
        expired = [
            tid
            for tid, t in app.state.tasks.items()
            if t["status"] in ("completed", "failed")
            and (now - t.get("_created_at", 0)) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            del app.state.tasks[tid]

        assert "old_done" not in app.state.tasks
        assert "old_failed" not in app.state.tasks
        assert "recent" in app.state.tasks


class TestQualityAPI:
    """Tests quality data in API responses."""

    def test_api_returns_quality_for_new_report(self):
        app.state.tasks["q1"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": {
                "survey": "text",
                "metadata": {},
                "review_history": [],
                "graph_data": {},
                "partial": False,
                "quality": {
                    "status": "failed",
                    "failed_metrics": ["citation_accuracy"],
                    "unverified_metrics": [],
                },
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/q1/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quality"]["status"] == "failed"

    def test_api_defaults_quality_for_legacy_report(self):
        app.state.tasks["old2"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": {
                "survey": "old",
                "metadata": {},
                "review_history": [],
                "graph_data": {},
                "partial": False,
            },
            "_created_at": 0,
        }
        resp = client.get("/survey/old2/report")
        assert resp.status_code == 200
        assert resp.json()["quality"]["status"] == "unverified"


class TestDeliveryAPI:
    """Tests delivery state in API responses."""

    @staticmethod
    def _blocked_result():
        return {
            "survey": "untrusted draft",
            "metadata": {},
            "review_history": [],
            "graph_data": {},
            "partial": False,
            "quality": {
                "status": "failed",
                "failed_metrics": ["faithfulness"],
                "unverified_metrics": [],
            },
            "delivery": {
                "status": "blocked",
                "publishable": False,
                "reason_codes": ["quality_failed"],
            },
        }

    def test_status_carries_delivery_status_when_completed(self):
        app.state.tasks["d1"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": self._blocked_result(),
            "_created_at": 0,
        }
        resp = client.get("/survey/d1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["delivery_status"] == "blocked"

    def test_status_delivery_none_while_running(self):
        app.state.tasks["d2"] = {
            "status": "running",
            "progress": "search",
            "error": None,
            "result": None,
            "_created_at": 0,
        }
        resp = client.get("/survey/d2")
        assert resp.json()["delivery_status"] is None

    def test_report_carries_delivery_and_stays_readable(self):
        """Blocked reports remain readable with delivery diagnostics."""
        app.state.tasks["d3"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": self._blocked_result(),
            "_created_at": 0,
        }
        resp = client.get("/survey/d3/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["survey"] == "untrusted draft"
        assert body["delivery"]["status"] == "blocked"
        assert body["delivery"]["publishable"] is False

    def test_status_and_report_derive_consistently_for_legacy(self):
        """Legacy status and report endpoints derive delivery consistently."""
        legacy = {
            "survey": "old",
            "metadata": {},
            "review_history": [],
            "graph_data": {},
            "partial": False,
            "quality": {
                "status": "failed",
                "failed_metrics": ["citation_accuracy"],
                "unverified_metrics": [],
            },
        }
        app.state.tasks["d4"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": legacy,
            "_created_at": 0,
        }
        status_body = client.get("/survey/d4").json()
        report_body = client.get("/survey/d4/report").json()
        assert status_body["delivery_status"] == "blocked"
        assert report_body["delivery"]["status"] == "blocked"

    def test_unverified_legacy_maps_to_needs_review(self):
        app.state.tasks["d5"] = {
            "status": "completed",
            "progress": "done",
            "error": None,
            "result": {
                "survey": "s",
                "metadata": {},
                "review_history": [],
                "graph_data": {},
                "partial": False,
            },
            "_created_at": 0,
        }
        assert client.get("/survey/d5").json()["delivery_status"] == "needs_review"
