"""Verify workbench API boundaries without executing a live survey."""

from fastapi.testclient import TestClient

from litagent.api import app
from litagent.workbench import DecisionReceipt, ExportResult

client = TestClient(app)


class FakeService:
    def __init__(self):
        self.decisions = []

    async def start(self, query, options):
        assert query == "few-shot vision"
        assert options.report_language == "zh-CN"
        return "interactive-1"

    async def get(self, run_id):
        if run_id != "interactive-1":
            raise KeyError(run_id)
        return {
            "run_id": run_id,
            "query": "few-shot vision",
            "status": "awaiting_human",
            "review": {
                "request_id": "request-1",
                "stage": "paper_selection",
                "revision": 1,
                "allowed_actions": ["confirm", "cancel"],
                "candidates": [],
            },
            "result": None,
            "error": None,
        }

    async def submit_decision(self, run_id, decision):
        self.decisions.append(decision)
        return DecisionReceipt(
            request_id=decision.request_id,
            run_id=run_id,
            revision=decision.revision,
            action=decision.action,
            execution_status="queued",
        )

    async def export(self, run_id, revision, format):
        return ExportResult(
            filename="report.md",
            media_type="text/markdown; charset=utf-8",
            content=b"# Report",
            report_sha256="a" * 64,
        )

    async def versions(self, run_id):
        if run_id != "interactive-1":
            raise KeyError(run_id)
        return []

    async def events(self, run_id, *, after=0):
        events = [
            {"seq": 1, "event": "worker.start", "data": {"task_id": "one"}},
            {"seq": 2, "event": "worker.complete", "data": {"task_id": "one"}},
        ]
        return [event for event in events if event["seq"] > after]

    async def get_view(self, run_id):
        if run_id != "interactive-1":
            raise KeyError(run_id)
        return {
            "schema_version": 1,
            "source": "interactive",
            "summary": {"run_id": run_id, "query": "few-shot vision"},
            "graph": {"tasks": {}, "dependencies": {}, "nodes": {}},
            "timeline": [],
            "report": None,
            "evidence": [],
            "evaluation": {},
            "quality": None,
            "delivery": None,
            "review": None,
            "versions": [],
            "capabilities": {"can_review": False},
        }

    async def get_version(self, run_id, revision):
        if run_id != "interactive-1" or revision != 1:
            raise KeyError((run_id, revision))
        return {
            "run_id": run_id,
            "revision": revision,
            "content_sha256": "a" * 64,
            "report_sha256": "a" * 64,
            "report": {"survey": "# Report"},
        }


def headers():
    return {"X-LitAgent-Operation-Token": app.state.operation_token}


def test_interactive_create_requires_local_operation_token(monkeypatch):
    monkeypatch.setattr(app.state, "interactive_service", FakeService())
    payload = {
        "query": "few-shot vision",
        "report_language": "zh-CN",
        "human_review": True,
    }
    assert client.post("/survey", json=payload).status_code == 403
    response = client.post("/survey", json=payload, headers=headers())
    assert response.status_code == 202
    assert response.json()["task_id"] == "interactive-1"


def test_review_decision_and_export_use_typed_contracts(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(app.state, "interactive_service", service)
    review = client.get("/survey/interactive-1/review")
    assert review.status_code == 200
    assert review.json()["pending"]["stage"] == "paper_selection"

    response = client.post(
        "/survey/interactive-1/review/decision",
        json={
            "request_id": "request-1",
            "revision": 1,
            "action": "confirm",
            "selected_paper_ids": ["p1"],
        },
        headers=headers(),
    )
    assert response.status_code == 202
    assert service.decisions[0].selected_paper_ids == ("p1",)

    exported = client.get("/survey/interactive-1/export?revision=1&format=md")
    assert exported.status_code == 200
    assert exported.content == b"# Report"
    assert exported.headers["x-report-sha256"] == "a" * 64


def test_interactive_event_stream_resumes_after_last_event_id(monkeypatch):
    monkeypatch.setattr(app.state, "interactive_service", FakeService())

    response = client.get(
        "/survey/interactive-1/events", headers={"Last-Event-ID": "1"}
    )

    assert response.status_code == 200
    assert "id: 1" not in response.text
    assert "id: 2" in response.text
    assert "event: state" in response.text


def test_interactive_view_and_version_preview_are_read_only(monkeypatch):
    monkeypatch.setattr(app.state, "interactive_service", FakeService())

    view = client.get("/survey/interactive-1/view")
    version = client.get("/survey/interactive-1/versions/1")
    missing = client.get("/survey/interactive-1/versions/99")

    assert view.status_code == 200
    assert view.json()["source"] == "interactive"
    assert version.status_code == 200
    assert version.json()["report"]["survey"] == "# Report"
    assert missing.status_code == 404
