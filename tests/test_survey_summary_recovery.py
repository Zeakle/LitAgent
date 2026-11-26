"""Exercise the summary-recovery retry boundary without sending model requests."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from litagent.benchmark.survey_metrics import evaluate_survey_artifact
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.safety.planned_budget import EnvelopeExceeded
from litagent.writing_v2 import WriterError, WriterSession
from scripts import run_survey_summary_recovery as recovery
from scripts.run_survey_summary_recovery import shorten_summary


class LocalLedger:
    """Record only the accounting fields read by the retry boundary."""

    def __init__(self):
        self.stop_reason = None
        self.usage_uncertain = False
        self.entries = {}

    def send(self, status="settled"):
        self.entries[str(len(self.entries))] = {"status": status, "sent": True}

    def snapshot(self):
        return {
            "requests_sent": len(self.entries),
            "call_count": sum(
                entry["status"] == "settled" for entry in self.entries.values()
            ),
            "usage_uncertain": self.usage_uncertain,
            "entries": copy.deepcopy(self.entries),
        }


@pytest.fixture
def frozen_case():
    return SimpleNamespace(
        body={"methods": [{"sentences": [{"text": "Frozen original."}]}]},
        summaries={"abstract": [{"sentences": [{"text": "finding " * 251}]}]},
    )


@pytest.mark.asyncio
async def test_summary_recovery_retries_two_settled_overflows_from_original(
    frozen_case,
):
    owner = LocalLedger()
    original = copy.deepcopy(frozen_case)
    shortened = {"abstract": [{"sentences": [{"text": "Shortened summary."}]}]}
    outcomes = iter(
        (
            WriterError("writer_summary_capacity_exceeded", "quality"),
            EnvelopeExceeded("summary", "summary_json_bytes", 6001, 6000),
            None,
        )
    )

    async def repair(body, summaries, call_name, *, shorter_target):
        assert body is frozen_case.body and summaries is frozen_case.summaries
        assert frozen_case == original
        assert call_name == "summary_initial" and shorter_target is True
        owner.send()
        if failure := next(outcomes):
            raise failure
        return shortened

    session = SimpleNamespace(
        repair_initial_summary_length=AsyncMock(side_effect=repair)
    )
    assert await shorten_summary(session, frozen_case, owner) is shortened
    assert session.repair_initial_summary_length.await_count == 3
    assert frozen_case == original


@pytest.mark.asyncio
async def test_summary_recovery_propagates_third_settled_overflow(frozen_case):
    owner = LocalLedger()

    async def repair(*args, **kwargs):
        owner.send()
        raise WriterError("writer_patch_capacity_exceeded", "quality")

    session = SimpleNamespace(
        repair_initial_summary_length=AsyncMock(side_effect=repair)
    )
    with pytest.raises(WriterError, match="writer_patch_capacity_exceeded"):
        await shorten_summary(session, frozen_case, owner)
    assert owner.snapshot()["requests_sent"] == 3
    assert session.repair_initial_summary_length.await_count == 3


@pytest.mark.parametrize(
    ("failure", "send_status", "uncertain", "stop_reason"),
    [
        (
            WriterError("writer_summary_capacity_exceeded", "quality"),
            "settled",
            True,
            None,
        ),
        (WriterError("writer_summary_capacity_exceeded", "quality"), None, False, None),
        (
            WriterError("writer_summary_capacity_exceeded", "quality"),
            "started_no_replay",
            False,
            None,
        ),
        (WriterError("writer_reference_invalid", "quality"), "settled", False, None),
        (
            EnvelopeExceeded("body", "body_json_bytes", 42001, 42000),
            "settled",
            False,
            None,
        ),
        (
            WriterError("writer_summary_capacity_exceeded", "quality"),
            "settled",
            False,
            "strict_usage_unknown",
        ),
    ],
    ids=[
        "unknown-usage",
        "no-send",
        "unsettled",
        "semantic-error",
        "body-overflow",
        "terminal-owner",
    ],
)
@pytest.mark.asyncio
async def test_summary_recovery_does_not_retry_outside_its_authority(
    frozen_case, failure, send_status, uncertain, stop_reason
):
    owner = LocalLedger()

    async def repair(*args, **kwargs):
        if send_status is not None:
            owner.send(send_status)
        owner.usage_uncertain = uncertain
        owner.stop_reason = stop_reason
        raise failure

    session = SimpleNamespace(
        repair_initial_summary_length=AsyncMock(side_effect=repair)
    )
    with pytest.raises(type(failure)) as caught:
        await shorten_summary(session, frozen_case, owner)
    assert caught.value is failure
    assert session.repair_initial_summary_length.await_count == 1


@pytest.mark.asyncio
async def test_recovery_installs_owner_only_after_runtime_wiring(monkeypatch, tmp_path):
    wired_budget = object()

    class Agent:
        def __init__(self, *args, **kwargs):
            self._cost_budget = None

        async def __aenter__(self):
            self._cost_budget = wired_budget
            return self

        async def __aexit__(self, *args):
            pass

        def benchmark_accounting(self):
            return {"known_tokens": 0, "requests": 0, "usage_uncertain": False}

    def reject_after_wiring(budget, parent, path, *, after_capacity_failure):
        assert budget is wired_budget
        assert not after_capacity_failure
        raise ValueError("test_stop_before_transport")

    config_summary = {"effective": {"llm": {"model_routing": {}}}}
    task = SimpleNamespace(status=SimpleNamespace(value="pending"))
    graph = SimpleNamespace(get_task=lambda task_id: task, get_results=dict)
    case = SimpleNamespace(
        archive={"config": config_summary},
        task_input={},
        packet={"query": "Frozen question?"},
        row={"run_id": "parent", "snapshot_sha256": "source-hash"},
    )
    recorder = Mock()
    monkeypatch.setattr(recovery, "LitAgent", Agent)
    monkeypatch.setattr(recovery, "task_graph", lambda artifact: graph)
    monkeypatch.setattr(recovery, "options_for", lambda config: None)
    monkeypatch.setattr(recovery, "build_config_summary", lambda *args: config_summary)
    monkeypatch.setattr(recovery, "restore_child_budget", reject_after_wiring)
    await recovery.continue_case(None, tmp_path, case, {}, recorder, "child", "input")
    recorder.finalize.assert_called_once_with(
        report=None, error="test_stop_before_transport", terminal_status="failed"
    )
    recorder.assert_any_call(
        "benchmark.accounting",
        {"known_tokens": 0, "requests": 0, "usage_uncertain": False},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["settled", "started_no_replay"])
async def test_rebuild_uses_compaction_only_after_a_settled_overflow(
    monkeypatch, frozen_case, status
):
    owner = LocalLedger()
    frozen_case.summaries["conclusion"] = [
        {"sentences": [{"text": text}]} for text in ("Opening", "Detail", "Limits")
    ]
    candidate = {"conclusion": []}

    async def regenerate(body, name):
        assert body is frozen_case.body and name == recovery.CHECKPOINT_SUMMARY_REBUILD
        owner.send(status)
        raise WriterError("writer_summary_capacity_exceeded", "quality")

    session = SimpleNamespace(summaries=regenerate, journal={}, save=Mock())
    compact = Mock(return_value=candidate)
    monkeypatch.setattr(recovery, "compact_frozen_conclusion", compact)
    if status == "settled":
        assert await recovery.rebuild_summary(session, frozen_case, owner) is candidate
        compact.assert_called_once_with(frozen_case, session)
        assert (
            session.journal["benchmark_summary_compaction"][
                "failed_regeneration_requests"
            ]
            == 1
        )
    else:
        with pytest.raises(WriterError, match="writer_summary_capacity_exceeded"):
            await recovery.rebuild_summary(session, frozen_case, owner)
        compact.assert_not_called()


def test_summary_checkpoint_archive_keeps_citation_evidence_without_new_work(tmp_path):
    paper = {"paper_id": "arxiv:1606.04080"}
    evidence = [{**paper, "evidence_items": [{"evidence_id": "e1"}]}]
    graph = TaskGraph()
    for task_id, agent_type, output in (
        ("recall_q0", "recall", [paper]),
        ("dedup", "dedup", [paper]),
        ("relevance_gate", "relevance_gate", [paper]),
        ("extract", "extractor", evidence),
    ):
        graph.add_task(SubTask(task_id, agent_type, agent_type, {"query": "Frozen"}))
        graph.mark_done(task_id, output)

    repository = ArchiveRepository(tmp_path)
    recorder = RunRecorder("child", "Frozen", repository)
    recorder.capture_worker_checkpoint(
        graph, source_run_id="parent", source_snapshot_sha256="a" * 64
    )
    recorder.finalize({"survey": "Grounded finding [E:e1]."})
    saved = repository.get("child")
    node = saved["nodes"]["worker:extract"]
    assert node["output"] == evidence
    assert node["input"] == {"query": "Frozen"}
    assert node["metadata"] == {"checkpoint_source_run_id": "parent"}
    assert all(
        node[key] is None
        for key in ("started_at", "completed_at", "elapsed_ms", "usage")
    )
    assert [event["event"] for event in saved["events"]] == [
        "benchmark.summary_checkpoint"
    ]

    case = SimpleNamespace(
        required_paper_ids=[paper["paper_id"]],
        relevant_paper_ids=[paper["paper_id"]],
        allowed_delivery_statuses=["ready"],
    )
    metrics = evaluate_survey_artifact(case=case, artifact=saved)
    assert metrics.paper_flow.recall == metrics.paper_flow.evidence == 1
    assert metrics.citations.recall == metrics.citations.precision == 1
    assert metrics.citations.unknown_evidence_ids == []
    assert not metrics.stage_latency_ms
    assert metrics.cost.worker_calls == metrics.cost.llm_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", [True, False])
async def test_checkpoint_summary_only_regenerates_after_known_capacity_failure(
    monkeypatch, settled
):
    from litagent.safety import planned_budget

    prefix = "summary_repair_2"
    oversized = {"abstract": [{"sentences": [{"text": "claim " * 251}]}]}
    compact = {"abstract": [{"sentences": [{"text": "Bounded summary."}]}]}
    body_attempts = [{"index": 1, "candidate_sha256": "frozen-body"}]
    owner = SimpleNamespace(
        entries={
            f"owner:{prefix}": {
                "status": "settled" if settled else "started_no_replay",
                "sent": True,
            }
        },
        stop_reason=None,
        snapshot=lambda: {"usage_uncertain": False},
    )
    monkeypatch.setattr(planned_budget, "planned_backend", lambda _: owner)
    session = SimpleNamespace(
        llm=SimpleNamespace(_cost_budget=owner),
        packet={"language": "en"},
        envelope=None,
        identity="owner",
        state={"attempts": body_attempts},
        journal={"calls": {prefix: {}}},
        call=AsyncMock(side_effect=[oversized, compact]),
        save=Mock(),
    )
    if settled:
        assert (
            await WriterSession.checkpoint_summary(session, prefix, "Prompt", {}, {})
            is compact
        )
        assert [call.args[0] for call in session.call.await_args_list] == [
            prefix,
            f"{prefix}:length_retry",
        ]
        assert "120–160 English words" in session.call.await_args_list[0].args[1]
        assert (
            "previous response exceeded capacity"
            in session.call.await_args_list[1].args[1]
        )
        assert (
            session.journal["calls"][prefix]["length_retry_reason"]["code"]
            == "writer_summary_capacity_exceeded"
        )
    else:
        with pytest.raises(WriterError, match="writer_summary_capacity_exceeded"):
            await WriterSession.checkpoint_summary(session, prefix, "Prompt", {}, {})
        assert session.call.await_count == 1
    assert session.state["attempts"] == [
        {"index": 1, "candidate_sha256": "frozen-body"}
    ]
