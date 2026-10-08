"""Verify paid comparison preflight and transport gates without provider calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from litagent.config import ContextConfig, load_config
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import (
    EvidenceSelection,
    format_evidence_selection,
)
from litagent.demo import project_flow_demo_view
from litagent.eval.scoped import _scientific_materials
from litagent.llm.client import LLMResponse
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import ArchiveRepository
from litagent.safety.budget import CostBudget

_spec = importlib.util.spec_from_file_location(
    "quality_comparison",
    Path(__file__).parents[1] / "scripts/compare_survey_quality.py",
)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def captured():
    """Supply old prompts with variables that must never replace the new draft."""
    messages = [
        {
            "role": "system",
            "content": "old <run_methodology>old record</run_methodology>",
        },
        {
            "role": "user",
            "content": "old draft <evidence_plan>[E:e1]</evidence_plan><evidence_ledger>old evidence</evidence_ledger>",
        },
    ]
    return {
        "captured": {
            f"adversarial_review:{part}:r{r}": messages
            for part in ["synthesis", "reviewer"]
            for r in [1, 2]
        },
        "record": {"selected": 1},
    }


def test_baseline_uses_captured_contract_but_keeps_current_draft():
    """Bind the old evidence representation and update only trusted run records."""
    messages = [
        {"role": "system", "content": "new"},
        {
            "role": "user",
            "content": "current draft <evidence_plan>new</evidence_plan><evidence_ledger>new</evidence_ledger>",
        },
    ]
    result = comparison.transform_messages(
        messages, "adversarial_review:synthesis:r2", "baseline", captured()
    )
    assert result[0]["content"].startswith("old")
    assert '"selected": 1' in result[0]["content"]
    assert "current draft" in result[-1]["content"]
    assert "old draft" not in result[-1]["content"]
    assert "old evidence" in result[-1]["content"]
    assert messages[0]["content"] == "new"


def test_candidate_and_evaluation_contracts_are_not_replaced():
    """Both arms share Evaluation v2 rather than replaying the historical metric."""
    messages = [
        {"role": "system", "content": "v2"},
        {"role": "user", "content": "frozen"},
    ]
    assert (
        comparison.transform_messages(
            messages, "evaluation.initial", "baseline", captured()
        )
        == messages
    )
    assert (
        comparison.transform_messages(
            messages, "adversarial_review:reviewer:r1", "candidate", captured()
        )
        == messages
    )


def test_prepare_rejects_changed_frozen_input_before_network(tmp_path):
    """A SHA conflict must not initialize a model client or restore an old checkpoint."""
    (tmp_path / "frozen.json").write_text("{}")
    (tmp_path / "controlled-comparison-plan.json").write_text(
        json.dumps({"input": {"path": "frozen.json", "sha256": "invalid"}})
    )
    with pytest.raises(ValueError, match="frozen_materials_sha_conflict"):
        comparison.prepare(tmp_path)


@pytest.mark.asyncio
async def test_failed_transport_is_not_reissued_and_reservation_released(tmp_path):
    """A caught fallback cannot submit another request after uncertain failure."""
    calls = []

    async def fail(messages, **kwargs):
        calls.append(kwargs)
        raise TimeoutError()

    budget = CostBudget(10000)
    provider = SimpleNamespace(chat=fail, _max_tokens=100)
    client = comparison.BoundedClient(
        provider, budget, "candidate", captured(), tmp_path
    )
    with pytest.raises(TimeoutError):
        await client.chat([{"role": "user", "content": "one"}])
    with pytest.raises(ValueError, match="gate_closed"):
        await client.chat([{"role": "user", "content": "two"}])
    assert len(calls) == 1 and calls[0]["transport_max_attempts"] == 1
    assert budget.remaining() == 10000
    assert (
        json.loads((tmp_path / "candidate-call-01.json").read_text())["status"]
        == "interrupted_no_replay"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["evaluation.initial", "evidence_rewrite"])
async def test_evaluation_and_repair_reuse_existing_reservation(tmp_path, stage):
    """Do not reserve request inputs already owned by the calling stage twice."""
    budget = CostBudget(10000)
    budget.reserve(9500)

    async def respond(messages, **kwargs):
        return LLMResponse("{}", model="offline", usage={"usage_available": True})

    client = comparison.BoundedClient(
        SimpleNamespace(chat=respond, _max_tokens=100),
        budget,
        "candidate",
        captured(),
        tmp_path,
    )
    token = set_task_id(stage)
    try:
        await client.chat([{"role": "user", "content": "one"}])
    finally:
        reset_task_id(token)
    assert budget.remaining() == 500


@pytest.mark.asyncio
async def test_budget_failure_is_local_before_started_marker(tmp_path):
    """Do not charge a call when input/output cannot fit the remaining budget."""
    client = comparison.BoundedClient(
        SimpleNamespace(_max_tokens=100),
        CostBudget(10),
        "candidate",
        captured(),
        tmp_path,
    )
    with pytest.raises(ValueError, match="run_budget_exceeded"):
        await client.chat([{"role": "user", "content": "one"}])
    assert client.calls == 0 and not list(tmp_path.iterdir())


def test_sixty_evidence_items_share_complete_originals_within_same_budget():
    """Repeated claims retain all original text and bindings without layer truncation."""
    items = {
        f"e{i}": {
            "evidence_id": f"e{i}",
            "paper_id": f"p{i % 33}",
            "paper_title": f"Method {i % 33}",
            "text": "Claim " + str(i),
            "supporting_text": "Full original abstract " + str(i % 33) + "x" * 1400,
        }
        for i in range(60)
    }
    selection = EvidenceSelection(
        185, items, {"methods": list(items)}, {"methods": "cross_encoder"}, 125, 0
    )
    formatted = format_evidence_selection(selection)
    assert (
        BudgetManager().count_tokens(formatted)
        <= ContextConfig().synthesis_evidence_max_tokens
    )
    assert formatted.count("<s id=") == 33
    assert formatted.endswith("</evidence_ledger>")
    assert all(f"[E:{eid}] source=" in formatted for eid in items)
    assert formatted.count(items["e0"]["supporting_text"]) == 1
    materials, supports = _scientific_materials(items, list(items))
    assert len(materials) == 60 and len(supports) == 33
    assert (
        materials["e0"]["original_support_ref"]
        == materials["e33"]["original_support_ref"]
    )


def test_same_original_text_from_different_or_unknown_papers_is_not_merged():
    """Equal strings do not establish equal paper identity."""
    items = {
        "a": {"text": "claim", "paper_id": "first", "supporting_text": "original"},
        "b": {"text": "claim", "paper_id": "second", "supporting_text": "original"},
        "c": {"text": "claim", "supporting_text": "original"},
        "d": {"text": "claim", "supporting_text": "original"},
    }
    evidence, supports = _scientific_materials(items, list(items))
    assert len(supports) == 4
    assert len({v["original_support_ref"] for v in evidence.values()}) == 4


def frozen_data():
    """Build self-contained source records matching the real archive producers."""
    papers = [
        {
            "paper_id": "paper-1",
            "title": "Token Evolution",
            "authors": ["Test Author"],
            "year": 2022,
        }
    ]
    evidence = {
        "ev-1": {
            "evidence_id": "ev-1",
            "paper_id": "paper-1",
            "paper_title": "Token Evolution",
            "text": "Finding A",
            "supporting_text": "Original abstract",
            "source_locator": "abstract",
            "content_scope": "abstract",
        }
    }
    outputs = {
        "dedup": papers,
        "relevance_gate": papers,
        "extract": [{"evidence_items": list(evidence.values()), "claims": []}],
        "graph_analysis": {},
    }
    selection = EvidenceSelection(
        1, evidence, {"methods": ["ev-1"]}, {"methods": "cross_encoder"}, 0, 100
    )
    return {
        "frozen": {
            "query": "Token evolution",
            "source_artifact": "historical.json",
            "source_artifact_sha256": "frozen-sha",
            "host_runtime_record": {},
            "paper_catalog": {},
            "worker_outputs": {
                f"worker:{k}": {"task_id": k, "output": v} for k, v in outputs.items()
            },
        },
        "selection": selection,
        "upstream": {"extract": outputs["extract"], "graph_analysis": {}},
        "record": {},
        "plan": {"input": {"sha256": "frozen-input-sha"}},
        "captured": {},
    }


def complete_evaluation():
    """A low supported score may be complete, while missing coverage may not."""
    return {
        key: {
            "score": 0.5 if key == "faithfulness" else 1,
            "passed": key != "faithfulness",
            "skipped": False,
            "details": {
                "evaluation_version": "survey-evaluation-v2",
                "coverage": {"complete": True},
            },
        }
        for key in ("citation_accuracy", "faithfulness", "internal_consistency")
    }


@pytest.mark.parametrize(
    "broken",
    [
        "missing_metric",
        "skipped",
        "no_coverage",
        "wrong_version",
        "no_score",
        "invalid_score",
        "boolean_score",
        "no_passed",
    ],
)
def test_next_paid_arm_requires_complete_valid_evaluation(broken):
    evaluation = complete_evaluation()
    assert comparison.evaluation_complete(evaluation)
    metric = evaluation["faithfulness"]
    if broken == "missing_metric":
        evaluation.pop("internal_consistency")
    elif broken == "skipped":
        metric["skipped"] = True
    elif broken == "no_coverage":
        metric["details"]["coverage"] = {}
    elif broken == "wrong_version":
        metric["details"]["evaluation_version"] = "legacy"
    elif broken == "no_score":
        metric.pop("score")
    elif broken == "invalid_score":
        metric["score"] = float("nan")
    elif broken == "boolean_score":
        metric["score"] = True
    else:
        metric.pop("passed")
    assert not comparison.evaluation_complete(evaluation)


@pytest.mark.asyncio
async def test_real_comparison_archive_restores_frozen_evidence_and_paper_metadata(
    tmp_path, monkeypatch
):
    """Exercise the arm producer, atomic repository and workbench consumer offline."""
    repository = ArchiveRepository(tmp_path / "archives")
    monkeypatch.setattr(comparison, "ArchiveRepository", lambda: repository)
    data = frozen_data()

    class OfflineTrace:
        trace_id, trace_url = "offline", ""

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args):
            pass

        def flush(self):
            pass

    class NoProvider:
        def __init__(self, *args, **kwargs):
            self._client = self

        async def close(self):
            pass

        async def chat(self, *args, **kwargs):
            pytest.fail("Archive integration must not call a paid provider.")

    async def draft(self, task):
        assert task.input_data["review_structure"] == "academic_v2"
        return {
            "final_draft": "# Review\n\nFinding A [E:ev-1].\n",
            "rounds": [],
            "review_structure": task.input_data["review_structure"],
            "review_methodology": {},
            "evidence_selection": {"valid": True},
            "total_rounds": 0,
        }

    async def evaluate(self, **kwargs):
        assert (
            kwargs["results"]["adversarial_review"]["review_structure"] == "academic_v2"
        )
        result = complete_evaluation()
        result["faithfulness"].update(score=1, passed=True)
        return result

    monkeypatch.setattr(comparison, "LangFuseTracer", OfflineTrace)
    monkeypatch.setattr(comparison, "OpenAICompatibleClient", NoProvider)
    monkeypatch.setattr(comparison.AdversarialReviewWorker, "execute", draft)
    monkeypatch.setattr(comparison.LitAgent, "_evaluate", evaluate)
    root = tmp_path / "fresh-arm"
    root.mkdir()
    data["execution"] = {"review_structure": "academic_v2", "arms": ["candidate"]}
    result = await comparison.run_arm("candidate", data, load_config(), root)
    assert result["calls"] == 0 and result["contract_complete"]
    artifact = repository.get(result["run_id"])
    view = project_flow_demo_view(artifact, source="archive")
    assert view["evidence"][0]["evidence_id"] == "ev-1"
    assert view["evidence"][0]["locator"] == "abstract"
    rendered = view["rendered_report"]
    assert "Token Evolution" in rendered["html"] and "Test Author" in rendered["html"]
    assert not rendered["warnings"]
    assert artifact["report"]["metadata"]["citation_style"] == "ieee"
    assert artifact["report"]["metadata"]["review_structure"] == "academic_v2"
    assert (
        artifact["nodes"]["worker:extract"]["metadata"]["origin"]
        == "frozen_input_replay"
    )
    assert artifact["nodes"]["worker:extract"]["started_at"] is None
    assert not any("search" in k for k in artifact["nodes"])
    assert data["selection"].selected_items["ev-1"]["text"] == "Finding A"
    with pytest.raises(ValueError, match="already_started_no_replay"):
        await comparison.run_arm("candidate", data, load_config(), root)


@pytest.mark.asyncio
async def test_incomplete_baseline_stops_main_before_candidate(tmp_path, monkeypatch):
    """The pair coordinator must not submit a second paid arm for invalid coverage."""
    data = frozen_data()
    data["captured"] = captured()["captured"]
    cfg = load_config()
    data["plan"]["fixed_parameters"] = comparison.fixed_parameters(cfg)
    monkeypatch.setattr(comparison, "prepare", lambda path: data)
    monkeypatch.setenv("LLM_API_KEY", "offline-unused")
    arms = []

    async def incomplete(arm, *args):
        arms.append(arm)
        return {"contract_complete": comparison.evaluation_complete({})}

    monkeypatch.setattr(comparison, "run_arm", incomplete)
    await comparison.main(
        SimpleNamespace(input_root=tmp_path, output=tmp_path / "pair", live=True)
    )
    assert arms == ["baseline"]
    assert set(
        json.loads((tmp_path / "pair" / "comparison.json").read_text())["results"]
    ) == {"baseline"}


@pytest.mark.asyncio
async def test_existing_archive_blocks_replay_even_without_started_marker(
    tmp_path, monkeypatch
):
    """A reused directory basename must not overwrite an older report artifact."""
    repository = ArchiveRepository(tmp_path / "archive")
    repository.save(
        {
            "run_id": "quality-v2-compare-pair-candidate",
            "report": {"survey": "Original"},
        }
    )
    original = repository.path_for("quality-v2-compare-pair-candidate").read_bytes()
    monkeypatch.setattr(comparison, "ArchiveRepository", lambda: repository)
    root = tmp_path / "pair"
    root.mkdir()
    with pytest.raises(ValueError, match="artifact_already_exists_no_replay"):
        await comparison.run_arm("candidate", frozen_data(), load_config(), root)
    assert not list(root.iterdir())
    assert (
        repository.path_for("quality-v2-compare-pair-candidate").read_bytes()
        == original
    )


def test_execution_defaults_preserve_pair_and_reject_old_prompts_for_v2():
    assert comparison.execution_options(SimpleNamespace()) == (
        "method_transparent_v1",
        ["baseline", "candidate"],
    )
    for arm in ["both", "baseline"]:
        with pytest.raises(ValueError, match="requires_current_single_arm"):
            comparison.execution_options(
                SimpleNamespace(review_structure="academic_v2", arm=arm)
            )
    with pytest.raises(ValueError, match="structure_invalid"):
        comparison.execution_options(SimpleNamespace(review_structure="wrong"))
    with pytest.raises(ValueError, match="arm_invalid"):
        comparison.execution_options(SimpleNamespace(arm="wrong"))


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
async def test_academic_v2_single_attempt_preflight_has_no_provider(
    tmp_path, monkeypatch, live
):
    """Build real current prompts, then submit only the selected current arm."""
    data = frozen_data()
    data["captured"] = captured()["captured"]
    cfg = load_config()
    data["plan"]["fixed_parameters"] = comparison.fixed_parameters(cfg)
    monkeypatch.setattr(comparison, "prepare", lambda path: data)
    monkeypatch.setenv("LLM_API_KEY", "offline-unused")

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not initialize a paid provider.")

    monkeypatch.setattr(comparison, "OpenAICompatibleClient", forbidden)
    arms = []

    async def complete(arm, runtime, config, root):
        assert runtime["execution"]["review_structure"] == "academic_v2"
        arms.append(arm)
        return {"contract_complete": True}

    monkeypatch.setattr(comparison, "run_arm", complete)
    output = tmp_path / "fresh"
    await comparison.main(
        SimpleNamespace(
            input_root=tmp_path,
            output=output,
            live=live,
            review_structure="academic_v2",
            arm="candidate",
        )
    )
    assert arms == (["candidate"] if live else [])
    preflight = json.loads((output / "preflight.json").read_text())
    assert preflight["total_tokens_max"] == cfg.safety.max_cost_tokens
    assert preflight["execution"]["arms"] == ["candidate"]
    request = json.loads((output / "initial-request-dry-run.json").read_text())
    system = request["messages"][0]["content"]
    assert "academic_v2" in system
    assert "Write only the seven academic body sections" in system
    assert "creates References and Appendix A afterwards" in system
    assert not system.startswith("old")
    assert request["input_output_reservation"] < 131072


@pytest.mark.asyncio
async def test_preflight_rejects_changed_evaluation_limit(tmp_path, monkeypatch):
    """A material or context change cannot silently inherit the frozen budget."""
    data = frozen_data()
    data["plan"]["fixed_parameters"] = comparison.fixed_parameters(load_config())
    data["plan"]["fixed_parameters"]["evaluation_max_evidence_items"] = 40
    monkeypatch.setattr(comparison, "prepare", lambda path: data)
    with pytest.raises(ValueError, match="config_conflict"):
        await comparison.main(
            SimpleNamespace(input_root=tmp_path, output=tmp_path / "new", live=False)
        )
    assert not (tmp_path / "new").exists()
