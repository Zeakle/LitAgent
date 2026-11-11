"""Carry freshly prepared evidence and settled spend into full report consumers."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest
import yaml
from jsonschema import Draft202012Validator
from test_extraction_admission import dual_transport
from test_extraction_paper_batches import strategy, task
from test_native_responses import owner_for
from test_question_focused_preparation import POLICY, focused_config
from test_structured_repair_pipeline import LongScientificGateway
from test_writer_source_view_pipeline import conform_evaluation_replies, source_flags
from test_writer_v2 import case  # noqa: F401

from litagent.agents.extractor import ExtractorWorker
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.config import load_config
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import EvidenceSelector
from litagent.evidence import collect_ledger
from litagent.extraction_verification import ADMISSION_VERSION, ExtractionVerifier
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.llm.client import OpenAICompatibleClient
from litagent.quality import build_fact_cards, digest
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget
from litagent.workbench import default_verification_options
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    SCOPE,
    source_packet,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 2])
async def test_fresh_extraction_jev_writer_complete_evaluation_and_spend(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Mock only external peers; retain source, budget and transaction consumers."""
    config = focused_config()
    raw = tmp_path / "raw"
    raw.mkdir()
    entries, papers, meanings = [], [], {}
    for index, (name, item) in enumerate(case.ledger.items()):
        pid = f"arxiv:2601.{12345 + index}"
        quote = item["supporting_text"]
        if name == "b":
            quote += " The study considers few-shot classification."
        target = raw / f"source-{index}.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "2 Mechanisms")
        page.insert_text((72, 112), quote)
        document.save(target)
        document.close()
        entries.append(
            {
                "paper_id": pid,
                "arxiv_id": f"2601.{12345 + index}",
                "title": f"Controlled source {name}",
                "pdf": {
                    "kind": "local_pdf",
                    "path": target.name,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                },
            }
        )
        papers.append(
            {
                "paper_id": pid,
                "title": f"Controlled source {name}",
                "abstract": quote,
                "chunks": [],
            }
        )
        meanings[pid] = (name, item["supporting_text"])
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "corpus_version": "fixture", "papers": entries}
        )
    )
    config.rag.raw_root, config.rag.manifest_path = str(raw), str(manifest)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    extraction_client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    sdk_calls, jev_calls = [], []

    async def create(**kwargs):
        sdk_calls.append(copy.deepcopy(kwargs))
        payload = json.loads(kwargs["messages"][1]["content"])
        assert payload["host_context"]["research_question"] == case.packet["query"]
        response = {"papers": {}}
        for member, source in payload["papers"].items():
            key = next(
                k for k in source["allowed_source_chunk_keys"] if k != "abstract"
            )
            response["papers"][member] = {
                "claim_records": [
                    {
                        "finding": meanings[source["paper_id"]][1],
                        "source_chunk_key": key,
                        "confidence": None,
                        "fact_fields": dict.fromkeys(
                            (
                                "method",
                                "conditions",
                                "dataset",
                                "metric",
                                "value",
                                "unit",
                                "qualifiers",
                            )
                        ),
                    }
                ]
            }
        Draft202012Validator(
            kwargs["response_format"]["json_schema"]["schema"]
        ).validate(response)
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(response), tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(
        extraction_client._client.chat.completions,
        "create",
        AsyncMock(side_effect=create),
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    try:
        async with httpx.AsyncClient(transport=dual_transport(jev_calls)) as external:
            from litagent.rag.targeted_fulltext import TargetedFulltext

            loader = TargetedFulltext(
                config.rag, client=external, preparation_policy=POLICY
            )
            gate = ExtractionVerifier(
                tmp_path / "jev",
                protocol=ADMISSION_VERSION,
                auditor=JevCitationAuditor(client=external),
            )
            worker = ExtractorWorker(
                strategy(extraction_client),
                fulltext_loader=loader,
                strict_paper_batch_size=2,
                strict_grounded_claims=True,
                extraction_verifier=gate,
                run_id_provider=lambda: "fresh-preparation",
                evidence_preparation_policy=POLICY,
                preparation_budget_path_provider=lambda: tmp_path / "preparation.json",
            )
            request = task(papers)
            verification = default_verification_options().model_dump(mode="json")
            request.input_data.update(
                query=case.packet["query"],
                coverage_requirements=case.packet["requirements"],
                verification=verification,
            )
            extractions = await worker.execute(request)
    finally:
        await extraction_client.close()
    assert len(sdk_calls) == len(jev_calls) == 1 and budget.used == 20
    spend = budget.snapshot()
    ledger = collect_ledger(extractions)
    assert len(ledger) == 2
    selector = EvidenceSelector(None, BudgetManager())
    selection = await selector.select(
        case.packet["query"], ledger, max_tokens=16000, reuse_selected=True
    )
    selection.writing_scope_policy = SCOPE
    assert set(selection.selected_items) == set(ledger)
    prepared = copy.deepcopy(case)
    prepared.ledger, prepared.selection = ledger, selection
    prepared.cards = build_fact_cards(ledger)
    prepared.packet, prepared.aliases = source_packet(
        prepared.cards,
        selection,
        case.packet["query"],
        case.packet["requirements"],
        "en",
    )
    prepared.by_eid = {
        meanings[prepared.cards[cid]["paper_id"]][0]: ref
        for ref, cid in prepared.aliases.items()
    }
    for section in prepared.body.values():
        if not isinstance(section, list):
            continue
        for paragraph in section:
            for sentence in paragraph["sentences"]:
                sentence["fact_card_refs"] = [prepared.by_eid["b"]]
    bundle = {
        "query": prepared.packet["query"],
        "requirements": prepared.packet["requirements"],
        "selection": selection.to_dict(),
        "source_run_id": "controlled-preparation-fixture",
        "extractions": extractions,
        "fact_cards_sha256": digest(prepared.cards),
        "methodology": build_methodology(
            prepared.packet["query"],
            {
                "__search_queries__": {"search_fixture": prepared.packet["query"]},
                "search_fixture": papers,
                "dedup": papers,
                "relevance_gate": papers,
            },
            source_outcomes=[
                {
                    "task_id": "search_fixture",
                    "source": "controlled-fixture",
                    "query": prepared.packet["query"],
                    "status": "completed",
                    "result_count": 2,
                }
            ],
            started_at=1700000000,
        ),
    }
    original = copy.deepcopy(bundle)
    gateway = conform_evaluation_replies(
        LongScientificGateway(prepared, failing_candidates, reviewer=True)
    )
    owner, _, _ = await owner_for(gateway)
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens, config.adversarial.review_max_tokens = 20000, 4096
    config.observability.enabled, config.mcp_servers = False, {}
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model, owner._cost_budget, owner._trace_hook = (
            kwargs["model"],
            kwargs["cost_budget"],
            kwargs["trace_hook"],
        )
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    original_options = frozen_options
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: original_options().model_copy(update={"report_language": "en"}),
    )
    flags = source_flags()
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        preparation_budget_snapshot=spend,
        **flags,
    )
    assert plan["spent_tokens"] == 20 and plan["spent_requests"] == 1
    try:
        report = await run_frozen(
            agent,
            bundle,
            "fresh-preparation",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            preparation_budget_snapshot=spend,
            verification=verification,
            **flags,
        )
    finally:
        await agent.cleanup()
    (tmp_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    assert bundle == original
    assert report["metadata"]["preparation_usage"]["used_tokens"] == 20
    assert report["metadata"]["preparation_usage"]["requests_sent"] == 1
    assert len(jev_calls) == 1
    assert gateway.repair_count == (1 if failing_candidates == 0 else 2)
    assert gateway.inventories == (2 if failing_candidates == 0 else 3)
    assert report["metadata"]["automatic_acceptance"]["publishable"] is (
        failing_candidates == 0
    ), report["metadata"]["automatic_acceptance"]["failure_codes"]
    if failing_candidates:
        assert "FLOPs by 44%" in report["survey"]
    else:
        assert "FLOPs by 44%" not in report["survey"]
        assert report["evaluation"]["faithfulness"]["score"] == 1.0
        assert (
            report["metadata"]["automatic_acceptance"]["gates"]["extraction_jev"]
            == "passed"
        )
