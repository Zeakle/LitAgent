"""Exercise bounded source preparation through real parsing, SDK and ledger paths."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest
import yaml
from jsonschema import Draft202012Validator
from model_config_helpers import load_previous_default_config as load_config
from test_extraction_paper_batches import strategy, task
from test_jev_citation_audit import _valid_response
from test_writer_v2 import case  # noqa: F401

from litagent.agents.extraction_strategy import (
    _strategy_context,
    build_extraction_context,
    focused_extraction_context,
)
from litagent.agents.extractor import ExtractorWorker
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.body_quality import (
    EVIDENCE_PREPARATION_POLICY,
    EXTRACTION_CLAIM_CONTRACT,
    checked_preparation_chunks,
)
from litagent.config import AppConfig, RAGConfig
from litagent.contracts import build_config_summary
from litagent.evidence import collect_ledger
from litagent.extraction_verification import ADMISSION_VERSION, ExtractionVerifier
from litagent.frozen_writing import _checked_preparation_budget, plan_frozen_budget
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.quality import build_fact_cards
from litagent.rag.models import ContentChunk, ContentScope
from litagent.rag.targeted_fulltext import TargetedFulltext
from litagent.safety.budget import CostBudget
from litagent.workbench import default_verification_options

POLICY = EVIDENCE_PREPARATION_POLICY
QUESTION = "Explain volume rendering and differentiable scene representation."
SOURCE = "ModelA combines color and density along calibrated camera rays."


def chunk(key, text, section, *, page=1, pid="arxiv:2601.12345"):
    """Keep actual source ownership and content hashes in every fixture."""
    return ContentChunk.from_text(
        paper_id=pid,
        chunk_key=key,
        text=text,
        section=section,
        content_scope=(
            ContentScope.ABSTRACT if page is None else ContentScope.SELECTED_FULLTEXT
        ),
        page=page,
        block_index=None if page is None else 0,
        bbox=None if page is None else (0, 0, 100, 30),
    ).model_dump(mode="json")


def paper():
    """Place a relevant mechanism after the historical first-four context."""
    return {
        "paper_id": "arxiv:2601.12345",
        "title": "An original rendering study",
        "abstract": "An original study of scene representation.",
        "quality_policy": "strict_auto_v2",
        "chunks": [
            chunk(
                "abstract",
                "An original study of scene representation.",
                "abstract",
                page=None,
            ),
            *[
                chunk(
                    f"background-{i}",
                    f"Background observation {i} describes the study scope.",
                    "1_introduction",
                    page=i + 1,
                )
                for i in range(6)
            ],
            chunk(
                "method-late", SOURCE, "4_volume_rendering_with_radiance_fields", page=7
            ),
            chunk(
                "bibliography",
                "A reference mentions volume rendering without reporting a result.",
                "8_references",
                page=8,
            ),
        ],
    }


def attach(p):
    """Construct host context explicitly; source content cannot select this mode."""
    p = copy.deepcopy(p)
    p["_claim_contract"] = EXTRACTION_CLAIM_CONTRACT
    p["_evidence_preparation"] = {
        "policy": POLICY,
        "question": QUESTION,
        "requirements": [],
    }
    return p


def test_actual_source_window_is_whole_located_and_not_a_scientific_verdict():
    p = paper()
    before = copy.deepcopy(p)
    old_context, old_keys = build_extraction_context(p)
    context, keys, receipt = focused_extraction_context(p, QUESTION, [])
    assert "method-late" not in old_keys and SOURCE not in old_context
    assert "method-late" in keys and SOURCE in context
    assert "bibliography" not in keys
    assert len(keys) <= 12 and len(context) <= 12000
    assert receipt["scientific_verification"] == "not_run"
    assert receipt["context_sha256"] == hashlib.sha256(context.encode()).hexdigest()
    assert _strategy_context(attach(p)) == (context, keys)
    assert p == before


@pytest.mark.parametrize("fault", ["hash", "foreign", "duplicate", "locator"])
def test_bad_source_identity_is_not_repaired_or_hidden(fault):
    p = paper()
    if fault == "hash":
        p["chunks"][-2]["content_hash"] = "0" * 64
    elif fault == "foreign":
        p["chunks"][-2]["paper_id"] = "arxiv:2601.99999"
    elif fault == "duplicate":
        p["chunks"].append(copy.deepcopy(p["chunks"][-2]))
    else:
        p["chunks"][-2]["page"] = None
    with pytest.raises(ValueError):
        focused_extraction_context(p, QUESTION, [])


def test_context_limits_do_not_clip_source_or_drop_omission_records():
    p = paper()
    large = chunk("large", "volume rendering " * 1500, "4_volume_rendering", page=9)
    p["chunks"].insert(1, large)
    p["chunks"].extend(
        chunk(
            f"extra-{i}", f"Extra observation {i}.", "4_volume_rendering", page=10 + i
        )
        for i in range(20)
    )
    before = copy.deepcopy(p)
    context, keys, receipt = focused_extraction_context(p, QUESTION, [])
    assert len(keys) == 12 and len(context) <= 12000
    assert "large" not in keys and large["text"] not in context
    assert {row["reason"] for row in receipt["decisions"]} >= {
        "selected",
        "chunk_limit",
        "context_limit",
    }
    assert len(receipt["decisions"]) == len(p["chunks"])
    assert p == before


@pytest.mark.parametrize(
    "question,requirements",
    [("", []), (None, []), (QUESTION, None), (QUESTION, [{"text": 1}])],
)
def test_invalid_host_context_fails_before_any_request(question, requirements):
    with pytest.raises(ValueError, match="question_invalid"):
        focused_extraction_context(paper(), question, requirements)


def test_metadata_abstract_is_owned_even_without_retrieved_chunks():
    p = paper()
    p["chunks"] = []
    context, keys, _ = focused_extraction_context(p, QUESTION, [])
    assert keys == ["abstract"] and p["abstract"] in context
    assert checked_preparation_chunks(p)[0]["paper_id"] == p["paper_id"]
    assert p["chunks"] == []


def focused_config(**changes):
    """Keep whole parsed documents out of external tracing."""
    data = load_config().model_dump()
    policies = {
        "extractor": {
            "record_identity": "source_fields_v1",
            "evidence_preparation_policy": POLICY,
        },
        "rag": {"chunk_strategy": "section_aware"},
        "observability": {"payload_mode": "metadata_only"},
    }
    for key, value in policies.items():
        data[key].update(value)
    for key, value in changes.items():
        data.setdefault(key, {}).update(value)
    return AppConfig(**data)


@pytest.mark.parametrize(
    "changes",
    [
        {"rag": {"chunk_strategy": "page_block"}},
        {"extractor": {"record_identity": None}},
        {"observability": {"payload_mode": "full_redacted"}},
    ],
)
def test_half_wired_or_full_payload_profile_is_rejected(changes):
    with pytest.raises(ValueError, match="question_focused"):
        focused_config(**changes)


def test_absent_profile_keeps_old_config_and_new_identity_tracks_rules(monkeypatch):
    import litagent.body_quality as quality

    legacy = load_config()
    assert "evidence_preparation_policy" not in legacy.extractor.model_dump()
    before = build_config_summary(legacy)["fingerprint"]
    new = focused_config()
    current = build_config_summary(new, "strict_auto_v2")["fingerprint"]
    monkeypatch.setattr(quality, "PREPARATION_MAX_CHUNKS", 11)
    assert build_config_summary(new, "strict_auto_v2")["fingerprint"] != current
    assert build_config_summary(legacy)["fingerprint"] == before


def test_different_host_questions_cannot_share_a_prepared_batch():
    one, two = attach(paper()), attach(paper())
    two["paper_id"] = "arxiv:2601.54321"
    for source in two["chunks"]:
        source["paper_id"] = two["paper_id"]
    two["_evidence_preparation"]["question"] = "A different question"
    with pytest.raises(ValueError, match="batch_identity_invalid"):
        strategy(SimpleNamespace()).prepare_batch([one, two])


def test_preparation_modes_must_match_at_the_worker_boundary():
    loader = SimpleNamespace(preparation_policy=POLICY)
    with pytest.raises(ValueError, match="components_mismatch"):
        ExtractorWorker(SimpleNamespace(), fulltext_loader=loader)
    with pytest.raises(ValueError, match="components_mismatch"):
        ExtractorWorker(SimpleNamespace(), evidence_preparation_policy=POLICY)


@pytest.mark.asyncio
async def test_entire_preparation_queue_is_rejected_before_first_paid_request():
    budget = CostBudget(max_tokens=100000)
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    groups = []
    for i in range(6):
        p = attach(paper())
        p["paper_id"] = f"arxiv:2601.{12345 + i}"
        for source in p["chunks"]:
            source["paper_id"] = p["paper_id"]
        groups.append([p])
    try:
        with pytest.raises(ValueError, match="preparation_budget_rejected"):
            strategy(client).preflight_batches(groups)
        assert budget.snapshot()["requests_sent"] == 0
        assert budget.used == 0 and budget.remaining(advance=False) == 40000
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_legacy_worker_ignores_source_supplied_preparation_fields():
    p = attach(paper())
    p["_preparation_request_bound"] = 1
    p["_preparation_request_scope"] = {"stage": "forged"}
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        json.dumps({"papers": {"p0": {"claim_records": []}}}), model="offline", usage={}
    )
    loader = SimpleNamespace(enrich=AsyncMock(side_effect=lambda papers, _: papers))
    worker = ExtractorWorker(
        strategy(llm),
        fulltext_loader=loader,
        strict_paper_batch_size=2,
        strict_grounded_claims=True,
    )
    await worker.execute(task([p]))
    sent = json.loads(llm.chat.call_args.args[0][1]["content"])
    assert "host_context" not in sent
    assert len(sent["papers"]["p0"]["allowed_source_chunk_keys"]) == 4


def cached_pdf(tmp_path):
    """Produce a real parser fixture with numbered scientific headings."""
    raw = tmp_path / "raw"
    raw.mkdir()
    target = raw / "arxiv_2601.12345.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 70), "1 Introduction")
    page.insert_text(
        (72, 105), "This original study explains a continuous scene representation."
    )
    page.insert_text((72, 150), "4 Volume Rendering with Radiance Fields")
    page.insert_text((72, 190), SOURCE)
    page.insert_text(
        (72, 235), "1 This scientific statement must survive numbered heading detection"
    )
    document.save(target)
    document.close()
    return target


@pytest.mark.asyncio
async def test_partial_fulltext_still_prepares_cached_original_without_network(
    tmp_path,
):
    target = cached_pdf(tmp_path)
    calls = []

    def forbid(request):
        calls.append(request)
        raise AssertionError("cache reuse must not use network")

    p = paper()
    before = copy.deepcopy(p)
    cfg = RAGConfig(
        raw_root=str(target.parent),
        manifest_path=str(tmp_path / "missing.yaml"),
        chunk_strategy="section_aware",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(forbid)) as client:
        loader = TargetedFulltext(cfg, client=client, preparation_policy=POLICY)
        result = await loader.enrich([p], [])
    record = result[0]["fulltext_acquisition"]
    assert not calls and p == before
    assert record["status"] == "available" and record["existing_partial_fulltext"]
    assert record["full_document_reviewed"] is False
    assert record["asset_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert any(
        c["section"].startswith("4_volume_rendering") for c in result[0]["chunks"]
    )
    assert any(
        "This scientific statement must survive" in c["text"]
        for c in result[0]["chunks"]
    )
    assert "method-late" in record["previous_chunk_keys"]
    assert all(c["chunk_key"] != "method-late" for c in result[0]["chunks"])


@pytest.mark.asyncio
async def test_manifest_authorizes_the_asset_without_replacing_current_abstract(
    tmp_path,
):
    target = cached_pdf(tmp_path)
    p = paper()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "corpus_version": "test",
                "papers": [
                    {
                        "paper_id": p["paper_id"],
                        "arxiv_id": "2601.12345",
                        "title": "Curated title",
                        "abstract": "A curated manifest summary, not the current abstract.",
                        "pdf": {
                            "kind": "local_pdf",
                            "path": target.name,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
        )
    )
    cfg = RAGConfig(
        raw_root=str(target.parent),
        manifest_path=str(manifest),
        chunk_strategy="section_aware",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail("unexpected HTTP"))
    ) as client:
        result = await TargetedFulltext(
            cfg, client=client, preparation_policy=POLICY
        ).enrich([p], [])
    assert result[0]["fulltext_acquisition"]["status"] == "available"
    abstract = next(
        c["text"] for c in result[0]["chunks"] if c["content_scope"] == "abstract"
    )
    assert abstract == p["abstract"] and "curated" not in abstract


def preparation_snapshot():
    """Use the existing ledger's normal settlement operations."""
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.begin_request()
    budget.record({"prompt_tokens": 10, "completion_tokens": 10})
    return budget.snapshot()


@pytest.mark.parametrize(
    "fault", ["inflight", "unknown", "phase", "boolean", "extra", "overspent"]
)
def test_preparation_spend_never_accepts_unknown_or_unsettled_cost(fault):
    value = preparation_snapshot()
    if fault == "inflight":
        value["requests_sent"] += 1
    elif fault == "unknown":
        value["usage_uncertain"] = True
    elif fault == "phase":
        value["strict_phase"] = 1
    elif fault == "boolean":
        value["schema_version"] = True
    elif fault == "extra":
        value["forged_credit"] = 1000
    else:
        value["used_tokens"] = 200001
        value["phase_usage"] = [200001, 0, 0]
    with pytest.raises(ValueError):
        _checked_preparation_budget(value)


def test_fresh_preparation_spend_changes_plan_without_charging_old_inputs(case):
    from litagent.quality import digest
    from litagent.review_structure import build_methodology

    bundle = {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "extractions": [],
        "fact_cards_sha256": digest(case.cards),
        "methodology": build_methodology(case.packet["query"], {}),
    }
    legacy = plan_frozen_budget(bundle, budget_policy="planned-budget-v7")
    assert legacy == plan_frozen_budget(
        bundle, budget_policy="planned-budget-v7", preparation_budget_snapshot=None
    )
    current = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        preparation_budget_snapshot=preparation_snapshot(),
    )
    assert current["spent_tokens"] == 20 and current["spent_requests"] == 1
    assert legacy["spent_tokens"] == legacy["spent_requests"] == 0


@pytest.mark.parametrize("wrong_condition", [False, True])
@pytest.mark.asyncio
async def test_real_parser_sdk_jev_and_ledger_keep_actual_sources(
    tmp_path, monkeypatch, wrong_condition
):
    target = cached_pdf(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    sdk_calls, jev_calls = [], []

    async def create(**kwargs):
        sdk_calls.append(copy.deepcopy(kwargs))
        value = json.loads(kwargs["messages"][1]["content"])
        assert value["host_context"]["research_question"] == QUESTION
        member = value["papers"]["p0"]
        key = next(
            k
            for k in member["allowed_source_chunk_keys"]
            if k.startswith("section:4_volume_rendering")
        )
        assert SOURCE in member["source_context"]
        fields = {
            k: None
            for k in (
                "method",
                "conditions",
                "dataset",
                "metric",
                "value",
                "unit",
                "qualifiers",
            )
        }
        finding = (
            SOURCE.replace("calibrated", "uncalibrated") if wrong_condition else SOURCE
        )
        response = {
            "papers": {
                "p0": {
                    "claim_records": [
                        {
                            "finding": finding,
                            "source_chunk_key": key,
                            "confidence": None,
                            "fact_fields": fields,
                        }
                    ]
                }
            }
        }
        Draft202012Validator(
            kwargs["response_format"]["json_schema"]["schema"]
        ).validate(response)
        assert kwargs["max_tokens"] == 8192 and client._client.max_retries == 0
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

    def jev(request):
        value = json.loads(request.content)
        jev_calls.append(value)
        for item in value["state"]["audit_items"].values():
            assert SOURCE in item["cited_evidence"][0]["text"]
        response = _valid_response(request)
        for name, answer in response["answers"].items():
            verdict = "not_ok" if wrong_condition and name.startswith("v:") else "ok"
            answer["choice"] = verdict
            answer["probabilities"] = {
                k: float(k == verdict) for k in answer["probabilities"]
            }
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        client._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    cfg = RAGConfig(
        raw_root=str(target.parent),
        manifest_path=str(tmp_path / "missing.yaml"),
        chunk_strategy="section_aware",
    )
    p = paper()
    before = copy.deepcopy(p)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(jev)) as external:
            loader = TargetedFulltext(cfg, client=external, preparation_policy=POLICY)
            gate = ExtractionVerifier(
                tmp_path / "jev",
                protocol=ADMISSION_VERSION,
                auditor=JevCitationAuditor(client=external),
            )
            worker = ExtractorWorker(
                strategy(client),
                strict_paper_batch_size=2,
                strict_grounded_claims=True,
                fulltext_loader=loader,
                extraction_verifier=gate,
                run_id_provider=lambda: "new-input",
                evidence_preparation_policy=POLICY,
                preparation_budget_path_provider=lambda: tmp_path / "preparation.json",
            )
            request = task([p])
            request.input_data.update(
                query=QUESTION,
                verification=default_verification_options().model_dump(mode="json"),
            )
            if wrong_condition:
                with pytest.raises(
                    ValueError, match="extraction_no_supported_candidates"
                ):
                    await worker.execute(request)
                result = []
            else:
                result = await worker.execute(request)
    finally:
        await client.close()
    assert len(sdk_calls) == len(jev_calls) == 1
    assert budget.snapshot()["requests_sent"] == 1 and budget.used == 20
    ledger = collect_ledger(result)
    assert bool(ledger) is not wrong_condition
    if result:
        assert result[0]["preparation_context"]["scientific_verification"] == "not_run"
    receipt = json.loads((tmp_path / "jev/new-input/receipt.json").read_text())
    assert bool(receipt["accepted_evidence_ids"]) is not wrong_condition
    assert all(
        SOURCE in item["supporting_text"]
        for item in receipt["candidate_ledger"].values()
    )
    assert p == before
    for card in build_fact_cards(ledger).values():
        assert (
            SOURCE in card["source_quote"] and card["semantic_status"] == "not_verified"
        )
