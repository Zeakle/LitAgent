"""Reproduce the 404 evidence, inventory, and repetition failures offline."""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from litagent.agents.extraction_strategy import _strategy_context
from litagent.config import load_config
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import DEFAULT_SECTIONS, EvidenceSelector
from litagent.eval.engine_policy import KEYED_GLOBAL_QUOTED_OMISSIONS
from litagent.eval.ragas_global_wire import unpack_global_input
from litagent.eval.ragas_native import (
    _restore_issue_anchors,
    canonical_assertions,
    parse_native_global_response,
)
from litagent.eval.ragas_repair import (
    preserves_repetition_findings,
    repetition_followup,
)
from litagent.rag.models import RawPaperAsset
from litagent.rag.targeted_fulltext import TargetedFulltext
from litagent.runner import RewriteOutcome

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_404_few_shot_admits_only_an_exact_bound_repetition_retry():
    """Preserve the fully supported draft for one targeted attempt."""
    run = "7c2d5b5ed271"
    path = ARTIFACTS / "acceptance/workbench-vit-386-20261110/runs" / f"{run}.json"
    if not path.is_file():
        pytest.skip("local 404 archive is unavailable")
    original = json.loads(path.read_bytes())["report"]
    journal = json.loads(
        (ARTIFACTS / "targeted-repairs" / f"{run}.json.writer-v2.json").read_bytes()
    )
    draft = journal["candidates"][1]
    assert draft["committed"] is False
    assert (
        draft["source_report_sha256"]
        == hashlib.sha256(original["survey"].encode()).hexdigest()
    )
    proposal = RewriteOutcome(
        attempted=True,
        committed=False,
        candidate=draft["survey"],
        validation=None,
        candidate_evaluation=draft["evaluation"],
        candidate_quality=draft["quality"],
        reason_code="writer_candidate_failed",
        candidate_bindings=draft["bindings"],
        candidate_metadata={"writer_document": draft["writer_document"]},
    )
    envelope = {
        "semantic_engine": {
            "global_unit_response_contract": KEYED_GLOBAL_QUOTED_OMISSIONS
        }
    }
    assert repetition_followup(
        original["survey"], original["evaluation"], proposal, envelope
    )
    assert not repetition_followup(
        original["survey"], original["evaluation"], proposal, {}
    )
    assert not repetition_followup(
        original["survey"],
        original["evaluation"],
        replace(proposal, candidate=proposal.candidate + "\nTampered."),
        envelope,
    )
    assert not repetition_followup(
        original["survey"],
        original["evaluation"],
        replace(proposal, candidate_evaluation=journal["candidates"][2]["evaluation"]),
        envelope,
    )


def test_repetition_followup_can_remove_copy_but_preserves_unique_findings():
    """A successful candidate must retain at least one supported occurrence."""
    claim = {
        "section": "Methods",
        "claim_text": "The first method uses a coarse grid.",
        "evidence_ids": ["a"],
        "kind": "scientific",
        "status": "supported",
    }
    other = {**claim, "claim_text": "The second method uses an explicit grid."}
    before = {"faithfulness": {"details": {"assertions": [claim, claim, other]}}}
    after = {"faithfulness": {"details": {"assertions": [claim, other]}}}
    assert preserves_repetition_findings(before, after)
    after["faithfulness"]["details"]["assertions"].pop()
    assert not preserves_repetition_findings(before, after)


@pytest.mark.parametrize("status", ["supported", "unsupported"])
def test_omission_quote_distinguishes_extracted_claim_from_real_missing_claim(status):
    """Tie the finding to an exact clause regardless of NLI verdict."""
    known = "The coarse model uses stratified sampling."
    missing = "The fine model uses importance sampling."
    unit = {
        "coverage_key": "u1",
        "source_unit_id": "unit:0:0",
        "text": f"{known} {missing}",
        "extracted_claim_ids": ["claim_0"],
    }
    payload = {
        "global_unit_response_contract": KEYED_GLOBAL_QUOTED_OMISSIONS,
        "source_units": [unit],
        "assertions": [{"claim_id": "claim_0", "claim_text": known, "status": status}],
    }

    def reply(clause):
        return {
            "issues": [
                {
                    "code": "missing_assertion",
                    "anchor_ids": ["u1"],
                    "explanation": clause,
                }
            ]
        }

    progress = {}
    _restore_issue_anchors(reply(known), payload, progress)
    assert progress["omission_review_conflicts"] == [
        {"source_unit_id": "unit:0:0", "claim_ids": ["claim_0"]}
    ]
    assert payload["assertions"][0]["status"] == status
    progress = {}
    issue = reply(missing)
    _restore_issue_anchors(issue, payload, progress)
    assert issue["issues"][0]["anchor_ids"] == ["unit:unit:0:0"]
    assert "omission_review_conflicts" not in progress
    with pytest.raises(ValueError, match="evaluation_global_finding_identity_invalid"):
        _restore_issue_anchors(
            reply("A quote from a different unit."), payload, progress
        )


def test_host_organization_uses_exact_scope_and_keeps_scientific_nli():
    """Preserve mixed or altered scientific sentences and their original verdicts."""
    scope = (
        "本综述按“场景表示—光线积分—可微优化”解释 foundational NeRF representation。"
    )
    science = "本综述按“场景表示—光线积分”解释该方法可以提高准确率。"
    text = scope + science
    sample = {
        "id": "u1",
        "source_unit_id": "unit:0:0",
        "source_fragment_id": "fragment",
        "source_quote": text,
        "source_start": 0,
        "source_end": len(text),
        "section": "摘要",
        "evidence_ids": ["a"],
        "statements": [scope, science],
        "verdicts": [
            {"verdict": 0, "reason": "The cited paper does not make this statement."},
            {"verdict": 0, "reason": "Missing quantitative support."},
        ],
    }
    old, _ = canonical_assertions([sample])
    new, groups = canonical_assertions([sample], host_organization=True)
    assert [(a["kind"], a["status"]) for a in old] == [
        ("scientific", "unsupported")
    ] * 2
    assert [(a["kind"], a["status"]) for a in new] == [
        ("report_structure", "supported"),
        ("scientific", "unsupported"),
    ]
    assert groups == {"u1": ["claim_0", "claim_1"]}


def test_404_nerf_archived_omissions_already_had_negative_assertions():
    """Preserve the old review and verify the actual request's coverage evidence."""
    path = ARTIFACTS / "acceptance/workbench-vit-386-20261110/runs/a5d623063442.json"
    if not path.is_file():
        pytest.skip("local 404 archive is unavailable")
    archive = json.loads(path.read_bytes())
    node = next(
        node
        for node in archive["nodes"].values()
        if node.get("task_id") == "evaluation.initial"
        and node["kind"] == "llm"
        and node["input"]["messages"][0]["content"].startswith(
            "Review the complete frozen report"
        )
    )
    wire = json.loads(node["input"]["messages"][1]["content"])
    payload = unpack_global_input(wire)
    parsed = parse_native_global_response(node["output"], payload, {})
    assert [issue.code for issue in parsed.issues] == ["missing_assertion"] * 2
    units = {u["source_unit_id"]: u for u in payload["source_units"]}
    claims = {c["claim_id"]: c for c in payload["assertions"]}
    for issue, ids in zip(parsed.issues, (range(93, 97), range(115, 120)), strict=True):
        unit = units[issue.source_unit_id]
        assert all(
            f"claim_{i}" in unit["extracted_claim_ids"]
            and claims[f"claim_{i}"]["status"] == "unsupported"
            for i in ids
        )
    payload["global_unit_response_contract"] = KEYED_GLOBAL_QUOTED_OMISSIONS
    progress = {}
    _restore_issue_anchors(
        {
            "issues": [
                {
                    "code": "missing_assertion",
                    "anchor_ids": ["u15"],
                    "explanation": claims["claim_94"]["claim_text"].rstrip("。"),
                }
            ]
        },
        payload,
        progress,
    )
    assert progress["omission_review_conflicts"] == [
        {"source_unit_id": "unit:14:6599", "claim_ids": ["claim_94"]}
    ]


def test_404_nerf_extraction_context_keeps_the_actual_method_text():
    """Use the exact local paper and the new Live reader, with no HTTP or LLM."""
    pdf = ARTIFACTS / "corpus/raw/arxiv_2003.08934.pdf"
    if not pdf.is_file():
        pytest.skip("local NeRF source PDF is unavailable")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert digest == "c7fc70b38f84cb2e42890a69a7c9b9c26ded464b221e0355031004f24fb8e727"
    asset = RawPaperAsset(
        paper_id="arxiv:2003.08934",
        title="NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        abstract="A continuous neural field for view synthesis.",
        asset_hash=digest,
        pdf_path=pdf,
    )
    reader = TargetedFulltext(
        load_config(live_workbench=True).rag, prefer_method_sections=True
    )
    parsed = reader.parser.parse_with_quality(asset).record
    paper = {
        "title": asset.title,
        "quality_policy": "strict_auto_v2",
        "_extraction_question": "Explain foundational NeRF representation and volume rendering.",
        "chunks": [chunk.model_dump(mode="json") for chunk in parsed.chunks],
    }
    context, keys = _strategy_context(paper)
    assert len(keys) <= 6 and len(context) <= 12_000
    source = " ".join(context.split())
    assert "r(t) = o" in source and "positional_encoding" in source
    assert (
        re.search(r"strati.f?ied sampling", source, re.I)
        or "stratified sampling" in source
    )
    assert "inverse transform sampling" in source


@pytest.mark.asyncio
async def test_writing_selects_admitted_method_blocks_under_the_same_cap():
    """Reserve the method slot for an available original method passage."""
    ledger = {
        "p0:claim:0": {
            "evidence_id": "p0:claim:0",
            "paper_id": "p0",
            "text": "NeRF rendering methods mechanisms representations.",
        },
        "p1:claim:0": {
            "evidence_id": "p1:claim:0",
            "paper_id": "p1",
            "text": "The coarse weights are normalized.",
            "section": "5_2_hierarchical_volume_sampling",
            "chunk_key": "section:5_2_hierarchical_volume_sampling:0:abcd",
        },
    }
    methods = tuple(section for section in DEFAULT_SECTIONS if section.key == "methods")
    for prefer, expected in ((False, "p0:claim:0"), (True, "p1:claim:0")):
        selected = await EvidenceSelector(
            None,
            BudgetManager(max_tokens=2000),
            top_k_per_section=1,
            max_items=1,
            prefer_method_sections=prefer,
        ).select(
            "NeRF rendering methods",
            ledger,
            methods,
            max_tokens=1000,
            reuse_selected=True,
        )
        assert list(selected.selected_items) == [expected]
