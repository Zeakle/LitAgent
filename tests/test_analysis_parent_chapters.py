"""Keep nested report headings from erasing academic claim ownership."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_batched_evaluation import materials
from test_evaluation_wire_contract import AnalysisTransport

from litagent.body_quality import ROLES, analysis_claim_ids_by_section
from litagent.config import EvalConfig
from litagent.eval.scoped import _section, evaluate_scoped
from litagent.report_render import build_report_source_units
from litagent.review_structure import structure_sections
from litagent.workbench import RunOptions


def test_received_report_keeps_all_science_in_its_parent_chapter():
    """Replay actual offsets without changing citations, prose or old decisions."""
    fixture = json.loads(
        Path("tests/fixtures/analysis-subheadings-real.json").read_text()
    )
    original = copy.deepcopy(fixture)
    text = fixture["survey"]
    assert hashlib.sha256(text.encode()).hexdigest() == fixture["source_sha256"]
    units = build_report_source_units(text)
    structure = [
        {
            "source_unit_id": unit.unit_id,
            "section": _section(text, unit.start, academic_parent=True),
        }
        for unit in units
    ]
    claims = [
        {
            **claim,
            "section": _section(text, claim["source_start"], academic_parent=True),
        }
        for claim in fixture["assertions"]
    ]
    grouped = analysis_claim_ids_by_section(claims, structure)
    assert all(grouped[key] for key in ROLES)
    assert sum(map(len, grouped.values())) == sum(
        c["kind"] == "scientific" for c in claims
    )
    for claim in claims:
        assert (
            text[claim["source_start"] : claim["source_end"]] == claim["source_quote"]
        )
    old = fixture["original_analysis"]["sections"]
    assert all(
        not row["host_validation"]["allowed_claim_ids"]
        for row in old
        if row["section_key"] not in {"abstract", "conclusion"}
    )
    # This replay establishes ownership only, never scientific support.
    assert fixture["scientific_outcome"] == "not_verified"
    assert fixture == original


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_nested_headings_tables_and_fenced_fake_titles_keep_parent(language):
    """Follow real Markdown hierarchy and stop ownership at the appendix."""
    titles = dict(structure_sections("academic_v2", language))
    text = (
        f"# Report\n\n## {titles['methods']}\n\n### Shared heading\n\n"
        "A mechanism. [E:a]\n\n#### Detail\n\n"
        "| Finding | Evidence |\n| --- | --- |\n| A result | [E:b] |\n\n"
        f"```markdown\n## {titles['conclusion']}\n```\n\n"
        "A qualified result. [E:c]\n\n"
        f"## {titles['open_problems']}\n\n### Shared heading\n\n"
        "A limitation. [E:d]\n\n## Appendix A: Search\n\n"
        "### Shared heading\n\nA scientific premise. [E:e]\n"
    )
    for quote in ["A mechanism.", "A result", "A qualified result."]:
        start = text.index(quote)
        assert _section(text, start, academic_parent=True) == titles["methods"]
    assert _section(text, text.index("A mechanism.")) == "Shared heading"
    assert (
        _section(text, text.index("A limitation."), academic_parent=True)
        == titles["open_problems"]
    )
    assert (
        _section(text, text.index("A scientific premise."), academic_parent=True)
        == "Appendix A: Search"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "unsupported"])
async def test_real_evaluator_sends_parent_ownership_and_keeps_negative_gates(fault):
    """Replace transport only; inventory, batching and global consumers stay real."""
    _, context = materials(1)
    context["body_quality"] = "analysis_v1"
    titles = dict(structure_sections("academic_v2", "en"))
    survey = "# Report\n\n" + "\n\n".join(
        f"## {titles[key]}\n\n### Shared heading\n\n" "ViT uses image patches. [E:a0]"
        for key in ROLES
    )
    llm = AnalysisTransport(fault)
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    payload = llm.calls[-1]
    grouped = payload["analysis_claim_ids_by_section"]
    assert all(grouped[key] for key in ROLES)
    assert len({cid for ids in grouped.values() for cid in ids}) == 7
    assert all(a["section"] in titles.values() for a in result["assertions"])
    assert result["coverage"]["complete"]
    # Transport makes no substantive analytical judgment, so ownership alone
    # must not make this synthetic report deliverable.
    assert result["analysis_check"]["status"] == "failed"
    if fault:
        assert result["unsupported_count"] >= 1
        assert result["score"] < 1.0


def test_chapter_contract_is_bound_only_to_analysis_identity(monkeypatch):
    """A changed ownership contract invalidates analysis without changing legacy."""
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    current_before, legacy_before = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "ANALYSIS_CHAPTER_CONTRACT", "different-contract")
    assert current.fingerprint() != current_before
    assert legacy.fingerprint() == legacy_before
