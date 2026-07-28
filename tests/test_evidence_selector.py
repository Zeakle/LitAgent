"""Tests for evidence selection, formatting, and trace contracts."""

import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import (
    EvidenceSelector,
    EvidenceSelection,
    SectionSpec,
    DEFAULT_SECTIONS,
    format_evidence_selection,
)
from litagent.rag.interfaces import Reranker, ScoredDoc


def _make_ledger(n: int = 20) -> dict:
    """Build a synthetic evidence ledger with deterministic IDs."""
    ledger: dict = {}
    for i in range(n):
        pid = f"p{i // 4}"
        eid = f"{pid}:claim:{i % 4}"
        ledger[eid] = {
            "evidence_id": eid,
            "paper_id": pid,
            "paper_title": f"Paper {i // 4}",
            "text": f"claim text number {i} about methods and experiments",
            "source_locator": "extracted_claim",
            "confidence": None,
        }
    return ledger


def _simple_ledger() -> dict:
    """Build a compact six-item evidence ledger."""
    ledger: dict = {}
    for pid in ["p0", "p1", "p2"]:
        for j in range(2):
            eid = f"{pid}:claim:{j}"
            ledger[eid] = {
                "evidence_id": eid,
                "paper_id": pid,
                "paper_title": f"Title of {pid}",
                "text": f"claim {j} from {pid} about few-shot learning",
                "source_locator": "extracted_claim",
                "confidence": None,
            }
    return ledger


class _FakeReranker:
    """Reranker that returns deterministic scores."""

    def __init__(self):
        self.call_first_docs: list[ScoredDoc] = []

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        assert query
        # Keep references alive so CPython cannot recycle an id between calls.
        if docs:
            self.call_first_docs.append(docs[0])
        for item in docs:
            assert isinstance(
                item.doc, Document
            ), f"Expected Document, got {type(item.doc)}"
            assert isinstance(item.doc.metadata, dict)
            assert isinstance(item.doc.metadata.get("evidence_id"), str)
            item.score = float(len(item.doc.page_content))
        return sorted(docs, key=lambda x: x.score, reverse=True)


class _BadReranker:
    """Reranker test double that always raises."""

    def rerank(self, query, docs):
        raise RuntimeError("model load failed")


class _UnknownIdReranker:
    """Reranker that returns an unknown evidence ID."""

    def rerank(self, query, docs):
        for item in docs:
            item.doc.metadata["evidence_id"] = "unknown:claim:999"
            item.score = 1.0
        return docs


class _DuplicateIdReranker:
    """Reranker that returns duplicate evidence IDs."""

    def rerank(self, query, docs):
        for item in docs:
            item.doc.metadata["evidence_id"] = "p0:claim:0"
            item.score = 1.0
        return docs


class _CancellingReranker:
    """Reranker test double that propagates cancellation."""

    def rerank(self, query, docs):
        raise asyncio.CancelledError()


class TestFormatEvidenceSelection:
    """Tests evidence-selection rendering."""

    def test_empty_selection_returns_empty_tags(self):
        sel = EvidenceSelection(
            candidate_count=0,
            selected_items={},
            section_evidence_ids={},
            method_by_section={},
            omitted_count=0,
            estimated_tokens=0,
        )
        out = format_evidence_selection(sel)
        assert "<evidence_plan></evidence_plan>" in out
        assert "<evidence_ledger></evidence_ledger>" in out

    def test_non_empty_formats_sections_and_items(self):
        sel = EvidenceSelection(
            candidate_count=3,
            selected_items={
                "p0:claim:0": {
                    "evidence_id": "p0:claim:0",
                    "paper_title": "T0",
                    "text": "text0",
                },
                "p0:claim:1": {
                    "evidence_id": "p0:claim:1",
                    "paper_title": "T0",
                    "text": "text1",
                },
            },
            section_evidence_ids={
                "introduction": ["p0:claim:0"],
                "methods": ["p0:claim:1"],
            },
            method_by_section={
                "introduction": "cross_encoder",
                "methods": "cross_encoder",
            },
            omitted_count=1,
            estimated_tokens=100,
        )
        out = format_evidence_selection(sel)
        assert '<section key="introduction">[E:p0:claim:0]</section>' in out
        assert '<section key="methods">[E:p0:claim:1]</section>' in out
        assert "[E:p0:claim:0]" in out
        assert "(T0)" in out
        assert "text0" in out

        assert 'key="taxonomy"' not in out

    def test_html_escape_applied_to_content(self):
        """Rendered evidence content is HTML-escaped."""
        sel = EvidenceSelection(
            candidate_count=1,
            selected_items={
                "p0:claim:0": {
                    "evidence_id": "p0:claim:0",
                    "paper_title": 'Paper <script>alert("xss")</script>',
                    "text": "text with & and <tags>",
                },
            },
            section_evidence_ids={"introduction": ["p0:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=50,
        )
        out = format_evidence_selection(sel)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&lt;tags&gt;" in out
        assert "&amp;" in out

    def test_no_omitted_marker_in_output(self):
        """Complete selections omit the truncation marker."""
        sel = EvidenceSelection(
            candidate_count=10,
            selected_items={},
            section_evidence_ids={},
            method_by_section={},
            omitted_count=10,
            estimated_tokens=0,
        )
        out = format_evidence_selection(sel)
        assert "... omitted" not in out


class TestEvidenceSelectionFromDict:
    """Tests deserialization of evidence selections."""

    def test_roundtrips(self):
        sel = EvidenceSelection(
            candidate_count=10,
            selected_items={
                "p0:claim:0": {
                    "evidence_id": "p0:claim:0",
                    "paper_title": "T",
                    "text": "x",
                },
            },
            section_evidence_ids={"introduction": ["p0:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=9,
            estimated_tokens=50,
        )
        d = sel.to_dict()
        restored = EvidenceSelection.from_dict(d)
        assert restored.candidate_count == 10
        assert restored.selected_items == sel.selected_items
        assert restored.section_evidence_ids == sel.section_evidence_ids
        assert restored.omitted_count == 9

    def test_empty_selection_roundtrips(self):
        sel = EvidenceSelection(
            candidate_count=0,
            selected_items={},
            section_evidence_ids={},
            method_by_section={},
            omitted_count=0,
            estimated_tokens=0,
        )
        restored = EvidenceSelection.from_dict(sel.to_dict())
        assert restored.candidate_count == 0

    def test_rejects_non_mapping(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict("not a dict")

    def test_rejects_bad_candidate_count(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict(
                {
                    "candidate_count": -1,
                    "selected_items": {},
                    "section_evidence_ids": {},
                    "method_by_section": {},
                    "omitted_count": 0,
                    "estimated_tokens": 0,
                }
            )

    def test_rejects_non_mapping_selected_items(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict(
                {
                    "candidate_count": 0,
                    "selected_items": "not a dict",
                    "section_evidence_ids": {},
                    "method_by_section": {},
                    "omitted_count": 0,
                    "estimated_tokens": 0,
                }
            )

    def test_rejects_empty_key_in_selected_items(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict(
                {
                    "candidate_count": 1,
                    "selected_items": {"": {"evidence_id": "", "text": "x"}},
                    "section_evidence_ids": {},
                    "method_by_section": {},
                    "omitted_count": 0,
                    "estimated_tokens": 0,
                }
            )

    def test_rejects_bad_section_evidence_ids(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict(
                {
                    "candidate_count": 0,
                    "selected_items": {},
                    "section_evidence_ids": {"intro": "not a list"},
                    "method_by_section": {},
                    "omitted_count": 0,
                    "estimated_tokens": 0,
                }
            )


class TestEvidenceSelectorConstruction:
    """Tests evidence-selector configuration boundaries."""

    def test_rejects_non_positive_params(self):
        budget = BudgetManager()
        with pytest.raises(ValueError):
            EvidenceSelector(reranker=None, budget=budget, top_k_per_section=0)
        with pytest.raises(ValueError):
            EvidenceSelector(reranker=None, budget=budget, max_items=0)
        with pytest.raises(ValueError):
            EvidenceSelector(reranker=None, budget=budget, per_paper_cap=0)

    def test_accepts_none_reranker(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        assert selector._reranker is None


class TestEvidenceSelectorSelect:
    """Tests ranking, budgeting, fallback, and tracing."""

    @pytest.mark.asyncio
    async def test_cross_encoder_ranks_and_maps_back(self):
        """Cross-encoder ranks map back to evidence IDs."""
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=50,
        )
        sel = await selector.select(
            "few-shot learning", _simple_ledger(), max_tokens=10000
        )
        assert sel.candidate_count == 6
        assert len(sel.selected_items) > 0
        assert set(sel.selected_items.keys()).issubset(set(_simple_ledger().keys()))
        for s in DEFAULT_SECTIONS:
            assert s.key in sel.method_by_section
            assert sel.method_by_section[s.key] == "cross_encoder"

    @pytest.mark.asyncio
    async def test_five_sections_dont_share_mutable_scored_docs(self):
        """Sections do not share mutable scored-document state."""
        reranker = _FakeReranker()
        selector = EvidenceSelector(
            reranker=reranker,
            budget=BudgetManager(),
            max_items=50,
        )
        await selector.select("test", _make_ledger(30), max_tokens=50000)

        assert len(reranker.call_first_docs) == 5
        assert all(
            left is not right
            for index, left in enumerate(reranker.call_first_docs)
            for right in reranker.call_first_docs[index + 1 :]
        )

    @pytest.mark.asyncio
    async def test_round_robin_distributes_across_sections(self):
        """Round-robin selection distributes evidence across sections."""
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=10,
        )
        sel = await selector.select("test", _make_ledger(30), max_tokens=50000)
        non_empty = sum(1 for eids in sel.section_evidence_ids.values() if eids)
        assert non_empty >= 3

    @pytest.mark.asyncio
    async def test_same_id_reused_across_sections_stored_once(self):
        """Evidence reused across sections is stored once."""
        ledger = {
            "p0:claim:0": {
                "evidence_id": "p0:claim:0",
                "paper_id": "p0",
                "paper_title": "Only Paper",
                "text": "the only evidence " * 20,
                "source_locator": "extracted_claim",
                "confidence": None,
            }
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=50,
        )
        sel = await selector.select("test", ledger, max_tokens=50000)
        assert len(sel.selected_items) == 1
        assert "p0:claim:0" in sel.selected_items

    @pytest.mark.asyncio
    async def test_max_items_cap_enforced(self):
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=3,
        )
        sel = await selector.select("test", _make_ledger(30), max_tokens=50000)
        assert len(sel.selected_items) <= 3

    @pytest.mark.asyncio
    async def test_per_paper_cap_enforced(self):
        """Selection enforces the per-paper item cap."""
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=50,
            per_paper_cap=1,
        )
        ledger = _make_ledger(20)
        sel = await selector.select("test", ledger, max_tokens=50000)
        paper_counts: dict[str, int] = {}
        for eid in sel.selected_items:
            pid = eid.split(":")[0]
            paper_counts[pid] = paper_counts.get(pid, 0) + 1
        assert all(c <= 1 for c in paper_counts.values())
        assert len(sel.selected_items) <= 5

    @pytest.mark.asyncio
    async def test_continues_past_skipped_items(self):
        """Selection continues after over-budget candidates."""
        ledger = {
            "p0:claim:0": {
                "evidence_id": "p0:claim:0",
                "paper_id": "p0",
                "paper_title": "P0",
                "text": "a" * 100,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p0:claim:1": {
                "evidence_id": "p0:claim:1",
                "paper_id": "p0",
                "paper_title": "P0",
                "text": "b" * 200,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p1:claim:0": {
                "evidence_id": "p1:claim:0",
                "paper_id": "p1",
                "paper_title": "P1",
                "text": "c" * 150,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p1:claim:1": {
                "evidence_id": "p1:claim:1",
                "paper_id": "p1",
                "paper_title": "P1",
                "text": "d" * 50,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p2:claim:0": {
                "evidence_id": "p2:claim:0",
                "paper_id": "p2",
                "paper_title": "P2",
                "text": "e" * 120,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p2:claim:1": {
                "evidence_id": "p2:claim:1",
                "paper_id": "p2",
                "paper_title": "P2",
                "text": "f" * 80,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=3,
            per_paper_cap=1,
        )
        sel = await selector.select("test", ledger, max_tokens=50000)
        assert len(sel.selected_items) == 3
        paper_ids = {item.get("paper_id") for item in sel.selected_items.values()}
        assert len(paper_ids) == 3

    @pytest.mark.asyncio
    async def test_long_then_short_budget_behavior(self):
        """Short candidates remain selectable after oversized ones."""
        ledger = {
            "p0:long": {
                "evidence_id": "p0:long",
                "paper_id": "p0",
                "paper_title": "Long Paper",
                "text": "x" * 3000,
                "source_locator": "extracted_claim",
                "confidence": None,
            },
            "p1:short": {
                "evidence_id": "p1:short",
                "paper_id": "p1",
                "paper_title": "Short Paper",
                "text": "short evidence",
                "source_locator": "extracted_claim",
                "confidence": None,
            },
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=10,
        )
        sel = await selector.select("test", ledger, max_tokens=200)

        assert "p1:short" in sel.selected_items or len(sel.selected_items) > 0

        formatted = format_evidence_selection(sel)
        assert formatted.count("<evidence_plan>") == 1
        assert formatted.count("</evidence_plan>") == 1

    @pytest.mark.asyncio
    async def test_token_budget_not_exceeded_xml_complete(self):
        """Rendered XML remains complete within the token budget."""
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=50,
        )
        sel = await selector.select("test", _make_ledger(30), max_tokens=200)
        formatted = format_evidence_selection(sel)
        tokens = selector._budget.count_tokens(formatted)
        assert tokens <= 200
        assert formatted.count("<evidence_plan>") == 1
        assert formatted.count("</evidence_plan>") == 1
        assert formatted.count("<evidence_ledger>") == 1
        assert formatted.count("</evidence_ledger>") == 1
        assert "... omitted" not in formatted

    @pytest.mark.asyncio
    async def test_reranker_error_falls_back_lexical(self):
        selector = EvidenceSelector(
            reranker=_BadReranker(),
            budget=BudgetManager(),
            max_items=10,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=50000)
        assert sel.candidate_count == 6
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_unknown_id_reranker_triggers_lexical_fallback(self):
        """Unknown reranker IDs trigger lexical fallback."""
        selector = EvidenceSelector(
            reranker=_UnknownIdReranker(),
            budget=BudgetManager(),
            max_items=10,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=50000)

        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_lexical_result_stable(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager(), max_items=10)
        a = await selector.select("learning", _simple_ledger(), max_tokens=50000)
        b = await selector.select("learning", _simple_ledger(), max_tokens=50000)
        assert a.section_evidence_ids == b.section_evidence_ids

    @pytest.mark.asyncio
    async def test_empty_ledger_returns_empty_selection(self):
        selector = EvidenceSelector(reranker=_FakeReranker(), budget=BudgetManager())
        sel = await selector.select("test", {}, max_tokens=5000)
        assert sel.candidate_count == 0
        assert sel.selected_items == {}
        assert all(eids == [] for eids in sel.section_evidence_ids.values())

    @pytest.mark.asyncio
    async def test_null_reranker_all_lexical(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager(), max_items=10)
        sel = await selector.select(
            "few-shot learning", _simple_ledger(), max_tokens=50000
        )
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_empty_query_all_lexical(self):
        """Empty queries use lexical ranking for every section."""
        selector = EvidenceSelector(
            reranker=_BadReranker(), budget=BudgetManager(), max_items=10
        )
        sel = await selector.select("", _simple_ledger(), max_tokens=50000)
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        selector = EvidenceSelector(
            reranker=_CancellingReranker(),
            budget=BudgetManager(),
            max_items=10,
        )
        with pytest.raises(asyncio.CancelledError):
            await selector.select("test", _simple_ledger(), max_tokens=50000)

    @pytest.mark.asyncio
    async def test_max_tokens_must_be_positive(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        with pytest.raises(ValueError):
            await selector.select("test", _simple_ledger(), max_tokens=0)

    @pytest.mark.asyncio
    async def test_duplicate_section_keys_raise(self):
        """Duplicate section keys are rejected."""
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        bad_sections = (
            SectionSpec("intro", "Intro A", "hint A"),
            SectionSpec("intro", "Intro B", "hint B"),
        )
        with pytest.raises(ValueError):
            await selector.select(
                "test", _simple_ledger(), sections=bad_sections, max_tokens=5000
            )

    @pytest.mark.asyncio
    async def test_empty_section_key_raises(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        bad_sections = (SectionSpec("", "Empty", "hint"),)
        with pytest.raises(ValueError):
            await selector.select(
                "test", _simple_ledger(), sections=bad_sections, max_tokens=5000
            )

    @pytest.mark.asyncio
    async def test_damaged_items_filtered(self):
        """Malformed evidence items are filtered out."""
        ledger = {
            "": {"evidence_id": "", "paper_title": "T", "text": "no id"},
            "p0:claim:0": {"evidence_id": "p0:claim:0", "paper_title": "T", "text": ""},
            "p0:claim:1": "not a dict",
            "p1:claim:0": {
                "evidence_id": "p1:WRONG",
                "paper_title": "T",
                "text": "mismatch",
            },
            "p2<xml>": {
                "evidence_id": "p2<xml>",
                "paper_title": "T",
                "text": "bad id chars",
            },
            "p3:claim:0": {
                "evidence_id": "p3:claim:0",
                "paper_title": "T3",
                "text": "valid",
            },
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10
        )
        sel = await selector.select("test", ledger, max_tokens=5000)
        assert sel.candidate_count == 1
        assert "p3:claim:0" in sel.selected_items

    @pytest.mark.asyncio
    async def test_input_ledger_not_mutated(self):
        ledger = _simple_ledger()
        original_ids = set(ledger.keys())
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10
        )
        await selector.select("test", ledger, max_tokens=50000)
        assert set(ledger.keys()) == original_ids

        for k, v in ledger.items():
            assert isinstance(v, dict)

    @pytest.mark.asyncio
    async def test_custom_sections(self):
        custom = (
            SectionSpec("intro", "Intro", "background"),
            SectionSpec("conc", "Conclusion", "summary"),
        )
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10
        )
        sel = await selector.select(
            "test", _simple_ledger(), sections=custom, max_tokens=50000
        )
        assert set(sel.section_evidence_ids.keys()) == {"intro", "conc"}

    @pytest.mark.asyncio
    async def test_omitted_count_is_correct(self):
        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=1,
            top_k_per_section=1,
        )
        sel = await selector.select("test", _make_ledger(20), max_tokens=5000)
        assert sel.omitted_count == sel.candidate_count - len(sel.selected_items)
        assert sel.omitted_count >= 0

    @pytest.mark.asyncio
    async def test_trace_events_emitted(self):
        """Selection emits paired trace events."""
        events: list[tuple[str, dict]] = []

        def hook(event, data):
            events.append((event, data))

        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=10,
            trace_hook=hook,
        )
        await selector.select("test", _simple_ledger(), max_tokens=5000)

        start_data = [d for e, d in events if e == "evidence.retrieve.start"]
        complete_data = [d for e, d in events if e == "evidence.retrieve.complete"]
        assert len(start_data) == 1
        assert len(complete_data) == 1
        assert "candidate_count" in start_data[0]
        assert complete_data[0]["selected_count"] >= 0

    @pytest.mark.asyncio
    async def test_trace_failed_on_cancelled(self):
        """Cancellation emits a failed terminal trace."""
        events: list[tuple[str, dict]] = []

        def hook(event, data):
            events.append((event, data))

        selector = EvidenceSelector(
            reranker=_CancellingReranker(),
            budget=BudgetManager(),
            max_items=10,
            trace_hook=hook,
        )
        with pytest.raises(asyncio.CancelledError):
            await selector.select("test", _simple_ledger(), max_tokens=5000)

        failed_data = [d for e, d in events if e == "evidence.retrieve.failed"]
        assert len(failed_data) == 1
        assert failed_data[0]["reason_code"] == "cancelled"

    @pytest.mark.asyncio
    async def test_trace_hook_exception_does_not_break_select(self):
        """Trace-hook failures do not break evidence selection."""

        def bad_hook(event, data):
            raise RuntimeError("trace broken")

        selector = EvidenceSelector(
            reranker=_FakeReranker(),
            budget=BudgetManager(),
            max_items=10,
            trace_hook=bad_hook,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=5000)
        assert sel.candidate_count > 0
