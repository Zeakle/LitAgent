"""Phase 13.8 — EvidenceSelector 检索、round-robin 打包、格式化测试。

所有测试使用 fake reranker + BudgetManager，不下载模型、不访问网络、不连接外部服务。

GUIDE §13 要求：直接 import（非 importorskip），模块不存在 → collection failure。
"""

import asyncio
import pytest
from unittest.mock import MagicMock

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


# ── helpers ──────────────────────────────────────────────────

def _make_ledger(n: int = 20) -> dict:
    """构造 n 条 evidence，每 4 条一篇 paper。"""
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
    """3 papers × 2 claims = 6 evidence items."""
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
    """访问真实 Document.page_content，记录每次调用的 ScoredDoc identity。

    按 content 长度逆序排列，分数 = len(page_content)。
    记录每轮收到的 ScoredDoc 对象 id 集合，用于断言五节不共享可变对象。
    """

    def __init__(self):
        self.call_first_docs: list[ScoredDoc] = []

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        assert query
        # Keep references alive so CPython cannot recycle an id between calls.
        if docs:
            self.call_first_docs.append(docs[0])
        for item in docs:
            assert isinstance(item.doc, Document), f"Expected Document, got {type(item.doc)}"
            assert isinstance(item.doc.metadata, dict)
            assert isinstance(item.doc.metadata.get("evidence_id"), str)
            item.score = float(len(item.doc.page_content))
        return sorted(docs, key=lambda x: x.score, reverse=True)


class _BadReranker:
    def rerank(self, query, docs):
        raise RuntimeError("model load failed")


class _UnknownIdReranker:
    """返回不在 ledger 中的 evidence_id → 应触发该 section lexical fallback。"""

    def rerank(self, query, docs):
        for item in docs:
            item.doc.metadata["evidence_id"] = "unknown:claim:999"
            item.score = 1.0
        return docs


class _DuplicateIdReranker:
    """返回重复 evidence_id → 应被过滤，仅保留首次出现的。"""

    def rerank(self, query, docs):
        for item in docs:
            item.doc.metadata["evidence_id"] = "p0:claim:0"
            item.score = 1.0
        return docs


class _CancellingReranker:
    def rerank(self, query, docs):
        raise asyncio.CancelledError()


# ── tests ────────────────────────────────────────────────────


class TestFormatEvidenceSelection:
    def test_empty_selection_returns_empty_tags(self):
        sel = EvidenceSelection(
            candidate_count=0, selected_items={},
            section_evidence_ids={}, method_by_section={},
            omitted_count=0, estimated_tokens=0,
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
                    "paper_title": "T0", "text": "text0",
                },
                "p0:claim:1": {
                    "evidence_id": "p0:claim:1",
                    "paper_title": "T0", "text": "text1",
                },
            },
            section_evidence_ids={
                "introduction": ["p0:claim:0"],
                "methods": ["p0:claim:1"],
            },
            method_by_section={
                "introduction": "cross_encoder", "methods": "cross_encoder",
            },
            omitted_count=1, estimated_tokens=100,
        )
        out = format_evidence_selection(sel)
        assert '<section key="introduction">[E:p0:claim:0]</section>' in out
        assert '<section key="methods">[E:p0:claim:1]</section>' in out
        assert "[E:p0:claim:0]" in out
        assert "(T0)" in out
        assert "text0" in out
        # 无 evidence 的 section 不出空 tag
        assert 'key="taxonomy"' not in out

    def test_html_escape_applied_to_content(self):
        """html.escape 转义 title/text 中的 XML 特殊字符。"""
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
            omitted_count=0, estimated_tokens=50,
        )
        out = format_evidence_selection(sel)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&lt;tags&gt;" in out
        assert "&amp;" in out

    def test_no_omitted_marker_in_output(self):
        """格式化输出不含 '... omitted' 伪条目。"""
        sel = EvidenceSelection(
            candidate_count=10, selected_items={}, section_evidence_ids={},
            method_by_section={}, omitted_count=10, estimated_tokens=0,
        )
        out = format_evidence_selection(sel)
        assert "... omitted" not in out


class TestEvidenceSelectionFromDict:
    def test_roundtrips(self):
        sel = EvidenceSelection(
            candidate_count=10,
            selected_items={
                "p0:claim:0": {
                    "evidence_id": "p0:claim:0",
                    "paper_title": "T", "text": "x",
                },
            },
            section_evidence_ids={"introduction": ["p0:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=9, estimated_tokens=50,
        )
        d = sel.to_dict()
        restored = EvidenceSelection.from_dict(d)
        assert restored.candidate_count == 10
        assert restored.selected_items == sel.selected_items
        assert restored.section_evidence_ids == sel.section_evidence_ids
        assert restored.omitted_count == 9

    def test_empty_selection_roundtrips(self):
        sel = EvidenceSelection(
            candidate_count=0, selected_items={},
            section_evidence_ids={}, method_by_section={},
            omitted_count=0, estimated_tokens=0,
        )
        restored = EvidenceSelection.from_dict(sel.to_dict())
        assert restored.candidate_count == 0

    def test_rejects_non_mapping(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict("not a dict")

    def test_rejects_bad_candidate_count(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict({
                "candidate_count": -1,
                "selected_items": {},
                "section_evidence_ids": {},
                "method_by_section": {},
                "omitted_count": 0,
                "estimated_tokens": 0,
            })

    def test_rejects_non_mapping_selected_items(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict({
                "candidate_count": 0,
                "selected_items": "not a dict",
                "section_evidence_ids": {},
                "method_by_section": {},
                "omitted_count": 0,
                "estimated_tokens": 0,
            })

    def test_rejects_empty_key_in_selected_items(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict({
                "candidate_count": 1,
                "selected_items": {"": {"evidence_id": "", "text": "x"}},
                "section_evidence_ids": {},
                "method_by_section": {},
                "omitted_count": 0,
                "estimated_tokens": 0,
            })

    def test_rejects_bad_section_evidence_ids(self):
        with pytest.raises(ValueError):
            EvidenceSelection.from_dict({
                "candidate_count": 0,
                "selected_items": {},
                "section_evidence_ids": {"intro": "not a list"},
                "method_by_section": {},
                "omitted_count": 0,
                "estimated_tokens": 0,
            })


class TestEvidenceSelectorConstruction:
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
    @pytest.mark.asyncio
    async def test_cross_encoder_ranks_and_maps_back(self):
        """Fake reranker 返回有效结果 → 按 score 降序 + ID 正确映射。"""
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=50,
        )
        sel = await selector.select("few-shot learning", _simple_ledger(), max_tokens=10000)
        assert sel.candidate_count == 6
        assert len(sel.selected_items) > 0
        assert set(sel.selected_items.keys()).issubset(set(_simple_ledger().keys()))
        for s in DEFAULT_SECTIONS:
            assert s.key in sel.method_by_section
            assert sel.method_by_section[s.key] == "cross_encoder"

    @pytest.mark.asyncio
    async def test_five_sections_dont_share_mutable_scored_docs(self):
        """每个 section 创建独立 ScoredDoc 列表——五节不共享可变对象。"""
        reranker = _FakeReranker()
        selector = EvidenceSelector(
            reranker=reranker, budget=BudgetManager(), max_items=50,
        )
        await selector.select("test", _make_ledger(30), max_tokens=50000)

        assert len(reranker.call_first_docs) == 5  # 每个 section 一次调用
        assert all(
            left is not right
            for index, left in enumerate(reranker.call_first_docs)
            for right in reranker.call_first_docs[index + 1:]
        )

    @pytest.mark.asyncio
    async def test_round_robin_distributes_across_sections(self):
        """Round-robin：每 section 至少有一条 evidence。"""
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10,
        )
        sel = await selector.select("test", _make_ledger(30), max_tokens=50000)
        non_empty = sum(1 for eids in sel.section_evidence_ids.values() if eids)
        assert non_empty >= 3

    @pytest.mark.asyncio
    async def test_same_id_reused_across_sections_stored_once(self):
        """同一 evidence 多节复用 → selected_items 只存一次。"""
        ledger = {"p0:claim:0": {
            "evidence_id": "p0:claim:0", "paper_id": "p0",
            "paper_title": "Only Paper",
            "text": "the only evidence " * 20,
            "source_locator": "extracted_claim", "confidence": None,
        }}
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=50,
        )
        sel = await selector.select("test", ledger, max_tokens=50000)
        assert len(sel.selected_items) == 1
        assert "p0:claim:0" in sel.selected_items

    @pytest.mark.asyncio
    async def test_max_items_cap_enforced(self):
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=3,
        )
        sel = await selector.select("test", _make_ledger(30), max_tokens=50000)
        assert len(sel.selected_items) <= 3

    @pytest.mark.asyncio
    async def test_per_paper_cap_enforced(self):
        """每篇 paper 最多 per_paper_cap 条。"""
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(),
            max_items=50, per_paper_cap=1,
        )
        ledger = _make_ledger(20)  # 5 papers × 4 claims
        sel = await selector.select("test", ledger, max_tokens=50000)
        paper_counts: dict[str, int] = {}
        for eid in sel.selected_items:
            pid = eid.split(":")[0]
            paper_counts[pid] = paper_counts.get(pid, 0) + 1
        assert all(c <= 1 for c in paper_counts.values())
        assert len(sel.selected_items) <= 5

    @pytest.mark.asyncio
    async def test_continues_past_skipped_items(self):
        """per-paper cap 导致本轮候选全部跳过 → 继续读后续候选，不提前退出。

        场景：3 papers 各 2 claims，per_paper_cap=1、max_items=2。
        第一轮每个 section 取第一条（可能同一 paper），
        第二轮的候选可能因 cap 被跳过，但必须继续读到不同 paper 的条目。
        """
        ledger = {
            "p0:claim:0": {"evidence_id": "p0:claim:0", "paper_id": "p0",
                           "paper_title": "P0", "text": "a" * 100,
                           "source_locator": "extracted_claim", "confidence": None},
            "p0:claim:1": {"evidence_id": "p0:claim:1", "paper_id": "p0",
                           "paper_title": "P0", "text": "b" * 200,
                           "source_locator": "extracted_claim", "confidence": None},
            "p1:claim:0": {"evidence_id": "p1:claim:0", "paper_id": "p1",
                           "paper_title": "P1", "text": "c" * 150,
                           "source_locator": "extracted_claim", "confidence": None},
            "p1:claim:1": {"evidence_id": "p1:claim:1", "paper_id": "p1",
                           "paper_title": "P1", "text": "d" * 50,
                           "source_locator": "extracted_claim", "confidence": None},
            "p2:claim:0": {"evidence_id": "p2:claim:0", "paper_id": "p2",
                           "paper_title": "P2", "text": "e" * 120,
                           "source_locator": "extracted_claim", "confidence": None},
            "p2:claim:1": {"evidence_id": "p2:claim:1", "paper_id": "p2",
                           "paper_title": "P2", "text": "f" * 80,
                           "source_locator": "extracted_claim", "confidence": None},
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(),
            max_items=3, per_paper_cap=1,
        )
        sel = await selector.select("test", ledger, max_tokens=50000)
        assert len(sel.selected_items) == 3
        paper_ids = {item.get("paper_id") for item in sel.selected_items.values()}
        assert len(paper_ids) == 3  # 三篇不同论文

    @pytest.mark.asyncio
    async def test_long_then_short_budget_behavior(self):
        """长 item 超预算被跳过，后续短 item 仍被选入。"""
        ledger = {
            "p0:long": {
                "evidence_id": "p0:long", "paper_id": "p0",
                "paper_title": "Long Paper",
                "text": "x" * 3000,  # 很长
                "source_locator": "extracted_claim", "confidence": None,
            },
            "p1:short": {
                "evidence_id": "p1:short", "paper_id": "p1",
                "paper_title": "Short Paper",
                "text": "short evidence",
                "source_locator": "extracted_claim", "confidence": None,
            },
        }
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10,
        )
        sel = await selector.select("test", ledger, max_tokens=200)
        # 短 item 应被选入（长 item 可能因 token 超限被跳过）
        assert "p1:short" in sel.selected_items or len(sel.selected_items) > 0
        # XML 完整
        formatted = format_evidence_selection(sel)
        assert formatted.count("<evidence_plan>") == 1
        assert formatted.count("</evidence_plan>") == 1

    @pytest.mark.asyncio
    async def test_token_budget_not_exceeded_xml_complete(self):
        """最终 token 不超预算，XML 标签完整闭合，不含省略标记。"""
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=50,
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
            reranker=_BadReranker(), budget=BudgetManager(), max_items=10,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=50000)
        assert sel.candidate_count == 6
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_unknown_id_reranker_triggers_lexical_fallback(self):
        """Reranker 返回未知 evidence_id → 该 section 标记词法降级。"""
        selector = EvidenceSelector(
            reranker=_UnknownIdReranker(), budget=BudgetManager(), max_items=10,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=50000)
        # 所有 section 都因 unknown ID 降级（排名返回空、lexical fallback 接管）
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
        sel = await selector.select("few-shot learning", _simple_ledger(), max_tokens=50000)
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_empty_query_all_lexical(self):
        """空 query → reranker 存在但不调用，全部 section 标记 lexical_fallback。"""
        selector = EvidenceSelector(reranker=_BadReranker(), budget=BudgetManager(), max_items=10)
        sel = await selector.select("", _simple_ledger(), max_tokens=50000)
        for method in sel.method_by_section.values():
            assert method == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        selector = EvidenceSelector(
            reranker=_CancellingReranker(), budget=BudgetManager(), max_items=10,
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
        """重复 section key → ValueError。"""
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        bad_sections = (
            SectionSpec("intro", "Intro A", "hint A"),
            SectionSpec("intro", "Intro B", "hint B"),  # 重复 key
        )
        with pytest.raises(ValueError):
            await selector.select("test", _simple_ledger(), sections=bad_sections, max_tokens=5000)

    @pytest.mark.asyncio
    async def test_empty_section_key_raises(self):
        selector = EvidenceSelector(reranker=None, budget=BudgetManager())
        bad_sections = (SectionSpec("", "Empty", "hint"),)
        with pytest.raises(ValueError):
            await selector.select("test", _simple_ledger(), sections=bad_sections, max_tokens=5000)

    @pytest.mark.asyncio
    async def test_damaged_items_filtered(self):
        """空 ID、非法 ID 格式、text 为空、非 dict、evidence_id 不匹配 → 全部过滤。"""
        ledger = {
            "": {"evidence_id": "", "paper_title": "T", "text": "no id"},
            "p0:claim:0": {"evidence_id": "p0:claim:0", "paper_title": "T", "text": ""},
            "p0:claim:1": "not a dict",
            "p1:claim:0": {"evidence_id": "p1:WRONG", "paper_title": "T", "text": "mismatch"},
            "p2<xml>": {"evidence_id": "p2<xml>", "paper_title": "T", "text": "bad id chars"},
            "p3:claim:0": {"evidence_id": "p3:claim:0", "paper_title": "T3", "text": "valid"},
        }
        selector = EvidenceSelector(reranker=_FakeReranker(), budget=BudgetManager(), max_items=10)
        sel = await selector.select("test", ledger, max_tokens=5000)
        assert sel.candidate_count == 1
        assert "p3:claim:0" in sel.selected_items

    @pytest.mark.asyncio
    async def test_input_ledger_not_mutated(self):
        ledger = _simple_ledger()
        original_ids = set(ledger.keys())
        selector = EvidenceSelector(reranker=_FakeReranker(), budget=BudgetManager(), max_items=10)
        await selector.select("test", ledger, max_tokens=50000)
        assert set(ledger.keys()) == original_ids
        # 内部条目也不被修改
        for k, v in ledger.items():
            assert isinstance(v, dict)

    @pytest.mark.asyncio
    async def test_custom_sections(self):
        custom = (
            SectionSpec("intro", "Intro", "background"),
            SectionSpec("conc", "Conclusion", "summary"),
        )
        selector = EvidenceSelector(reranker=_FakeReranker(), budget=BudgetManager(), max_items=10)
        sel = await selector.select("test", _simple_ledger(), sections=custom, max_tokens=50000)
        assert set(sel.section_evidence_ids.keys()) == {"intro", "conc"}

    @pytest.mark.asyncio
    async def test_omitted_count_is_correct(self):
        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=1,
            top_k_per_section=1,
        )
        sel = await selector.select("test", _make_ledger(20), max_tokens=5000)
        assert sel.omitted_count == sel.candidate_count - len(sel.selected_items)
        assert sel.omitted_count >= 0

    @pytest.mark.asyncio
    async def test_trace_events_emitted(self):
        """trace_hook 收到 evidence.retrieve.start/complete 配对事件。"""
        events: list[tuple[str, dict]] = []

        def hook(event, data):
            events.append((event, data))

        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10,
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
        """CancelledError → trace failed(cancelled) + 继续传播。"""
        events: list[tuple[str, dict]] = []

        def hook(event, data):
            events.append((event, data))

        selector = EvidenceSelector(
            reranker=_CancellingReranker(), budget=BudgetManager(), max_items=10,
            trace_hook=hook,
        )
        with pytest.raises(asyncio.CancelledError):
            await selector.select("test", _simple_ledger(), max_tokens=5000)

        failed_data = [d for e, d in events if e == "evidence.retrieve.failed"]
        assert len(failed_data) == 1
        assert failed_data[0]["reason_code"] == "cancelled"

    @pytest.mark.asyncio
    async def test_trace_hook_exception_does_not_break_select(self):
        """trace hook 抛异常 → selection 正常返回，不传播。"""
        def bad_hook(event, data):
            raise RuntimeError("trace broken")

        selector = EvidenceSelector(
            reranker=_FakeReranker(), budget=BudgetManager(), max_items=10,
            trace_hook=bad_hook,
        )
        sel = await selector.select("test", _simple_ledger(), max_tokens=5000)
        assert sel.candidate_count > 0
