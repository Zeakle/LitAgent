import pytest
from litagent.context.budget import BudgetManager, CharBasedCounter, TokenCounter
from litagent.context.pipeline import ContextPipeline, ContextLayer
from litagent.context.compressor import TierCompressor, PaperInfo
from litagent.context.templates import (
    wrap_xml, build_system_prompt,
)


class TestCharBasedCounter:
    def test_empty_string(self):
        counter = CharBasedCounter()
        assert counter.count("") == 0

    def test_short_string(self):
        counter = CharBasedCounter()
        assert counter.count("hi") == 1

    def test_normal_string(self):
        counter = CharBasedCounter()
        text = "a" * 100
        assert counter.count(text) == 25

    def test_chars_per_token(self):
        counter = CharBasedCounter()
        assert counter.chars_per_token() == 4


class TestBudgetManager:
    def test_count_tokens(self):
        bm = BudgetManager(max_tokens=1000)
        assert bm.count_tokens("hello world!") == 3

    def test_needs_compact_below_threshold(self):
        bm = BudgetManager(max_tokens=1000, compact_threshold=0.7)
        assert bm.needs_compact(600) is False

    def test_needs_compact_above_threshold(self):
        bm = BudgetManager(max_tokens=1000, compact_threshold=0.7)
        assert bm.needs_compact(800) is True

    def test_remaining(self):
        bm = BudgetManager(max_tokens=1000)
        assert bm.remaining(300) == 700
        assert bm.remaining(1500) == 0

    def test_truncate_short_text(self):
        bm = BudgetManager(max_tokens=1000)
        text = "short text"
        assert bm.truncate(text, 100) == text

    def test_truncate_at_sentence_boundary(self):
        bm = BudgetManager(max_tokens=1000)
        text = "First sentence. Second sentence. Third sentence. " * 20
        result = bm.truncate(text, 10)
        assert len(result) < len(text)

    def test_truncate_at_newline_boundary(self):
        bm = BudgetManager(max_tokens=1000)
        text = "Line one\nLine two\nLine three\n" * 20
        result = bm.truncate(text, 10)
        assert len(result) < len(text)

    def test_custom_counter(self):
        class DoubleCounter(TokenCounter):
            def count(self, text: str) -> int:
                return len(text) // 2
            def chars_per_token(self) -> int:
                return 2
        bm = BudgetManager(max_tokens=100, counter=DoubleCounter())
        assert bm.count_tokens("abcdef") == 3


def _async_builder(text: str):
    async def builder(state):
        return text
    return builder


class TestContextPipeline:
    @pytest.mark.asyncio
    async def test_single_layer(self):
        budget = BudgetManager(max_tokens=1000)
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("test", 0, 500, _async_builder("hello world")))
        text, tokens = await pipeline.build({})
        assert "hello world" in text
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        budget = BudgetManager(max_tokens=1000)
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("low", 2, 500, _async_builder("low priority")))
        pipeline.add_layer(ContextLayer("high", 0, 500, _async_builder("high priority")))
        text, _ = await pipeline.build({})
        assert text.index("high priority") < text.index("low priority")

    @pytest.mark.asyncio
    async def test_budget_enforcement(self):
        budget = BudgetManager(max_tokens=10)
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("first", 0, 5, _async_builder("short")))
        pipeline.add_layer(ContextLayer("second", 1, 5, _async_builder("a" * 200)))
        text, tokens = await pipeline.build({})
        assert tokens <= 10

    @pytest.mark.asyncio
    async def test_empty_builder_skipped(self):
        budget = BudgetManager(max_tokens=1000)
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("empty", 0, 500, _async_builder("")))
        pipeline.add_layer(ContextLayer("filled", 1, 500, _async_builder("content")))
        text, _ = await pipeline.build({})
        assert "content" in text

    @pytest.mark.asyncio
    async def test_builder_exception_skipped(self):
        async def failing_builder(state):
            raise RuntimeError("boom")
        budget = BudgetManager(max_tokens=1000)
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("bad", 0, 500, failing_builder))
        pipeline.add_layer(ContextLayer("good", 1, 500, _async_builder("ok")))
        text, _ = await pipeline.build({})
        assert "ok" in text

    def test_chain_add(self):
        budget = BudgetManager(max_tokens=1000)
        pipeline = ContextPipeline(budget)
        result = pipeline.add_layer(ContextLayer("a", 0, 100, _async_builder("a")))
        assert result is pipeline


class TestTierCompressor:
    def _make_paper(self, tier: int, title: str = "Paper") -> PaperInfo:
        return PaperInfo(
            paper_id="p1", title=title, tier=tier,
            claims=["claim A", "claim B"],
            metrics={"accuracy": "93.2%"},
            summary="A study on X.",
            tags=["few-shot", "CV"],
            extraction={"method": "ProtoNet", "backbone": "ResNet-12"},
        )

    def test_tier1_full_extraction(self):
        comp = TierCompressor()
        result = comp.compress([self._make_paper(1, "Seminal Paper")])
        assert "Seminal Paper" in result
        assert "ProtoNet" in result
        assert "claim A" in result
        assert "93.2%" in result

    def test_tier2_claims_metrics_only(self):
        comp = TierCompressor()
        result = comp.compress([self._make_paper(2, "High Cite")])
        assert "High Cite" in result
        assert "claim A" in result
        assert "93.2%" in result

    def test_tier3_summary_only(self):
        comp = TierCompressor()
        result = comp.compress([self._make_paper(3, "General")])
        assert "General" in result
        assert "A study on X." in result
        assert "few-shot" in result

    def test_mixed_tiers(self):
        comp = TierCompressor()
        papers = [
            self._make_paper(1, "Tier1"),
            self._make_paper(2, "Tier2"),
            self._make_paper(3, "Tier3"),
        ]
        result = comp.compress(papers)
        assert "Seminal Papers" in result
        assert "High-Citation Papers" in result
        assert "General Papers" in result

    def test_empty_list(self):
        comp = TierCompressor()
        assert comp.compress([]) == ""

    def test_invalid_tier_defaults_to_3(self):
        comp = TierCompressor()
        paper = self._make_paper(99, "Unknown Tier")
        result = comp.compress([paper])
        assert "General Papers" in result

    def test_paper_with_empty_fields(self):
        comp = TierCompressor()
        paper = PaperInfo(paper_id="p1", title="Minimal", tier=1)
        result = comp.compress([paper])
        assert "Minimal" in result


class TestTemplates:
    def test_wrap_xml(self):
        result = wrap_xml("role", "You are a search agent")
        assert result == "<role>\nYou are a search agent\n</role>"

    def test_wrap_xml_with_attrs(self):
        result = wrap_xml("papers", "content", attrs={"source": "rag"})
        assert '<papers source="rag">' in result

    def test_wrap_xml_empty_content(self):
        result = wrap_xml("role", "")
        assert result == ""

    def test_wrap_xml_whitespace_only(self):
        result = wrap_xml("role", "   ")
        assert result == ""

    def test_build_system_prompt_full(self):
        result = build_system_prompt(
            role="Search Agent",
            instructions="Find papers",
            tools="tool list",
            context="memory data",
            constraints="max 10 results",
        )
        assert "<role>" in result
        assert "<instructions>" in result
        assert "<available_tools>" in result
        assert "<context>" in result
        assert "<constraints>" in result

    def test_build_system_prompt_minimal(self):
        result = build_system_prompt(role="Agent", instructions="Do stuff")
        assert "<role>" in result
        assert "<available_tools>" not in result
