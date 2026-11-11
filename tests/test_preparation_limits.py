"""Keep focused preparation within its source and settled-request boundaries."""

import threading
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from test_extraction_paper_batches import strategy
from test_question_focused_preparation import POLICY, attach, cached_pdf, paper

from litagent.config import RAGConfig
from litagent.llm.client import OpenAICompatibleClient
from litagent.rag.chunking import SectionAwareChunker
from litagent.rag.pdf_parser import NUMBERED_HEADING_POLICY, PyMuPDFParser
from litagent.rag.targeted_fulltext import TargetedFulltext
from litagent.safety.budget import CostBudget


@pytest.mark.asyncio
async def test_actual_wire_must_stay_within_the_preflight_bound(monkeypatch):
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    create = AsyncMock()
    monkeypatch.setattr(client._client.chat.completions, "create", create)
    source = attach(paper())
    source["_preparation_request_bound"] = 1
    try:
        with pytest.raises(ValueError, match="budget_plan_violation"):
            await strategy(client).extract_batch([source])
        create.assert_not_called()
        assert budget.snapshot()["requests_sent"] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("uplift", [False, True])
async def test_new_policy_keeps_eight_originals_and_lower_parser_concurrency(
    tmp_path, uplift
):
    template = cached_pdf(tmp_path)
    originals, active, peak = [], 0, 0
    lock = threading.Lock()

    class CountedParser(PyMuPDFParser):
        """Observe actual parser ownership without replacing its implementation."""

        def parse_with_quality(self, asset):
            """Release observed ownership even if the real parser fails."""
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.005)
                return super().parse_with_quality(asset)
            finally:
                with lock:
                    active -= 1

    for index in range(10):
        source = paper()
        source["paper_id"] = f"arxiv:2601.{12400 + index}"
        for item in source["chunks"]:
            item["paper_id"] = source["paper_id"]
        target = template.parent / (source["paper_id"].replace(":", "_") + ".pdf")
        target.write_bytes(template.read_bytes())
        originals.append(source)
    cfg = RAGConfig(
        raw_root=str(template.parent),
        manifest_path=str(tmp_path / "missing.yaml"),
        chunk_strategy="section_aware",
        max_concurrent_parsers=1,
        resource_revision="operational-plus20-v1" if uplift else None,
    )
    parser = CountedParser(
        chunker=SectionAwareChunker(chunk_size=1200, chunk_overlap=150),
        heading_policy=NUMBERED_HEADING_POLICY,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail("unexpected HTTP"))
    ) as client:
        result = await TargetedFulltext(
            cfg, client=client, parser=parser, preparation_policy=POLICY
        ).enrich(originals, [])
    assert sum(p["fulltext_acquisition"]["status"] == "available" for p in result) == (
        10 if uplift else 8
    )
    assert sum(
        p["fulltext_acquisition"]["status"] == "capacity_not_selected" for p in result
    ) == (0 if uplift else 2)
    assert peak == 1 and active == 0
    assert all(
        p["fulltext_acquisition"]["full_document_reviewed"] is False for p in result
    )
