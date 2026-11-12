"""Verify bounded fulltext enrichment, failure disclosure and parser ownership."""

import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from litagent.config import RAGConfig
from litagent.rag.models import ContentChunk, ContentScope, PaperRecord
from litagent.rag.targeted_fulltext import TargetedFulltext


def configuration(tmp_path, **kwargs):
    """Keep tests outside the user's corpus and report history."""
    return RAGConfig(
        raw_root=str(tmp_path / "raw"),
        manifest_path=str(tmp_path / "missing.yaml"),
        **kwargs,
    )


class ParsedFixture:
    """Provide actual chunk schemas with controlled document quality."""

    def parse_with_quality(self, asset):
        chunk = ContentChunk.from_text(
            paper_id=asset.paper_id,
            chunk_key="method:1",
            text="ViT uses image patches and attention.",
            section="method",
            content_scope=ContentScope.SELECTED_FULLTEXT,
            page=1,
            block_index=0,
            bbox=(0, 0, 100, 20),
        )
        record = PaperRecord(
            paper_id=asset.paper_id,
            title=asset.title,
            asset_hash=asset.asset_hash,
            chunks=[chunk],
            content_scope=ContentScope.SELECTED_FULLTEXT,
        )
        return SimpleNamespace(record=record, quality=SimpleNamespace(reason_codes=[]))


class Audit:
    """Record only local parser outcomes in the fixture."""

    async def record(self, *args):
        return {}


@pytest.mark.asyncio
async def test_bounded_allowlist_fetch_uses_eight_original_sources_and_no_other_urls(
    tmp_path,
):
    calls = []
    active = 0
    high_water = 0

    async def handle(request):
        nonlocal active, high_water
        calls.append(str(request.url))
        active += 1
        high_water = max(active, high_water)
        await asyncio.sleep(0.001)
        active -= 1
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-fixture"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        loader = TargetedFulltext(
            configuration(tmp_path), client=client, parser=ParsedFixture()
        )
        loader.audit = Audit()
        papers = [
            {
                "paper_id": f"arxiv:2010.{10000+i}",
                "title": "ViT",
                "abstract": "ViT",
                "chunks": [],
            }
            for i in range(10)
        ]
        result = await loader.enrich(papers, [{"kind": "literal", "text": "ViT"}])
    assert len(calls) == 8 and high_water <= 2
    assert all(url.startswith("https://arxiv.org/pdf/") for url in calls)
    assert sum(p["fulltext_acquisition"]["status"] == "available" for p in result) == 8
    assert all(
        p["fulltext_acquisition"]["full_document_reviewed"] is False for p in result
    )
    assert papers[0]["chunks"] == []


@pytest.mark.asyncio
async def test_http_failure_and_non_allowlisted_source_never_claim_fulltext_review(
    tmp_path,
):
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        loader = TargetedFulltext(configuration(tmp_path), client=client)
        papers = [
            {
                "paper_id": "doi:10.1234/missing",
                "title": "paper",
                "abstract": "original",
                "pdf_url": "https://evil.test/secret",
            },
            {"paper_id": "arxiv:2010.11929", "title": "paper", "abstract": "original"},
        ]
        result = await loader.enrich(papers, [])
    assert len(calls) == 1
    assert all(p["fulltext_acquisition"]["status"] == "failed" for p in result)
    assert all(p["abstract"] == "original" for p in result)
    assert result[0]["fulltext_acquisition"]["reason_code"] == "fulltext_unavailable"


@pytest.mark.asyncio
async def test_external_redirect_is_rejected_before_contacting_its_target(tmp_path):
    calls = []

    async def handle(request):
        calls.append(str(request.url))
        return httpx.Response(
            302, headers={"location": "https://evil.test/private.pdf"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        loader = TargetedFulltext(configuration(tmp_path), client=client)
        result = await loader.enrich(
            [{"paper_id": "arxiv:2010.11929", "title": "ViT"}], []
        )
    assert calls == ["https://arxiv.org/pdf/2010.11929"]
    assert result[0]["fulltext_acquisition"]["status"] == "failed"


@pytest.mark.asyncio
async def test_parser_timeout_keeps_real_thread_slot_until_completion(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingParser:
        def parse_with_quality(self, asset):
            started.set()
            release.wait(5)
            return None

    loader = TargetedFulltext(
        configuration(tmp_path, parser_timeout_seconds=0.01), parser=BlockingParser()
    )
    try:
        with pytest.raises(TimeoutError):
            await loader._parse(None)
        assert started.is_set() and loader.slots._value == 1
    finally:
        release.set()
    for _ in range(100):
        if loader.slots._value == 2:
            break
        await asyncio.sleep(0.01)
    assert loader.slots._value == 2


@pytest.mark.asyncio
async def test_parser_cancellation_does_not_release_a_running_thread(tmp_path):
    release = threading.Event()

    class BlockingParser:
        def parse_with_quality(self, asset):
            release.wait(5)

    loader = TargetedFulltext(configuration(tmp_path), parser=BlockingParser())
    task = asyncio.create_task(loader._parse(None))
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert loader.slots._value == 1
    finally:
        release.set()
    for _ in range(100):
        if loader.slots._value == 2:
            break
        await asyncio.sleep(0.01)
    assert loader.slots._value == 2


@pytest.mark.asyncio
async def test_allowlisted_redirect_cannot_substitute_another_paper(tmp_path):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(
            302, headers={"location": "https://export.arxiv.org/pdf/2001.00001"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        loader = TargetedFulltext(configuration(tmp_path), client=client)
        result = await loader.enrich(
            [{"paper_id": "arxiv:2010.11929", "title": "ViT"}], []
        )
    assert len(calls) == 1
    assert (
        result[0]["fulltext_acquisition"]["reason_code"] == "source_identity_mismatch"
    )
