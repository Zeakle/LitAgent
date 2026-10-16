"""Enrich bounded Survey evidence from trusted local or allowlisted arXiv PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from litagent.rag.ingest import ParsedAuditRepository
from litagent.rag.manifest import load_manifest, materialize_manifest_assets
from litagent.rag.models import (
    RawPaperAsset,
    SourceKind,
    SourceRef,
    normalize_paper_identity,
)
from litagent.rag.pdf_parser import CorpusParseError, PyMuPDFParser
from litagent.rag.sources import ArxivPDFAdapter, LocalPDFAdapter


class TargetedFulltext:
    """Fetch at most eight original documents before freezing writing evidence."""

    def __init__(self, config, *, client=None, parser=None):
        """Reuse the corpus boundaries with strict limits independent of looser config."""
        self.config = config
        self.client = client
        self.parser = parser or PyMuPDFParser()
        self.slots = asyncio.Semaphore(2)
        self.audit = ParsedAuditRepository(Path(config.raw_root).parent / "parsed")

    def _release(self, task):
        """Retain timed-out thread ownership until the actual parser exits."""
        self.slots.release()
        if not task.cancelled():
            task.exception()

    async def _parse(self, asset):
        """Keep parser concurrency bounded even after cancellation or timeout."""
        await self.slots.acquire()
        task = asyncio.create_task(
            asyncio.to_thread(self.parser.parse_with_quality, asset)
        )
        deferred = False
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), min(30, self.config.parser_timeout_seconds)
            )
        except (TimeoutError, asyncio.CancelledError):
            deferred = True
            task.add_done_callback(self._release)
            raise
        finally:
            if not deferred:
                self.slots.release()

    def _local_assets(self):
        """Read only trusted manifest paths, never a path supplied by a search result."""
        path = Path(self.config.manifest_path)
        if not path.is_file():
            return {}
        manifest = load_manifest(path, raw_root=Path(self.config.raw_root))
        return {
            normalize_paper_identity(a.paper_id): a
            for a in materialize_manifest_assets(manifest)
        }

    async def enrich(self, papers, requirements):
        """Keep failed/missing fulltext visible while preserving the original abstracts."""
        local_error = None
        try:
            local = self._local_assets()
        except Exception as exc:
            local = {}
            local_error = getattr(exc, "code", "manifest_invalid")
        named = [
            r["text"].casefold() for r in requirements if r.get("kind") == "literal"
        ]
        order = sorted(
            range(len(papers)),
            key=lambda i: (
                not any(
                    n in json.dumps(papers[i], ensure_ascii=False).casefold()
                    for n in named
                ),
                i,
            ),
        )
        candidates = [
            i
            for i in order
            if not any(
                c.get("content_scope") == "selected_fulltext"
                for c in papers[i].get("chunks", [])
            )
        ]
        selected = set(candidates[:8])
        output = [dict(p) for p in papers]
        for i, p in enumerate(output):
            p["fulltext_acquisition"] = {
                "status": (
                    "already_available"
                    if i not in candidates
                    else "capacity_not_selected"
                ),
                "scope": "selected_fulltext",
                "full_document_reviewed": False,
                "manifest_error": local_error,
            }
            p["quality_policy"] = "strict_v1"
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(follow_redirects=False)
        downloads = asyncio.Semaphore(2)
        adapter = ArxivPDFAdapter(
            client,
            raw_root=Path(self.config.raw_root),
            max_pdf_bytes=min(self.config.max_pdf_bytes, 50 * 1024 * 1024),
            download_timeout_seconds=min(self.config.download_timeout_seconds, 60),
            allowed_content_types=self.config.allowed_pdf_content_types,
        )
        jobs = asyncio.Semaphore(2)

        async def one(i):
            paper = output[i]
            pid = normalize_paper_identity(str(paper.get("paper_id") or ""))
            record = paper["fulltext_acquisition"]
            record["status"] = "started"
            try:
                asset = local.get(pid)
                if asset is not None and asset.pdf_path is not None:
                    asset = await LocalPDFAdapter(
                        max_pdf_bytes=min(self.config.max_pdf_bytes, 50 * 1024 * 1024)
                    ).materialize(asset)
                elif (asset is not None and asset.pdf_url) or pid.startswith("arxiv:"):
                    aid = pid.removeprefix("arxiv:")
                    # Adapter validates the identifier-derived URL and every redirect.
                    asset = asset or RawPaperAsset(
                        paper_id=pid,
                        title=paper.get("title") or pid,
                        abstract=paper.get("abstract") or "",
                        asset_hash=hashlib.sha256(pid.encode()).hexdigest(),
                        pdf_url="https://arxiv.org/pdf/" + aid,
                    )
                    async with downloads:
                        asset = await asyncio.wait_for(
                            adapter.materialize(asset),
                            min(self.config.download_timeout_seconds, 60),
                        )
                    asset = asset.model_copy(
                        update={
                            "sources": [
                                SourceRef(
                                    kind=SourceKind.ARXIV_PDF,
                                    source_id=aid,
                                    uri=asset.pdf_url,
                                    sha256=asset.asset_hash,
                                )
                            ]
                        }
                    )
                else:
                    raise CorpusParseError(
                        "fulltext_unavailable", "no trusted public PDF source"
                    )
                parsed = await self._parse(asset)
                await self.audit.record(asset, parsed)
                if parsed.record is None:
                    raise CorpusParseError(
                        "fulltext_quarantined", "no usable source text"
                    )
                chunks = [
                    c.model_dump(mode="json")
                    for c in parsed.record.chunks
                    if c.content_scope.value == "selected_fulltext"
                ]
                if not chunks:
                    raise CorpusParseError(
                        "fulltext_parse_insufficient", "no selected fulltext blocks"
                    )
                paper["chunks"] = [*paper.get("chunks", []), *chunks]
                paper["content_scope"] = "selected_fulltext"
                paper["provenance"] = [
                    *paper.get("provenance", []),
                    *[s.model_dump(mode="json") for s in parsed.record.sources],
                ]
                record.update(
                    status="available",
                    asset_sha256=asset.asset_hash,
                    selected_chunks=len(chunks),
                    reason_codes=parsed.quality.reason_codes,
                )
            except asyncio.CancelledError:
                record.update(status="cancelled", reason_code="fulltext_cancelled")
                raise
            except Exception as exc:
                record.update(
                    status="failed",
                    error_type=type(exc).__name__,
                    reason_code=getattr(
                        exc,
                        "code",
                        (
                            "fulltext_timeout"
                            if isinstance(exc, TimeoutError)
                            else "fulltext_fetch_failed"
                        ),
                    ),
                )

        try:

            async def bounded_one(i):
                async with jobs:
                    await one(i)

            await asyncio.gather(*(bounded_one(i) for i in sorted(selected)))
        finally:
            if owns_client:
                await client.aclose()
        return output
