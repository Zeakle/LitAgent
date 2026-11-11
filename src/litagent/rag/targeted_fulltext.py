"""Enrich bounded Survey evidence from trusted local or allowlisted arXiv PDFs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import httpx

from litagent.body_quality import check_preparation_policy, checked_preparation_chunks
from litagent.operational_limits import PROFILE, increase
from litagent.rag.chunking import SectionAwareChunker, build_corpus_chunker
from litagent.rag.ingest import ParsedAuditRepository
from litagent.rag.manifest import load_manifest, materialize_manifest_assets
from litagent.rag.models import (
    RawPaperAsset,
    SourceKind,
    SourceRef,
    normalize_paper_identity,
)
from litagent.rag.pdf_parser import (
    NUMBERED_HEADING_POLICY,
    CorpusParseError,
    PyMuPDFParser,
)
from litagent.rag.quality import CorpusTextQualityGate
from litagent.rag.sources import ArxivPDFAdapter, LocalPDFAdapter


class TargetedFulltext:
    """Fetch a bounded set of original documents before freezing writing evidence."""

    def __init__(
        self,
        config,
        *,
        client=None,
        parser=None,
        preparation_policy=None,
        prefer_method_sections=False,
    ):
        """Reuse the corpus boundaries with strict limits independent of looser config."""
        self.config = config
        uplifted = config.resource_revision == PROFILE
        ceiling = increase if uplifted else lambda value: value
        self.max_papers = ceiling(8)
        self.concurrency = ceiling(2)
        self.max_pdf_bytes = min(config.max_pdf_bytes, ceiling(50 * 1024 * 1024))
        self.download_timeout = min(config.download_timeout_seconds, ceiling(60))
        self.parser_timeout = min(config.parser_timeout_seconds, ceiling(30))
        self.client = client
        self.preparation_policy = check_preparation_policy(preparation_policy)
        self.prefer_method_sections = prefer_method_sections
        if self.preparation_policy and config.chunk_strategy.value != "section_aware":
            raise ValueError("evidence_preparation_chunk_strategy_invalid")
        self.parser = parser or (
            PyMuPDFParser(
                chunker=build_corpus_chunker(config),
                quality_gate=CorpusTextQualityGate(**config.quality.model_dump()),
                heading_policy=NUMBERED_HEADING_POLICY,
            )
            if self.preparation_policy
            else PyMuPDFParser(
                **(
                    {
                        "heading_policy": NUMBERED_HEADING_POLICY,
                        "chunker": SectionAwareChunker(
                            chunk_size=max(config.chunk_size, 2200),
                            chunk_overlap=config.chunk_overlap,
                        ),
                    }
                    if prefer_method_sections
                    else {}
                )
            )
        )
        self.slots = asyncio.Semaphore(
            min(self.concurrency, config.max_concurrent_parsers)
            if self.preparation_policy
            else self.concurrency
        )
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
            return await asyncio.wait_for(asyncio.shield(task), self.parser_timeout)
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
        if self.preparation_policy:
            for paper in papers:
                checked_preparation_chunks(paper)
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
            if self.preparation_policy
            or not any(
                c.get("content_scope") == "selected_fulltext"
                for c in papers[i].get("chunks", [])
            )
        ]
        selected = set(candidates[: self.max_papers])
        output = (
            copy.deepcopy(papers)
            if self.preparation_policy
            else [dict(p) for p in papers]
        )
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
            if self.preparation_policy:
                p["fulltext_acquisition"].update(
                    preparation_policy=self.preparation_policy,
                    previous_chunk_keys=[c["chunk_key"] for c in p.get("chunks", [])],
                    existing_partial_fulltext=any(
                        c.get("content_scope") == "selected_fulltext"
                        for c in p.get("chunks", [])
                    ),
                )
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(follow_redirects=False)
        downloads = asyncio.Semaphore(self.concurrency)
        adapter = ArxivPDFAdapter(
            client,
            raw_root=Path(self.config.raw_root),
            max_pdf_bytes=self.max_pdf_bytes,
            download_timeout_seconds=self.download_timeout,
            allowed_content_types=self.config.allowed_pdf_content_types,
        )
        jobs = asyncio.Semaphore(self.concurrency)

        async def one(i):
            paper = output[i]
            pid = normalize_paper_identity(str(paper.get("paper_id") or ""))
            record = paper["fulltext_acquisition"]
            record["status"] = "started"
            try:
                asset = local.get(pid)
                if asset is not None and asset.pdf_path is not None:
                    asset = await LocalPDFAdapter(
                        max_pdf_bytes=self.max_pdf_bytes
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
                            self.download_timeout,
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
                if self.preparation_policy:
                    # The manifest authorizes asset access; it cannot replace the
                    # current source's abstract with a curated manifest summary.
                    asset = asset.model_copy(
                        update={
                            "title": paper.get("title") or pid,
                            "abstract": paper.get("abstract") or "",
                        }
                    )
                    record["asset_sha256"] = asset.asset_hash
                parsed = await self._parse(asset)
                if (
                    self.preparation_policy
                    and parsed.record is not None
                    and (normalize_paper_identity(parsed.record.paper_id) != pid)
                ):
                    raise CorpusParseError(
                        "source_identity_mismatch",
                        "parsed paper has different ownership",
                    )
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
                if self.preparation_policy:
                    # A new parsed document is one source version. Do not mix
                    # unbound fragments from a prior PDF into this version.
                    paper["chunks"] = [
                        c.model_dump(mode="json") for c in parsed.record.chunks
                    ]
                    checked_preparation_chunks(paper)
                    record.update(
                        parser_policy=NUMBERED_HEADING_POLICY,
                        chunk_strategy=self.config.chunk_strategy.value,
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                else:
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
                preparation_error = (
                    str(exc)
                    if self.preparation_policy
                    and str(exc)
                    in {
                        "evidence_preparation_source_identity_invalid",
                        "evidence_preparation_source_invalid",
                    }
                    else None
                )
                record.update(
                    status="failed",
                    error_type=type(exc).__name__,
                    reason_code=getattr(
                        exc,
                        "code",
                        preparation_error
                        or (
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
