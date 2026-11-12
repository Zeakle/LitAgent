"""Prepare isolated corpus collections shared by benchmark families."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import yaml

from litagent.config import AppConfig
from litagent.rag.chunking import build_corpus_chunker
from litagent.rag.ingest import (
    CorpusIngestor,
    ParsedAuditRepository,
    QuarantineRepository,
)
from litagent.rag.manifest import load_manifest
from litagent.rag.pdf_parser import PyMuPDFParser
from litagent.rag.quality import CorpusTextQualityGate
from litagent.rag.runtime import CorpusRuntime
from litagent.rag.sources import ArxivPDFAdapter, LocalPDFAdapter


def merge_manifest_bundle(
    manifest_paths: list[Path],
    *,
    raw_root: Path,
    corpus_version: str,
    output_path: Path,
    include_paper_ids: set[str] | None = None,
) -> Path:
    """Merge validated manifests into one deterministic ingestion snapshot."""
    if not manifest_paths:
        raise ValueError("manifest bundle must not be empty")
    papers: dict[str, dict] = {}
    for path in manifest_paths:
        manifest = load_manifest(path, raw_root=raw_root)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_by_id = {paper["paper_id"]: paper for paper in raw.get("papers", [])}
        for paper in manifest.papers:
            candidate = raw_by_id[paper.paper_id]
            previous = papers.get(paper.paper_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting manifest paper: {paper.paper_id}")
            if include_paper_ids is None or paper.paper_id in include_paper_ids:
                papers[paper.paper_id] = candidate
    if include_paper_ids is not None:
        missing = include_paper_ids - set(papers)
        if missing:
            raise ValueError(f"manifest bundle is missing papers: {sorted(missing)}")
    payload = {
        "schema_version": 1,
        "corpus_version": corpus_version,
        "papers": [papers[paper_id] for paper_id in sorted(papers)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    load_manifest(output_path, raw_root=raw_root)
    return output_path


async def ingest_benchmark_manifest(
    *,
    config: AppConfig,
    runtime: CorpusRuntime,
    manifest_path: Path,
) -> None:
    """Ingest one prepared manifest and reject incomplete benchmark indexes."""
    async with httpx.AsyncClient() as client:
        parser = PyMuPDFParser(
            quality_gate=CorpusTextQualityGate(**config.rag.quality.model_dump()),
            chunker=build_corpus_chunker(config.rag),
        )
        ingestor = CorpusIngestor(
            config=config.rag,
            service=runtime.service,
            parser=parser,
            local_pdf_adapter=LocalPDFAdapter(max_pdf_bytes=config.rag.max_pdf_bytes),
            pdf_adapter=ArxivPDFAdapter(
                client,
                raw_root=Path(config.rag.raw_root),
                max_pdf_bytes=config.rag.max_pdf_bytes,
                download_timeout_seconds=config.rag.download_timeout_seconds,
                allowed_content_types=config.rag.allowed_pdf_content_types,
            ),
            quarantine=QuarantineRepository(Path(config.rag.quarantine_root)),
            audit_repository=ParsedAuditRepository(Path(config.rag.parsed_root)),
        )
        summary = await ingestor.ingest_manifest(manifest_path, resume=True)
        if summary.status != "succeeded":
            raise RuntimeError("benchmark_ingestion_failed")
