"""Build and validate stable, citable evidence-ledger entries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

_EVIDENCE_REF_RE = re.compile(r"\[E:([^\[\]\s]+)\]")


@dataclass
class EvidenceItem:
    """Represent one citable evidence-ledger entry."""

    evidence_id: str
    paper_id: str
    paper_title: str
    text: str
    source_locator: str
    confidence: float | None
    claim_text: str | None = None
    supporting_text: str | None = None
    content_scope: str = "unknown"
    chunk_key: str | None = None
    section: str | None = None
    page: int | None = None
    block_index: int | None = None
    bbox: list[float] | None = None
    content_hash: str | None = None
    raw_content_hash: str | None = None
    evidence_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        """Return all evidence-ledger fields as a plain dictionary."""
        return asdict(self)


def _stable_paper_key(extraction: dict) -> str:
    """Return a stable paper identifier or title-derived fallback."""
    pid = extraction.get("paper_id", "")
    if pid:
        return pid

    title = extraction.get("title", "")
    return "t" + hashlib.md5(title.encode("utf-8")).hexdigest()[:8]


def build_evidence_items(extraction: dict) -> list[dict[str, Any]]:
    """Build locator-backed v2 evidence or fall back to v1 claim projections."""
    key = _stable_paper_key(extraction)
    title = str(extraction.get("title") or "")
    pid = str(extraction.get("paper_id") or "")
    raw_chunks = extraction.get("chunks")
    chunks = {
        str(chunk.get("chunk_key")): chunk
        for chunk in (raw_chunks or [])
        if isinstance(chunk, dict) and chunk.get("chunk_key")
    }
    items: list[EvidenceItem] = []
    claim_records = extraction.get("claim_records")

    if isinstance(claim_records, list):
        for record in claim_records:
            if not isinstance(record, dict):
                continue
            claim_text = str(record.get("text") or "").strip()
            chunk_key = str(record.get("source_chunk_key") or "")
            chunk = chunks.get(chunk_key)
            if not claim_text or chunk is None:
                continue
            digest = hashlib.sha256(
                f"{claim_text}|{chunk_key}".encode("utf-8")
            ).hexdigest()[:12]
            page = "" if chunk.get("page") is None else chunk["page"]
            block_index = (
                "" if chunk.get("block_index") is None else chunk["block_index"]
            )
            locator = (
                f"paper={pid};chunk={chunk_key};"
                f"section={chunk.get('section', 'unknown')};"
                f"page={page};block={block_index}"
            )
            items.append(
                EvidenceItem(
                    evidence_id=f"{key}:{chunk_key}:claim:{digest}",
                    paper_id=pid,
                    paper_title=title,
                    text=claim_text,
                    source_locator=locator,
                    confidence=record.get("confidence"),
                    claim_text=claim_text,
                    supporting_text=str(chunk.get("text") or ""),
                    content_scope=str(chunk.get("content_scope") or "unknown"),
                    chunk_key=chunk_key,
                    section=chunk.get("section"),
                    page=chunk.get("page"),
                    block_index=chunk.get("block_index"),
                    bbox=chunk.get("bbox"),
                    content_hash=chunk.get("content_hash"),
                    raw_content_hash=chunk.get("raw_content_hash"),
                    evidence_version="v2",
                )
            )
        return [item.to_dict() for item in items]

    # Compatibility evidence remains usable in the current run, but is v1 and
    # therefore never eligible for trusted cross-run promotion.
    for index, claim in enumerate(extraction.get("claims") or []):
        if isinstance(claim, str) and claim.strip():
            items.append(
                EvidenceItem(
                    evidence_id=f"{key}:claim:{index}",
                    paper_id=pid,
                    paper_title=title,
                    text=claim.strip(),
                    source_locator="extracted_claim",
                    confidence=None,
                )
            )
    abstract = str(extraction.get("abstract") or "").strip()
    if abstract:
        items.append(
            EvidenceItem(
                evidence_id=f"{key}:abstract",
                paper_id=pid,
                paper_title=title,
                text=abstract[:500],
                source_locator="abstract",
                confidence=None,
            )
        )
    return [item.to_dict() for item in items]


def collect_ledger(extractions: list[dict]) -> dict[str, dict[str, Any]]:
    """Index evidence items by ID while preserving the first occurrence."""
    ledger: dict[str, dict[str, Any]] = {}
    for ext in extractions or []:
        for item in ext.get("evidence_items", []) or []:
            eid = item.get("evidence_id", "")
            if eid and eid not in ledger:
                ledger[eid] = item

    return ledger


def _format_ledger_line(eid: str, item: dict[str, Any]) -> str:
    """Render one full EvidenceItem line, appending support when present."""
    line = f"[E:{eid}] ({item.get('paper_title', '')}) {item.get('text', '')}"
    support = item.get("supporting_text")
    if support:
        line += f"\n    support: {support}"
    return line


def format_ledger(
    ledger: dict[str, dict[str, Any]], max_chars: int | None = None
) -> str:
    """Format evidence for prompts with optional line-boundary truncation."""
    lines = [_format_ledger_line(eid, item) for eid, item in ledger.items()]

    if max_chars is None:
        return "\n".join(lines)

    out: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        if used + len(line) + 1 > max_chars:
            out.append(f"... ({len(lines) - i} more evidence items omitted)")
            break
        out.append(line)
        used += len(line) + 1

    return "\n".join(out)


def extract_evidence_refs_ordered(text: str) -> list[str]:
    """Return unique evidence IDs in first-appearance order."""
    seen: set[str] = set()
    ordered: list[str] = []

    for match in _EVIDENCE_REF_RE.finditer(text or ""):
        evidence_id = match.group(1)
        if evidence_id not in seen:
            seen.add(evidence_id)
            ordered.append(evidence_id)

    return ordered


def extract_evidence_refs(text: str) -> set[str]:
    """Return referenced evidence IDs as a compatibility set."""
    return set(extract_evidence_refs_ordered(text))


def find_unknown_refs(text: str, ledger: dict[str, Any]) -> list[str]:
    """Return sorted references that are absent from the ledger."""
    return sorted(ref for ref in extract_evidence_refs(text) if ref not in ledger)
