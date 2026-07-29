"""Build and validate stable, citable evidence-ledger entries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
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
    """Build claim and abstract evidence items for one extraction."""
    key = _stable_paper_key(extraction)
    title = extraction.get("title", "")
    pid = extraction.get("paper_id", "")
    items: list[EvidenceItem] = []

    for idx, claim in enumerate(extraction.get("claims", []) or []):
        if not isinstance(claim, str) or not claim.strip():
            continue

        items.append(
            EvidenceItem(
                evidence_id=f"{key}:claim:{idx}",
                paper_id=pid,
                paper_title=title,
                text=claim.strip(),
                source_locator="extracted_claim",
                confidence=None,
            )
        )

    abstract = (extraction.get("abstract") or "").strip()
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

    return [it.to_dict() for it in items]


def collect_ledger(extractions: list[dict]) -> dict[str, dict[str, Any]]:
    """Index evidence items by ID while preserving the first occurrence."""
    ledger: dict[str, dict[str, Any]] = {}
    for ext in extractions or []:
        for item in ext.get("evidence_items", []) or []:
            eid = item.get("evidence_id", "")
            if eid and eid not in ledger:
                ledger[eid] = item

    return ledger


def format_ledger(
    ledger: dict[str, dict[str, Any]], max_chars: int | None = None
) -> str:
    """Format evidence for prompts with optional line-boundary truncation."""
    lines = [
        f"[E:{eid}] ({item.get('paper_title', '')}) {item.get('text', '')}"
        for eid, item in ledger.items()
    ]

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
