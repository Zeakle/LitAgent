"""Render tiered paper metadata at progressively lower detail."""

from __future__ import annotations

from dataclasses import dataclass, field

from litagent.logging import get_logger

logger = get_logger("context.compressor")


@dataclass
class PaperInfo:
    """Store paper metadata used by tier-based compression."""

    paper_id: str
    title: str
    tier: int

    claims: list[str] = field(default_factory=list)
    metrics: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    extraction: dict[str, str] = field(default_factory=dict)


class TierCompressor:
    """Render paper context with detail determined by citation tier."""

    def compress(self, papers: list[PaperInfo]) -> str:
        """Group papers by tier and render their combined context."""
        tiers: dict[int, list[PaperInfo]] = {1: [], 2: [], 3: []}
        for p in papers:
            tier = p.tier if p.tier in tiers else 3
            tiers[tier].append(p)

        parts: list[str] = []
        if tiers[1]:
            parts.append(self._compress_tier1(tiers[1]))
        if tiers[2]:
            parts.append(self._compress_tier2(tiers[2]))
        if tiers[3]:
            parts.append(self._compress_tier3(tiers[3]))

        logger.debug(
            f"Compressed {len(papers)} papers: "
            f"T1={len(tiers[1])}, T2={len(tiers[2])}, T3={len(tiers[3])}"
        )

        return "\n\n".join(parts)

    def _compress_tier1(self, papers: list[PaperInfo]) -> str:
        """Render tier-one papers with full extraction details."""
        lines = ["## Seminal Papers (Full Extraction)"]

        for p in papers:

            lines.append(f"\n### {p.title}")
            if p.extraction:
                for k, v in p.extraction.items():
                    lines.append(f"- **{k}**: {v}")

            if p.claims:
                lines.append("- **Claims**: " + "; ".join(p.claims))

            if p.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in p.metrics.items())
                lines.append(f"- **Metrics**: {metrics_str}")

            if p.summary:
                lines.append(f"- **Summary**: {p.summary}")

        return "\n".join(lines)

    def _compress_tier2(self, papers: list[PaperInfo]) -> str:
        """Render tier-two papers with claims and metrics."""
        lines = ["## High-Citation Papers (Claims & Metrics)"]
        for p in papers:
            entry = [f"- **{p.title}**"]
            if p.claims:
                entry.append(f"  Claims: {'; '.join(p.claims[:3])}")
            if p.metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in p.metrics.items())
                entry.append(f"  Metrics: {metrics_str}")
            lines.append("\n".join(entry))
        return "\n".join(lines)

    def _compress_tier3(self, papers: list[PaperInfo]) -> str:
        """Render tier-three papers with summaries and tags."""
        lines = ["## General Papers"]
        for p in papers:
            tag_str = f" [{', '.join(p.tags)}]" if p.tags else ""
            lines.append(f"- {p.title}: {p.summary}{tag_str}")
        return "\n".join(lines)
