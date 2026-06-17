"""TierCompressor——按论文重要性分层压缩 context。

Tier 1 (基石论文, ~5 篇):  完整 extraction (~500 tokens/篇) → ~2500 tokens
Tier 2 (高引论文, ~15 篇): claims + metrics (~150 tokens/篇) → ~2250 tokens
Tier 3 (一般论文, ~30 篇): 一句话 summary + 标签 (~30 tokens/篇)  → ~900 tokens
Total: ~5650 tokens → 在 Synthesis Worker 的预算内
"""


from __future__ import annotations
from dataclasses import dataclass, field

from litagent.logging import get_logger


logger = get_logger('context.compressor')


@dataclass
class PaperInfo:
    """论文的结构化信息——Extractor Agent 的输出格式（Phase 8 定义）。

    此处只定义 Compressor 需要的字段。Phase 8 实际 extraction 可能有更多字段，
    但 Compressor 只读这些。
    """
    paper_id: str
    title: str
    tier: int
    # ── 通用字段（所有领域的论文都有）──
    claims: list[str] = field(default_factory=list)
    metrics: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    extraction: dict[str, str] = field(default_factory=dict)


class TierCompressor:
    """按引用网络重要性分三级压缩论文信息。

    Graph Agent 标记每篇论文的 tier（基于 citation count + PageRank），
    Compressor 按 tier 决定保留多少信息。

    50 篇 extraction 直接塞进 context 会超 25000 tokens。
    分层压缩后 ~5650 tokens，在 Synthesis Worker 的预算内。
    """

    def compress(self, papers: list[PaperInfo]) -> str:
        """讲论文列表按 tier 分组并压缩为文本"""
        tiers: dict[int, list[PaperInfo]] = {1: [], 2: [], 3: []}
        for p in papers:
            tier = p.tier if p.tier in tiers else 3
            tiers[tier].append(p)

        # 依次compact 各tiers
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

        return '\n\n'.join(parts)

    
    def _compress_tier1(self, papers: list[PaperInfo]) -> str:
        """Tier 1: 基石论文--完整extraction. ~500tokens/篇"""
        lines = ["## Seminal Papers (Full Extraction)"]
        # Tier1 拼入extraction\claims\metrics\summary4
        for p in papers:
            # lines逐行写入内容，拼成完整文本
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

        return '\n'.join(lines)

    
    def _compress_tier2(self, papers: list[PaperInfo]) -> str:
        """Tier 2: 高引论文——只保留 claims + metrics。~150 tokens/篇。"""
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
        """Tier 3: 一般论文——一句话 summary + 标签。~30 tokens/篇。"""
        lines = ["## General Papers"]
        for p in papers:
            tag_str = f" [{', '.join(p.tags)}]" if p.tags else ""
            lines.append(f"- {p.title}: {p.summary}{tag_str}")
        return "\n".join(lines)