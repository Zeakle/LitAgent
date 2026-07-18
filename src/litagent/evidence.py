"""Evidence Ledger——从 extraction 构建稳定 evidence id 的可追溯证据条目。

producer: ExtractorWorker（每篇论文产 evidence_items）
consumer: SynthesisWorker（<evidence_ledger> + [E:id] 引用）、
          faithfulness 诊断（unsupported_claims）、runner 的 bounded rewrite 校验。
"""


from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, asdict
from typing import Any


# survey 正文中的证据引用标记：[E:2401.00001:claim:0]
_EVIDENCE_REF_RE = re.compile(r'\[E:([^\[\]\s]+)\]')


@dataclass
class EvidenceItem:
    evidence_id: str
    paper_id: str
    paper_title: str
    text: str
    source_locator: str
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_paper_key(extraction: dict) -> str:
    """evidence_id 的 paper 前缀。无 paper_id 时用 title 哈希——同 title 跨 run 稳定。"""
    pid = extraction.get('paper_id', '')
    if pid:
        return pid

    title = extraction.get('title', '')
    return 't' + hashlib.md5(title.encode('utf-8')).hexdigest()[:8]


def build_evidence_items(extraction: dict) -> list[dict[str, Any]]:
    """一篇论文的 extraction dict → evidence_items（dict 形态，可直接进 result/JSON）。

    claims 逐条建 '<key>:claim:<idx>'；abstract 非空另建一条 '<key>:abstract'
    （零 claims 的论文也有可引用证据）。confidence：提取策略当前不产出
    per-claim 置信度 → 显式 None
    """
    key = _stable_paper_key(extraction)
    title = extraction.get('title', '')
    pid = extraction.get('paper_id', '')
    items: list[EvidenceItem] = []

    for idx, claim in enumerate(extraction.get('claims', []) or []):
        if not isinstance(claim, str) or not claim.strip():
            continue

        # 逐 claim append。id 不能含空格——[E:*] 正则排除空白，含空格的 id 无法被引用/校验
        items.append(EvidenceItem(
            evidence_id=f'{key}:claim:{idx}',
            paper_id=pid,
            paper_title=title,
            text=claim.strip(),
            source_locator='extracted_claim',
            confidence=None,
        ))

    abstract = (extraction.get('abstract') or '').strip()
    if abstract:
        items.append(EvidenceItem(
            evidence_id=f'{key}:abstract',
            paper_id=pid,
            paper_title=title,
            text=abstract[:500],
            source_locator='abstract',
            confidence=None
        ))

    return [it.to_dict() for it in items]


def collect_ledger(extractions: list[dict]) -> dict[str, dict[str, Any]]:
    """所有 extraction 的 evidence_items → {evidence_id: item}。id 冲突保留首个。"""
    ledger: dict[str, dict[str, Any]] = {}
    for ext in extractions or []:
        for item in ext.get('evidence_items', []) or []:
            eid = item.get('evidence_id', '')
            if eid and eid not in ledger:
                ledger[eid] = item

    return ledger


def format_ledger(ledger: dict[str, dict[str, Any]], max_chars: int | None = None) -> str:
    """ledger → prompt 文本，一行一条：[E:<id>] (<title>) <text>。

    max_chars: 诊断/rewrite 等无 pipeline 预算控制的 prompt 场景用它截断；
    在行边界截断，尾行标注省略条数。None=不截断（synthesis 层由 pipeline 管预算）。
    """
    lines = [
        f"[E:{eid}] ({item.get('paper_title', '')}) {item.get('text', '')}"
        for eid, item in ledger.items()
    ]

    if max_chars is None:
        return '\n'.join(lines)

    out: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        if used + len(line) + 1 > max_chars:
            out.append(f"... ({len(lines) - i} more evidence items omitted)")
            break
        out.append(line)
        used += len(line) + 1

    return '\n'.join(out)


def extract_evidence_refs(text: str) -> set[str]:
    """survey 正文中出现的所有 [E:<id>] 引用 id"""
    return set(_EVIDENCE_REF_RE.findall(text or ''))


def find_unknown_refs(text: str, ledger: dict[str, Any]) -> list[str]:
    """正文引用但 ledger 中不存在的 evidence id（排序稳定，供报错/测试）。"""
    return sorted(ref for ref in extract_evidence_refs(text) if ref not in ledger)
