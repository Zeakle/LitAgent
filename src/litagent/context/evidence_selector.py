"""Select evidence per survey section with reranking and lexical fallback."""

from __future__ import annotations

import asyncio
import html
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from litagent.context.budget import BudgetManager
from litagent.logging import get_logger
from litagent.observability.context import get_task_id
from litagent.rag.interfaces import Reranker, ScoredDoc

logger = get_logger("context.evidence_selector")

_WORD_RE = re.compile(r"[a-z0-9]+")
_SOURCE_INDEX = "_evidence_selector_source_index"
_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")


@dataclass(frozen=True)
class SectionSpec:
    """Define an immutable survey section and its retrieval hint."""

    key: str
    title: str
    retrieval_hint: str


DEFAULT_SECTIONS = (
    SectionSpec(
        "introduction",
        "Introduction and Motivation",
        "background context, problem statement, and motivation for the survey topic",
    ),
    SectionSpec(
        "taxonomy",
        "Taxonomy of Approaches",
        "categorization of methods, approaches, architectures, and techniques",
    ),
    SectionSpec(
        "methods",
        "Detailed Analysis of Key Methods",
        "specific algorithms, model architectures, training procedures, "
        "technical details",
    ),
    SectionSpec(
        "experiments",
        "Experimental Comparison",
        "benchmarks, datasets, metrics, quantitative results, performance numbers",
    ),
    SectionSpec(
        "open_problems",
        "Open Problems and Future Directions",
        "limitations, gaps, unresolved challenges, and future research directions",
    ),
)


@dataclass
class EvidenceSelection:
    """Store the serializable result of one evidence-selection pass."""

    candidate_count: int
    selected_items: dict[str, dict[str, Any]]
    section_evidence_ids: dict[str, list[str]]
    method_by_section: dict[str, str]
    omitted_count: int
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Return the complete selection result as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceSelection":
        """Validate and reconstruct a selection from a JSON-safe mapping."""
        if not isinstance(value, Mapping):
            raise ValueError(
                f"EvidenceSelection.from_dict expects Mapping, "
                f"got {type(value).__name__}"
            )

        cand_count = value.get("candidate_count")
        if not isinstance(cand_count, int) or cand_count < 0:
            raise ValueError(f"candidate_count must be int >= 0, got {cand_count!r}")

        raw_items = value.get("selected_items", {})
        if not isinstance(raw_items, Mapping):
            raise ValueError(
                f"selected_items must be Mapping, got {type(raw_items).__name__}"
            )

        selected_items: dict[str, dict[str, Any]] = {}
        for k, v in raw_items.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"selected_items key must be non-empty str, got {k!r}")
            if not _EVIDENCE_ID_RE.fullmatch(k):
                raise ValueError(f"selected_items key has invalid evidence id: {k!r}")
            if not isinstance(v, Mapping):
                raise ValueError(
                    f"selected_items[{k!r}] must be Mapping, got {type(v).__name__}"
                )

            item = dict(v)
            if item.get("evidence_id") != k:
                raise ValueError(
                    f"selected_items[{k!r}].evidence_id must equal its key"
                )

            selected_items[k] = item

        raw_sec = value.get("section_evidence_ids", {})

        if not isinstance(raw_sec, Mapping):
            raise ValueError(
                f"section_evidence_ids must be Mapping, got {type(raw_sec).__name__}"
            )

        section_evidence_ids: dict[str, list[str]] = {}
        for k, v in raw_sec.items():
            if not isinstance(k, str):
                raise ValueError(f"section_evidence_ids key must be str, got {k!r}")
            if not isinstance(v, list):
                raise ValueError(
                    f"section_evidence_ids[{k!r}] must be list, got {type(v).__name__}"
                )

            ids: list[str] = []
            for evidence_id in v:
                if (
                    not isinstance(evidence_id, str)
                    or evidence_id not in selected_items
                ):
                    raise ValueError(
                        f"section_evidence_ids[{k!r}] contains unknown id "
                        f"{evidence_id!r}"
                    )
                if evidence_id in ids:
                    raise ValueError(
                        f"section_evidence_ids[{k!r}] contains duplicate id "
                        f"{evidence_id!r}"
                    )
                ids.append(evidence_id)

            section_evidence_ids[k] = ids

        raw_method = value.get("method_by_section", {})

        if not isinstance(raw_method, Mapping):
            raise ValueError(
                f"method_by_section must be Mapping, got {type(raw_method).__name__}"
            )

        method_by_section: dict[str, str] = {}
        for k, v in raw_method.items():
            if not isinstance(k, str):
                raise ValueError(f"method_by_section key must be str, got {k!r}")
            if v not in {"cross_encoder", "lexical_fallback"}:
                raise ValueError(f"invalid retrieval method for section {k!r}: {v!r}")
            method_by_section[k] = v

        if set(method_by_section) - set(section_evidence_ids):
            raise ValueError("method_by_section contains an unknown section")
        for section_key, ids in section_evidence_ids.items():
            if ids and section_key not in method_by_section:
                raise ValueError(
                    f"section {section_key!r} has evidence ids but no retrieval method"
                )

        oc = value.get("omitted_count")
        if not isinstance(oc, int) or oc < 0:
            raise ValueError(f"omitted_count must be int >= 0, got {oc!r}")

        et = value.get("estimated_tokens")
        if not isinstance(et, int) or et < 0:
            raise ValueError(f"estimated_tokens must be int >= 0, got {et!r}")

        if cand_count < len(selected_items):
            raise ValueError(
                "candidate_count cannot be smaller than selected item count"
            )

        if oc != cand_count - len(selected_items):
            raise ValueError(
                "omitted_count must equal candidate_count - selected item count"
            )

        return cls(
            candidate_count=cand_count,
            selected_items=selected_items,
            section_evidence_ids=section_evidence_ids,
            method_by_section=method_by_section,
            omitted_count=oc,
            estimated_tokens=et,
        )


class EvidenceSelector:
    """Rank evidence per section and pack it within a shared budget."""

    def __init__(
        self,
        reranker: Reranker | None,
        budget: BudgetManager,
        *,
        top_k_per_section: int = 12,
        max_items: int = 60,
        per_paper_cap: int = 3,
        trace_hook: Callable[[str, dict], None] | None = None,
    ) -> None:
        """Initialize the evidence selector."""
        if top_k_per_section <= 0:
            raise ValueError("top_k_per_section must be positive")

        if max_items <= 0:
            raise ValueError("max_items must be positive")

        if per_paper_cap <= 0:
            raise ValueError("per_paper_cap must be positive")

        self._reranker = reranker
        self._budget = budget
        self._top_k_per_section = top_k_per_section
        self._max_items = max_items
        self._per_paper_cap = per_paper_cap
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event."""
        if self._trace_hook is None:
            return

        try:
            self._trace_hook(event, data)
        except Exception:
            logger.debug("trace_hook failed for %s", event, exc_info=True)

    async def select(
        self,
        query: str,
        ledger: dict[str, dict[str, Any]],
        sections: tuple[SectionSpec, ...] = DEFAULT_SECTIONS,
        *,
        max_tokens: int,
    ) -> EvidenceSelection:
        """Rank and round-robin pack ledger items within the token budget."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        keys = [s.key for s in sections]

        if any(not k for k in keys):
            raise ValueError("section key must be non-empty")
        if len(keys) != len(set(keys)):
            raise ValueError("section keys must be unique")

        operation_id = uuid.uuid4().hex[:12]
        task_id = get_task_id()
        t0 = time.perf_counter()

        self._emit(
            "evidence.retrieve.start",
            {
                "operation_id": operation_id,
                "task_id": task_id,
                "query": query,
                "section_keys": keys,
                "candidate_count": len(ledger),
                "top_k_per_section": self._top_k_per_section,
                "max_items": self._max_items,
                "max_tokens": max_tokens,
            },
        )

        try:
            result = await self._select_inner(query, ledger, sections, max_tokens, keys)
        except asyncio.CancelledError:
            elapsed = int((time.perf_counter() - t0) * 1000)
            self._emit(
                "evidence.retrieve.failed",
                {
                    "operation_id": operation_id,
                    "task_id": task_id,
                    "error_type": "CancelledError",
                    "reason_code": "cancelled",
                    "elapsed_ms": elapsed,
                },
            )
            raise
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            self._emit(
                "evidence.retrieve.failed",
                {
                    "operation_id": operation_id,
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                    "reason_code": "evidence_selection_error",
                    "elapsed_ms": elapsed,
                },
            )
            raise

        elapsed = int((time.perf_counter() - t0) * 1000)
        self._emit(
            "evidence.retrieve.complete",
            {
                "operation_id": operation_id,
                "task_id": task_id,
                "candidate_count": result.candidate_count,
                "selected_items": result.selected_items,
                "section_evidence_ids": result.section_evidence_ids,
                "method_by_section": result.method_by_section,
                "selected_count": len(result.selected_items),
                "omitted_count": result.omitted_count,
                "estimated_tokens": result.estimated_tokens,
                "elapsed_ms": elapsed,
            },
        )
        return result

    async def _select_inner(
        self,
        query: str,
        ledger: dict[str, dict[str, Any]],
        sections: tuple[SectionSpec, ...],
        max_tokens: int,
        section_keys: list[str],
    ) -> EvidenceSelection:

        # Reject malformed ledger entries before ranking.
        """Select evidence within section and token-budget constraints."""
        valid_items: list[tuple[str, dict[str, Any]]] = []

        for eid, item in ledger.items():
            if not isinstance(item, Mapping):
                continue

            if not eid or not _EVIDENCE_ID_RE.match(eid):
                continue

            if item.get("evidence_id") != eid:
                continue

            text = item.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            valid_items.append((eid, dict(item)))

        item_by_id: dict[str, dict[str, Any]] = dict(valid_items)

        if not valid_items:
            return EvidenceSelection(
                candidate_count=0,
                selected_items={},
                section_evidence_ids={k: [] for k in section_keys},
                method_by_section={},
                omitted_count=0,
                estimated_tokens=0,
            )

        # Rank each section independently, falling back per section.
        section_ranked: dict[str, list[str]] = {}
        method_by_section: dict[str, str] = {}

        for section in sections:
            section_query = f"{query}\n{section.title}\n{section.retrieval_hint}"

            if self._reranker is not None and query:
                try:
                    docs = [
                        ScoredDoc(
                            doc=Document(
                                page_content=(
                                    f"{item.get('paper_title', '')}\n"
                                    f"{item.get('text', '')}"
                                ),
                                metadata={_SOURCE_INDEX: idx, "evidence_id": eid},
                            ),
                            score=0.0,
                        )
                        for idx, (eid, item) in enumerate(valid_items)
                    ]

                    ranked = await asyncio.to_thread(
                        self._reranker.rerank,
                        section_query,
                        docs,
                    )

                    # Reject partial or ambiguous reranker output before using IDs.
                    if not isinstance(ranked, list) or not ranked:
                        raise ValueError("reranker returned no ranked documents")

                    eids: list[str] = []
                    seen: set[str] = set()

                    for scored_doc in ranked:
                        if not isinstance(scored_doc, ScoredDoc):
                            raise ValueError("reranker returned a non-ScoredDoc value")

                        if not isinstance(scored_doc.doc, Document):
                            raise ValueError(
                                "reranker returned ScoredDoc without Document"
                            )

                        eid = scored_doc.doc.metadata.get("evidence_id")

                        if not isinstance(eid, str) or eid not in item_by_id:
                            raise ValueError("reranker returned a unknown evidence id")

                        if eid in seen:
                            raise ValueError(
                                "reranker returned a duplicate evidence id"
                            )

                        seen.add(eid)
                        eids.append(eid)

                        if len(eids) >= self._top_k_per_section:
                            break

                    section_ranked[section.key] = eids
                    method_by_section[section.key] = "cross_encoder"

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Reranker failed for section %s: %s, falling back to lexical",
                        section.key,
                        type(exc).__name__,
                    )

                    section_ranked[section.key] = self._lexical_rank_section(
                        section_query, valid_items
                    )
                    method_by_section[section.key] = "lexical_fallback"

            else:
                section_ranked[section.key] = self._lexical_rank_section(
                    section_query, valid_items
                )
                method_by_section[section.key] = "lexical_fallback"

        # Pack round-robin while enforcing item, paper, and token caps.
        selected: dict[str, dict[str, Any]] = {}
        section_eids: dict[str, list[str]] = {k: [] for k in section_keys}
        paper_counts: dict[str, int] = {}
        estimated_tokens = 0

        section_iters = {s.key: iter(section_ranked.get(s.key, [])) for s in sections}
        exhausted: set[str] = set()

        while len(selected) < self._max_items and len(exhausted) < len(sections):
            for section in sections:
                if section.key in exhausted:
                    continue

                try:
                    eid = next(section_iters[section.key])
                except StopIteration:
                    exhausted.add(section.key)
                    continue

                if eid in selected:
                    if eid not in section_eids[section.key]:
                        section_eids[section.key].append(eid)
                    continue

                if len(selected) >= self._max_items:
                    break

                item = item_by_id[eid]

                paper_id = item.get("paper_id") or eid.split(":")[0]
                if paper_counts.get(paper_id, 0) >= self._per_paper_cap:
                    continue

                # Include serialized XML overhead in the budget trial.
                trial = EvidenceSelection(
                    candidate_count=len(valid_items),
                    selected_items={**selected, eid: item},
                    section_evidence_ids={
                        **{k: list(v) for k, v in section_eids.items()},
                        section.key: section_eids[section.key] + [eid],
                    },
                    method_by_section=method_by_section,
                    omitted_count=0,
                    estimated_tokens=0,
                )

                formatted = format_evidence_selection(trial)
                new_tokens = self._budget.count_tokens(formatted)
                if new_tokens > max_tokens:
                    continue

                selected[eid] = item
                section_eids[section.key].append(eid)
                paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
                estimated_tokens = new_tokens

        omitted = len(valid_items) - len(selected)

        return EvidenceSelection(
            candidate_count=len(valid_items),
            selected_items=selected,
            section_evidence_ids=section_eids,
            method_by_section=method_by_section,
            omitted_count=max(0, omitted),
            estimated_tokens=estimated_tokens,
        )

    def _lexical_rank_section(
        self, section_query: str, valid_items: list[tuple[str, dict[str, Any]]]
    ) -> list[str]:
        """Rank by token coverage, breaking ties by ledger order."""
        query_tokens = set(_WORD_RE.findall(section_query.lower()))
        scored: list[tuple[float, int, str]] = []

        for idx, (eid, item) in enumerate(valid_items):
            title = str(item.get("paper_title") or "")
            text = str(item.get("text") or "")
            title_tokens = set(_WORD_RE.findall(title.lower()))
            text_tokens = set(_WORD_RE.findall(text.lower()))

            if query_tokens:
                title_cov = len(query_tokens & title_tokens) / len(query_tokens)
                text_cov = len(query_tokens & text_tokens) / len(query_tokens)
                score = 2.0 * title_cov + text_cov
            else:
                score = 0.0

            scored.append((score, idx, eid))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [eid for _, _, eid in scored[: self._top_k_per_section]]


def format_evidence_selection(selection: EvidenceSelection) -> str:
    """Format escaped evidence-plan and evidence-ledger XML."""
    if not selection.selected_items:
        return "<evidence_plan></evidence_plan>\n<evidence_ledger></evidence_ledger>"

    plan_parts: list[str] = []

    for section_key, eids in selection.section_evidence_ids.items():
        if eids:
            escaped_key = html.escape(section_key, quote=True)
            refs = " ".join(f"[E:{html.escape(eid, quote=True)}]" for eid in eids)
            plan_parts.append(f'  <section key="{escaped_key}">{refs}</section>')

    ledger_parts: list[str] = []
    for eid, item in selection.selected_items.items():
        escaped_eid = html.escape(eid, quote=True)
        escaped_title = html.escape(str(item.get("paper_title", "")), quote=True)
        escaped_text = html.escape(str(item.get("text", "")), quote=True)
        ledger_parts.append(f"[E:{escaped_eid}] ({escaped_title}) {escaped_text}")

    plan = (
        "<evidence_plan>\n" + "\n".join(plan_parts) + "\n</evidence_plan>"
        if plan_parts
        else "<evidence_plan></evidence_plan>"
    )

    ledger = "<evidence_ledger>\n" + "\n".join(ledger_parts) + "\n</evidence_ledger>"

    return plan + "\n" + ledger
