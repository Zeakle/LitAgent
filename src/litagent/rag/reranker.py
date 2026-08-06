"""Provide cross-encoder and pass-through paper rerankers."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from litagent.rag.interfaces import Reranker, ScoredDoc
from litagent.rag.models import ScoredPaperHit

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_paper_rerank_text(
    paper: ScoredPaperHit,
    *,
    max_chars: int = 4000,
) -> str:
    """Build one bounded paper-level reranking document."""
    parts = [paper.title.strip(), paper.abstract.strip()]
    parts.extend(chunk.text.strip() for chunk in paper.chunks if chunk.text.strip())
    text = "\n".join(part for part in parts if part)
    return text[:max_chars]


class CrossEncoderReranker(Reranker):
    """Rerank document or parent-paper candidates with a CrossEncoder."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        model: Any | None = None,
        max_paper_chars: int = 4000,
    ) -> None:
        self.model_name = model_name
        self._model = model or CrossEncoder(model_name)
        self._max_paper_chars = max_paper_chars

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        """Rerank legacy document candidates."""
        if not docs:
            return docs
        scores = self._model.predict(
            [(query, item.doc.page_content[:1000]) for item in docs]
        )
        rescored = [
            ScoredDoc(doc=item.doc, score=float(score))
            for item, score in zip(docs, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)

    def rerank_papers(
        self,
        query: str,
        papers: list[ScoredPaperHit],
    ) -> list[ScoredPaperHit]:
        """Rerank unique parent papers after chunk aggregation."""
        if not papers:
            return papers
        scores = self._model.predict(
            [
                (
                    query,
                    build_paper_rerank_text(
                        paper,
                        max_chars=self._max_paper_chars,
                    ),
                )
                for paper in papers
            ]
        )
        rescored = [
            paper.model_copy(update={"score": float(score)})
            for paper, score in zip(papers, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)


class NoopReranker(Reranker):
    """Preserve incoming retrieval order."""

    model_name = "noop"

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        return docs

    def rerank_papers(
        self,
        query: str,
        papers: list[ScoredPaperHit],
    ) -> list[ScoredPaperHit]:
        return papers
