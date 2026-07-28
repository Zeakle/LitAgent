"""Provide cross-encoder and pass-through retrieval rerankers."""

from sentence_transformers import CrossEncoder

from litagent.rag.interfaces import ScoredDoc, Reranker

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    """Rerank candidates with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        """Rerank documents for the supplied query."""
        if not docs:
            return docs
        pairs = [(query, sd.doc.page_content[:1000]) for sd in docs]
        scores = self._model.predict(pairs)
        for sd, score in zip(docs, scores):
            sd.score = float(score)

        return sorted(docs, key=lambda s: s.score, reverse=True)


class NoopReranker(Reranker):
    """Preserve the incoming retrieval order without rescoring."""

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        """Return documents unchanged."""
        return docs
