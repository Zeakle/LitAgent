"""Reranker 实现——Cross-encoder + Noop baseline。"""

from sentence_transformers import CrossEncoder

from litagent.rag.interfaces import ScoredDoc, Reranker

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    """Cross-encoder 重排序。

    比 bi-encoder (dense embedding) 更精确——query-doc 对同时进模型。
    代价是更慢——只对 top-k 候选运行，不对全量。
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = CrossEncoder(model_name)

    
    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        if not docs:
            return docs
        pairs = [(query, sd.doc.page_content[:1000]) for sd in docs]
        scores = self._model.predict(pairs)
        for sd, score in zip(docs, scores):
            sd.score = float(score)

        return sorted(docs, key=lambda s: s.score, reverse=True)

    
class NoopReranker(Reranker):
    """透传——RAGAS baseline 评估用。"""
    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        return docs