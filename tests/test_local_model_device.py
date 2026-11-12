"""Keep local device selection compatible with injected model factories."""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from litagent.rag.embedder import LocalEmbedder
from litagent.rag.interfaces import ScoredDoc
from litagent.rag.reranker import CrossEncoderReranker


def test_embedding_uses_explicit_local_device(monkeypatch):
    monkeypatch.setenv("LITAGENT_LOCAL_MODEL_DEVICE", "cpu")
    factory = Mock()
    monkeypatch.setattr("litagent.rag.embedder.SentenceTransformer", factory)
    LocalEmbedder("test-model")._ensure_model()
    factory.assert_called_once_with("test-model", device="cpu")


def test_embedding_keeps_injected_factory_contract(monkeypatch):
    monkeypatch.setenv("LITAGENT_LOCAL_MODEL_DEVICE", "cpu")
    factory = Mock()
    LocalEmbedder("test-model", model_factory=factory)._ensure_model()
    factory.assert_called_once_with("test-model")


def test_embedding_dimension_uses_sentence_transformer_api():
    model = Mock(spec=["get_sentence_embedding_dimension"])
    model.get_sentence_embedding_dimension.return_value = 384
    assert LocalEmbedder(model_factory=lambda _: model).dim == 384


def test_reranker_uses_explicit_local_device(monkeypatch):
    monkeypatch.setenv("LITAGENT_LOCAL_MODEL_DEVICE", "cpu")
    factory = Mock()
    monkeypatch.setattr("litagent.rag.reranker.CrossEncoder", factory)
    CrossEncoderReranker("test-model")
    factory.assert_called_once_with("test-model", device="cpu")


def test_unconfigured_device_retains_automatic_selection(monkeypatch):
    monkeypatch.delenv("LITAGENT_LOCAL_MODEL_DEVICE", raising=False)
    factory = Mock()
    monkeypatch.setattr("litagent.rag.embedder.SentenceTransformer", factory)
    LocalEmbedder("test-model")._ensure_model()
    factory.assert_called_once_with("test-model", device=None)


def test_reranker_can_use_double_precision_on_local_cpu(monkeypatch):
    monkeypatch.setenv("LITAGENT_LOCAL_RERANKER_PRECISION", "float64")
    factory = Mock()
    monkeypatch.setattr("litagent.rag.reranker.CrossEncoder", factory)
    CrossEncoderReranker("test-model")
    factory.return_value.model.double.assert_called_once_with()


def test_precision_override_does_not_mutate_injected_model(monkeypatch):
    monkeypatch.setenv("LITAGENT_LOCAL_RERANKER_PRECISION", "float64")
    model = Mock()
    CrossEncoderReranker(model=model)
    model.model.double.assert_not_called()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_reranker_rejects_invalid_model_scores(score):
    model = Mock()
    model.predict.return_value = [score]
    reranker = CrossEncoderReranker(model=model)
    with pytest.raises(ValueError, match="cross_encoder_nonfinite_scores"):
        reranker.rerank(
            "query", [ScoredDoc(doc=Document(page_content="paper"), score=0.0)]
        )
