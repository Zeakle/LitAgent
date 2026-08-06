"""Provide legacy and asymmetric retrieval embedding backends."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from sentence_transformers import SentenceTransformer

from litagent.config import EmbeddingBackend, RAGConfig
from litagent.logging import get_logger
from litagent.rag.interfaces import Embedder, EmbeddingDocument, RetrievalEmbedder

logger = get_logger("rag.embedder")
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingBackendError(RuntimeError):
    """Report a stable embedding backend failure without silent fallback."""


class LocalEmbedder(Embedder):
    """Generate normalized SentenceTransformer embeddings lazily."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        model_factory: Callable[[str], Any] = SentenceTransformer,
    ) -> None:
        self._model_name = model_name
        self._model_factory = model_factory
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            logger.info("loading sentence-transformer model: %s", self._model_name)
            self._model = self._model_factory(self._model_name)
        return self._model

    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """Preserve the legacy single-or-batch input shape."""
        model = self._ensure_model()
        single = isinstance(texts, str)
        batch = [texts] if single else texts
        vectors = model.encode(batch, normalize_embeddings=True)
        encoded = [vector.tolist() for vector in vectors]
        return encoded[0] if single else encoded

    def embed_documents(
        self,
        documents: Sequence[EmbeddingDocument],
    ) -> list[list[float]]:
        """Embed chunk text while preserving the established MiniLM baseline."""
        if not documents:
            return []
        vectors = self.embed([document.text for document in documents])
        return list(vectors)

    def embed_query(self, query: str) -> list[float]:
        """Embed one query with the same symmetric model."""
        vector = self.embed(query)
        return list(vector)

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return int(self._ensure_model().get_embedding_dimension())


class Specter2Embedder:
    """Use separate SPECTER2 adapters for papers and short queries."""

    def __init__(
        self,
        *,
        model_name: str,
        document_adapter: str,
        query_adapter: str,
        tokenizer_factory: Callable[[str], Any] | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._document_adapter = document_adapter
        self._query_adapter = query_adapter
        self._tokenizer_factory = tokenizer_factory
        self._model_factory = model_factory
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._document_adapter_name = "litagent_document"
        self._query_adapter_name = "litagent_query"
        # Adapter activation and forward must be one critical section.
        self._lock = threading.RLock()

    def _ensure_model(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model
            try:
                if self._tokenizer_factory is None or self._model_factory is None:
                    from adapters import AutoAdapterModel
                    from transformers import AutoTokenizer

                    tokenizer_factory = AutoTokenizer.from_pretrained
                    model_factory = AutoAdapterModel.from_pretrained
                else:
                    tokenizer_factory = self._tokenizer_factory
                    model_factory = self._model_factory

                tokenizer = tokenizer_factory(self._model_name)
                model = model_factory(self._model_name)
                model.load_adapter(
                    self._document_adapter,
                    source="hf",
                    load_as=self._document_adapter_name,
                    set_active=False,
                )
                model.load_adapter(
                    self._query_adapter,
                    source="hf",
                    load_as=self._query_adapter_name,
                    set_active=False,
                )
                model.eval()
            except ImportError as exc:
                raise EmbeddingBackendError(
                    "SPECTER2 requires the optional benchmark dependencies; "
                    'install with pip install -e ".[benchmark]"'
                ) from exc
            except Exception as exc:
                raise EmbeddingBackendError(
                    "failed to load SPECTER2 model or adapters: "
                    f"{type(exc).__name__}"
                ) from exc
            self._tokenizer = tokenizer
            self._model = model
            return tokenizer, model

    def _encode(self, texts: Sequence[str], *, adapter_name: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise EmbeddingBackendError("SPECTER2 requires PyTorch") from exc

        with self._lock:
            tokenizer, model = self._ensure_model()
            model.set_active_adapters(adapter_name)
            inputs = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            device = next(model.parameters()).device
            inputs = {name: value.to(device) for name, value in inputs.items()}
            try:
                with torch.no_grad():
                    output = model(**inputs)
                    embeddings = functional.normalize(
                        output.last_hidden_state[:, 0, :],
                        p=2,
                        dim=1,
                    )
            except Exception as exc:
                raise EmbeddingBackendError(
                    f"SPECTER2 inference failed: {type(exc).__name__}"
                ) from exc
            return embeddings.detach().cpu().tolist()

    def embed_documents(
        self,
        documents: Sequence[EmbeddingDocument],
    ) -> list[list[float]]:
        """Encode title plus chunk text with the proximity adapter."""
        tokenizer, _ = self._ensure_model()
        separator = tokenizer.sep_token or "[SEP]"
        texts = [
            f"{document.title}{separator}{document.text}" for document in documents
        ]
        return self._encode(texts, adapter_name=self._document_adapter_name)

    def embed_query(self, query: str) -> list[float]:
        """Encode one short query with the adhoc-query adapter."""
        vectors = self._encode([query], adapter_name=self._query_adapter_name)
        return vectors[0]

    @property
    def dim(self) -> int:
        """Read the base model hidden size without an inference call."""
        _, model = self._ensure_model()
        return int(model.config.hidden_size)


def build_retrieval_embedder(config: RAGConfig) -> RetrievalEmbedder:
    """Build the configured paper retrieval embedder without fallback."""
    if config.embedding_backend is EmbeddingBackend.SENTENCE_TRANSFORMER:
        return LocalEmbedder(config.embedding_model)
    if config.embedding_backend is EmbeddingBackend.SPECTER2:
        if not config.embedding_document_adapter or not config.embedding_query_adapter:
            raise EmbeddingBackendError("SPECTER2 adapter configuration is incomplete")
        return Specter2Embedder(
            model_name=config.embedding_model,
            document_adapter=config.embedding_document_adapter,
            query_adapter=config.embedding_query_adapter,
        )
    raise EmbeddingBackendError(
        f"unsupported embedding backend: {config.embedding_backend}"
    )


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the shared legacy embedding backend."""
    global _embedder
    if not _embedder:
        _embedder = LocalEmbedder()
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    """Replace the shared legacy embedding backend."""
    global _embedder
    _embedder = embedder
