"""Provide a lazily loaded local embedder and shared backend instance."""

from sentence_transformers import SentenceTransformer

from litagent.rag.interfaces import Embedder
from litagent.logging import get_logger

logger = get_logger("rag.embedder")

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder(Embedder):
    """Generate normalized embeddings with a SentenceTransformer model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _ensure_model(self):
        if self._model is None:
            logger.info(f"loading model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one string or a batch while preserving the input shape."""
        self._ensure_model()
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors[0].tolist() if single else [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        self._ensure_model()
        return self._model.get_embedding_dimension()


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the shared embedding backend."""
    global _embedder
    if not _embedder:
        _embedder = LocalEmbedder()
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    """Replace the shared embedding backend."""
    global _embedder
    _embedder = embedder
