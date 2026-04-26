"""Local embeddings via the ``sentence-transformers`` library.

This provider runs entirely on-device. The default model
(``all-MiniLM-L6-v2``) is ~80MB and downloads on first use. CPU-friendly:
no GPU required.

This is an *optional* dependency — install with ``uv sync --extra local``.
"""

from __future__ import annotations

from functools import cached_property

from rag.config import settings
from rag.utils.logging import get_logger

log = get_logger(__name__)


class SentenceTransformersEmbeddings:
    """Embed text using a Hugging Face ``sentence-transformers`` model."""

    def __init__(self, model: str | None = None) -> None:
        """Construct a local embeddings provider.

        Args:
            model: HF model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.
                Defaults to ``settings.local_embeddings_model``.
        """
        self.model_name = model or settings.local_embeddings_model

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: uv sync --extra local"
            ) from exc
        log.info(
            "Loading sentence-transformers model %s (first call may download).", self.model_name
        )
        return SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        """Vector dimension reported by the underlying model."""
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings on-device."""
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]
