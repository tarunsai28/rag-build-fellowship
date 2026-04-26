"""Gemini provider — the default LLM and embeddings backend.

Uses the official ``google-genai`` SDK (the successor to ``google-generativeai``).
Both classes are wrapped with :func:`rag.utils.retry.retry_with_backoff` because
the free tier returns ``ResourceExhausted`` (429) under any sustained load.

Why default to Gemini for this teaching project:

- Free tier with no credit card.
- One key works for both LLM *and* embeddings.
- Available in 200+ countries.
"""

from __future__ import annotations

from functools import cached_property

from rag.config import settings
from rag.utils.logging import get_logger
from rag.utils.retry import retry_with_backoff

log = get_logger(__name__)


def _client():  # type: ignore[no-untyped-def]
    """Build and return a configured ``google.genai`` client.

    Imported lazily so that ``import rag.providers`` does not require the
    SDK to be installed (helpful in tests with mock providers).
    """
    from google import genai

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )
    return genai.Client(api_key=settings.gemini_api_key)


class GeminiLLM:
    """Gemini chat model wrapped in the :class:`LLMProvider` shape."""

    def __init__(self, model: str | None = None) -> None:
        """Construct a Gemini LLM provider.

        Args:
            model: Override the default model name. Defaults to ``settings.llm_model``.
        """
        self.model = model or settings.llm_model

    @cached_property
    def _genai_client(self):  # type: ignore[no-untyped-def]
        return _client()

    @retry_with_backoff(retries=5, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single response for ``prompt``."""
        from google.genai import types

        response = self._genai_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = (response.text or "").strip()
        return text


class GeminiEmbeddings:
    """Gemini ``text-embedding-004`` wrapped in the :class:`EmbeddingsProvider` shape."""

    # text-embedding-004 returns 768-dim vectors.
    _DIMENSION = 768

    def __init__(self, model: str | None = None) -> None:
        """Construct a Gemini embeddings provider.

        Args:
            model: Override the embeddings model name. Defaults to
                ``settings.embeddings_model``.
        """
        self.model = model or settings.embeddings_model

    @cached_property
    def _genai_client(self):  # type: ignore[no-untyped-def]
        return _client()

    @property
    def dimension(self) -> int:
        """Length of each embedding vector (768 for ``text-embedding-004``)."""
        return self._DIMENSION

    @retry_with_backoff(retries=5, base_delay=1.0)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        if not texts:
            return []
        result = self._genai_client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        # The SDK returns either ``embeddings`` (list with .values) or, for a
        # single-string call, a flat list — handle both shapes defensively.
        if hasattr(result, "embeddings") and result.embeddings is not None:
            return [list(e.values) for e in result.embeddings]
        if hasattr(result, "embedding") and result.embedding is not None:
            return [list(result.embedding.values)]
        raise RuntimeError(f"Unexpected embeddings response shape: {result!r}")
