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
    """Gemini embeddings wrapped in the :class:`EmbeddingsProvider` shape.

    Defaults to ``gemini-embedding-2`` (the current AI-Studio-surfaced GA
    embedding model). The legacy ``text-embedding-004`` was retired from
    the ``v1beta`` API in 2025; if you have that name pinned in ``.env``,
    embed calls fail with ``404 NOT_FOUND``. Update ``EMBEDDINGS_MODEL``
    and wipe ``.chroma/`` before re-ingesting (the vector dimension
    changes between models).

    Note on batching: ``gemini-embedding-2``'s ``batchEmbedContents``
    endpoint returns a *single combined* vector when handed a list of
    inputs, rather than one vector per input. So we iterate one text per
    call. This is slower than true batch embedding but is correct, and the
    ``retry_with_backoff`` decorator handles rate-limit pushback from the
    free tier.
    """

    # gemini-embedding-2 returns 3072-dim vectors by default.
    _DIMENSION = 3072

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
        """Length of each embedding vector (3072 for ``gemini-embedding-2``)."""
        return self._DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings, one API call per text.

        The per-call retry policy is wrapped around :meth:`_embed_one` rather
        than this method so that a transient 429 on chunk 30 of 47 only
        retries that single chunk, not the whole batch.
        """
        return [self._embed_one(text) for text in texts]

    @retry_with_backoff(retries=5, base_delay=1.0)
    def _embed_one(self, text: str) -> list[float]:
        """Embed a single string. Returns one vector."""
        result = self._genai_client.models.embed_content(
            model=self.model,
            contents=text,
        )
        # The google-genai SDK exposes either ``embeddings`` (plural, list with
        # one element here since we passed a single string) or ``embedding``
        # (singular). Handle both shapes defensively across SDK versions.
        if hasattr(result, "embeddings") and result.embeddings:
            return list(result.embeddings[0].values)
        if hasattr(result, "embedding") and result.embedding is not None:
            return list(result.embedding.values)
        raise RuntimeError(f"Unexpected embeddings response shape: {result!r}")
