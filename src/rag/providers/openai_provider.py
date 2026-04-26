"""OpenAI provider — chat-completions LLM and ``text-embedding-3-small`` embeddings.

**Workshop 7 stretch.** Implementation parallels :mod:`rag.providers.gemini`.
Implementing this proves the central pedagogical point of the codebase: once
provider implementations satisfy the Protocols, swapping is a one-line .env
change.
"""

from __future__ import annotations

from functools import cached_property

from rag.config import settings
from rag.utils.logging import get_logger
from rag.utils.retry import retry_with_backoff

log = get_logger(__name__)


def _client():  # type: ignore[no-untyped-def]
    """Build and return a configured OpenAI client. Imported lazily."""
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or change LLM_PROVIDER."
        )
    return OpenAI(api_key=settings.openai_api_key)


class OpenAILLM:
    """OpenAI chat model exposed through the :class:`LLMProvider` Protocol."""

    def __init__(self, model: str | None = None) -> None:
        """Construct an OpenAI LLM provider.

        Args:
            model: Override the model name. Defaults to ``settings.llm_model``
                if it looks like a GPT model, otherwise ``gpt-4o-mini``.
        """
        # TODO Workshop 7 (stretch):
        # - Resolve the model name. If ``settings.llm_model`` references a GPT
        #   model, use it; otherwise fall back to ``"gpt-4o-mini"``.
        raise NotImplementedError("Workshop 7 stretch: implement OpenAILLM.__init__")

    @cached_property
    def _openai_client(self):  # type: ignore[no-untyped-def]
        return _client()

    @retry_with_backoff(retries=5, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single response for ``prompt`` using chat completions."""
        # TODO Workshop 7 (stretch):
        # - Call ``self._openai_client.chat.completions.create(model=..., messages=[
        #     {"role": "user", "content": prompt}], temperature=..., max_tokens=...)``.
        # - Return ``response.choices[0].message.content`` (stripped).
        raise NotImplementedError("Workshop 7 stretch: implement OpenAILLM.generate")


class OpenAIEmbeddings:
    """OpenAI ``text-embedding-3-small`` wrapped in :class:`EmbeddingsProvider`."""

    _DEFAULT_MODEL = "text-embedding-3-small"
    _DIMENSION = 1536

    def __init__(self, model: str | None = None) -> None:
        """Construct an OpenAI embeddings provider.

        Args:
            model: Override the embeddings model. Defaults to
                ``text-embedding-3-small`` (1536-dim).
        """
        # TODO Workshop 7 (stretch):
        # - Store the model name on ``self``.
        raise NotImplementedError("Workshop 7 stretch: implement OpenAIEmbeddings.__init__")

    @cached_property
    def _openai_client(self):  # type: ignore[no-untyped-def]
        return _client()

    @property
    def dimension(self) -> int:
        """Vector dimension (1536 for ``text-embedding-3-small``)."""
        return self._DIMENSION

    @retry_with_backoff(retries=5, base_delay=1.0)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings using the OpenAI embeddings API."""
        # TODO Workshop 7 (stretch):
        # - Guard against empty input.
        # - Call ``self._openai_client.embeddings.create(model=..., input=texts)``.
        # - Return ``[item.embedding for item in response.data]`` (each as a list).
        raise NotImplementedError("Workshop 7 stretch: implement OpenAIEmbeddings.embed")
