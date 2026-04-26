"""OpenAI provider — chat-completions LLM and ``text-embedding-3-small`` embeddings.

Implementation parallels :mod:`rag.providers.gemini`. Used to demonstrate the
provider abstraction in Workshop 7: switch ``LLM_PROVIDER=openai`` and the
rest of the codebase keeps working unchanged.
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
        configured = (model or settings.llm_model or "").lower()
        self.model = model or (settings.llm_model if "gpt" in configured else "gpt-4o-mini")

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
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        return text.strip()


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
        self.model = model or self._DEFAULT_MODEL

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
        if not texts:
            return []
        response = self._openai_client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]
