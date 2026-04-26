"""Anthropic provider — Claude messages API.

Anthropic does not (currently) provide a public embeddings endpoint, so this
module exposes only an LLM. Combine with Gemini, OpenAI, or local embeddings
if you choose Anthropic for generation.
"""

from __future__ import annotations

from functools import cached_property

from rag.config import settings
from rag.utils.logging import get_logger
from rag.utils.retry import retry_with_backoff

log = get_logger(__name__)


def _client():  # type: ignore[no-untyped-def]
    """Build and return a configured Anthropic client. Imported lazily."""
    from anthropic import Anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or change LLM_PROVIDER."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


class AnthropicLLM:
    """Claude model wrapped in the :class:`LLMProvider` Protocol."""

    _DEFAULT_MODEL = "claude-3-5-haiku-latest"

    def __init__(self, model: str | None = None) -> None:
        """Construct an Anthropic LLM provider.

        Args:
            model: Override the model name. Falls back to ``settings.llm_model``
                if it references a Claude model, otherwise ``claude-3-5-haiku-latest``.
        """
        configured = (model or settings.llm_model or "").lower()
        self.model = model or (
            settings.llm_model if "claude" in configured else self._DEFAULT_MODEL
        )

    @cached_property
    def _anthropic_client(self):  # type: ignore[no-untyped-def]
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
        message = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # ``message.content`` is a list of content blocks; for text-only prompts
        # we expect a single TextBlock.
        parts: list[str] = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()
