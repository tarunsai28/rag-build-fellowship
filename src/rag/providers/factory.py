"""Factory functions: turn ``settings.llm_provider`` / ``settings.embeddings_provider``
into a concrete provider instance.

This is the only place in the codebase that imports every concrete provider.
Everywhere else depends on the :class:`LLMProvider` / :class:`EmbeddingsProvider`
Protocols and uses these factories.
"""

from __future__ import annotations

from rag.config import settings
from rag.providers.base import EmbeddingsProvider, LLMProvider


def get_llm() -> LLMProvider:
    """Return a concrete :class:`LLMProvider` based on ``settings.llm_provider``.

    Raises:
        ValueError: If ``settings.llm_provider`` is not a recognised value.
    """
    provider = settings.llm_provider.lower()
    match provider:
        case "gemini":
            from rag.providers.gemini import GeminiLLM

            return GeminiLLM()
        case "openai":
            from rag.providers.openai_provider import OpenAILLM

            return OpenAILLM()
        case "anthropic":
            from rag.providers.anthropic_provider import AnthropicLLM

            return AnthropicLLM()
        case "ollama":
            from rag.providers.ollama import OllamaLLM

            return OllamaLLM()
        case _:
            raise ValueError(
                f"Unknown llm_provider {provider!r}. "
                "Expected one of: gemini, openai, anthropic, ollama."
            )


def get_embeddings() -> EmbeddingsProvider:
    """Return a concrete :class:`EmbeddingsProvider` based on ``settings.embeddings_provider``.

    Raises:
        ValueError: If ``settings.embeddings_provider`` is not recognised.
    """
    provider = settings.embeddings_provider.lower()
    match provider:
        case "gemini":
            from rag.providers.gemini import GeminiEmbeddings

            return GeminiEmbeddings()
        case "openai":
            from rag.providers.openai_provider import OpenAIEmbeddings

            return OpenAIEmbeddings()
        case "ollama":
            from rag.providers.ollama import OllamaEmbeddings

            return OllamaEmbeddings()
        case "local":
            from rag.providers.local import SentenceTransformersEmbeddings

            return SentenceTransformersEmbeddings()
        case _:
            raise ValueError(
                f"Unknown embeddings_provider {provider!r}. "
                "Expected one of: gemini, openai, ollama, local."
            )
