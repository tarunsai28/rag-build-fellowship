"""Provider abstraction: anything that can ``generate`` text or ``embed`` it.

Concrete implementations live in their own modules (``gemini``, ``openai_provider``,
``anthropic_provider``, ``ollama``, ``local``). Always construct providers via
``rag.providers.factory.get_llm`` / ``get_embeddings`` so the right concrete class
is selected from environment configuration.
"""

from rag.providers.base import EmbeddingsProvider, LLMProvider

__all__ = ["EmbeddingsProvider", "LLMProvider"]
