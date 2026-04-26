"""Provider Protocols — the architectural seam of this codebase.

We use ``typing.Protocol`` rather than ``abc.ABC`` for two reasons:

1. **Duck typing.** Anything with a method of the right shape *is* a provider —
   no inheritance required. This keeps the surface area small and flexible.
2. **Pedagogy.** The point of this codebase is to teach what RAG *does*, not
   how to navigate a class hierarchy. Protocols make the contract crystal-clear:
   "an LLMProvider is an object with a ``generate(prompt) -> str`` method."

Both Protocols are decorated with ``@runtime_checkable`` so students can write
``isinstance(obj, LLMProvider)`` in tests and notebooks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can take a prompt and return a string answer."""

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return a model-generated string for ``prompt``.

        Args:
            prompt: The fully-formed prompt sent to the model.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Upper bound on generated tokens.

        Returns:
            The model's reply, with leading/trailing whitespace stripped.
        """
        ...


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Anything that can turn text into vectors.

    Implementations must be deterministic for a given input — calling
    ``embed`` twice with identical input must return numerically identical
    vectors, otherwise the vector store will silently misbehave.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string.

        Args:
            texts: List of strings to embed. Empty list returns empty list.

        Returns:
            List of vectors, one per input, each of length :attr:`dimension`.
        """
        ...

    @property
    def dimension(self) -> int:
        """Vector length produced by :meth:`embed` (constant for a model)."""
        ...
