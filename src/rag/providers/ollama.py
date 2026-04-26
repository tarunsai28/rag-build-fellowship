"""Ollama provider — fully local LLM and embeddings via the Ollama HTTP API.

Use this when:

- You're in a region where Gemini / OpenAI is blocked.
- You want fully offline operation (after the model is pulled).
- You want to demonstrate that the abstraction works for self-hosted models.

Setup::

    # macOS / Linux
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import httpx

from rag.config import settings
from rag.utils.logging import get_logger
from rag.utils.retry import retry_with_backoff

log = get_logger(__name__)


class OllamaLLM:
    """Local Ollama chat model exposed through :class:`LLMProvider`."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """Construct an Ollama LLM provider.

        Args:
            model: Ollama model tag, e.g. ``llama3.1:8b``. Defaults to
                ``settings.llm_model`` if it is non-Gemini, otherwise
                ``llama3.1:8b``.
            base_url: Ollama HTTP base URL. Defaults to ``settings.ollama_base_url``.
        """
        configured = (model or settings.llm_model or "").lower()
        if model:
            self.model = model
        elif "gemini" in configured or "gpt" in configured or "claude" in configured:
            self.model = "llama3.1:8b"
        else:
            self.model = settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    @retry_with_backoff(retries=4, base_delay=1.0)
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a single prompt to ``/api/generate`` and return the completion."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return (response.json().get("response") or "").strip()


class OllamaEmbeddings:
    """Local embeddings via Ollama's ``/api/embeddings`` endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """Construct an Ollama embeddings provider.

        Args:
            model: Ollama embeddings model tag (e.g. ``nomic-embed-text``).
            base_url: Ollama HTTP base URL.
        """
        self.model = model or settings.ollama_embeddings_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Vector dimension. Determined lazily by embedding a single token."""
        if self._dimension is None:
            vec = self.embed(["dimension probe"])[0]
            self._dimension = len(vec)
        return self._dimension

    @retry_with_backoff(retries=4, base_delay=1.0)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each input string. Ollama embeddings only accept one input per call."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                vectors.append(list(response.json()["embedding"]))
        return vectors
