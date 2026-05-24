"""Centralised settings loaded from ``.env`` and the process environment.

Every other module in the project reads configuration from the singleton
``settings`` exported here. Students change behaviour by editing ``.env`` —
they should never sprinkle ``os.getenv`` calls through the rest of the code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / ``.env``.

    Field names are case-insensitive. ``LLM_PROVIDER`` in ``.env`` populates
    ``llm_provider`` here.
    """

    # ----- Provider selection -----
    llm_provider: str = "gemini"
    embeddings_provider: str = "gemini"

    # ----- Model selection -----
    llm_model: str = "gemini-2.5-flash-lite"
    embeddings_model: str = "gemini-embedding-2"

    # ----- API keys (only the active provider's key needs to be set) -----
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # ----- Paths -----
    data_dir: Path = Path("data/sample_docs")
    chroma_persist_dir: Path = Path(".chroma")
    chroma_collection: str = "rag_chunks"

    # ----- Retrieval defaults -----
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # ----- Generation defaults -----
    temperature: float = 0.2
    max_tokens: int = 1024

    # ----- Ollama (only when an "ollama" provider is selected) -----
    ollama_base_url: str = "http://localhost:11434"
    ollama_embeddings_model: str = "nomic-embed-text"

    # ----- Local sentence-transformers fallback -----
    local_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
"""Process-wide settings instance. Import as ``from rag.config import settings``."""
