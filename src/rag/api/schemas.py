"""Pydantic request/response models for the FastAPI surface — Workshop 6."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(..., min_length=1, description="User question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of chunks to retrieve.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional override for LLM sampling temperature.",
    )


class Source(BaseModel):
    """One source chunk surfaced alongside an answer."""

    source: str = Field(..., description="Origin of the chunk (typically a file path).")
    chunk_index: int = Field(..., ge=0, description="0-based chunk position in its source.")
    text: str = Field(..., description="The chunk's text content.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response body for ``POST /ask``."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    question: str


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``."""

    path: str | None = Field(
        default=None,
        description="Override the path to ingest. Defaults to settings.data_dir.",
    )
    clear: bool = Field(
        default=False,
        description="If True, wipe the collection before re-ingesting.",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=100,
        le=2000,
        description="Override chunk size. Defaults to settings.chunk_size.",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        le=200,
        description="Override chunk overlap. Defaults to settings.chunk_overlap.",
    )


class IngestResponse(BaseModel):
    """Response body for ``POST /ingest``."""

    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    path: str


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    llm_provider: str
    embeddings_provider: str
    indexed_chunks: int