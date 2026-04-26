"""Route handlers for the FastAPI app — Workshop 6.

Routes are registered on a ``router`` and included by ``rag.api.main``. Each
handler depends on the singleton :class:`RAGPipeline` constructed in the app's
``lifespan``; we read it back via ``request.app.state``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from rag.config import settings
from rag.ingestion.chunker import Chunker
from rag.ingestion.loader import DocumentLoader

router = APIRouter()


def _state(request: Request):  # type: ignore[no-untyped-def]
    """Return the application state container set up in ``lifespan``."""
    return request.app.state


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Lightweight liveness check that also reports current configuration."""
    store = _state(request).store
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        embeddings_provider=settings.embeddings_provider,
        indexed_chunks=store.count(),
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Run the RAG pipeline against ``payload.question`` and return an answer."""
    pipeline = _state(request).pipeline
    answer = pipeline.answer(
        payload.question,
        k=payload.top_k,
        temperature=payload.temperature,
    )
    sources = [
        Source(
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            metadata=chunk.metadata,
        )
        for chunk in answer.sources
    ]
    return AskResponse(answer=answer.text, sources=sources, question=answer.question)


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    """Re-run ingestion against a directory or file path."""
    target = Path(payload.path) if payload.path else settings.data_dir
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")

    store = _state(request).store
    if payload.clear:
        store.clear()

    documents = DocumentLoader().load(target)
    chunks = Chunker().chunk(documents)
    store.add(chunks)

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=store.count(),
        path=str(target),
    )
