"""Route handlers for the FastAPI app — Workshop 6.

Routes are registered on a ``router`` and included by ``rag.api.main``. Each
handler depends on the singleton :class:`RAGPipeline` constructed in the app's
``lifespan``; we read it back via ``request.app.state``.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

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
    # TODO Workshop 6:
    # - Read the store from ``_state(request).store``.
    # - Return a HealthResponse with status="ok", the active providers, and
    #   ``store.count()`` as the number of indexed chunks.
    raise NotImplementedError("Workshop 6: implement GET /health")


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Run the RAG pipeline against ``payload.question`` and return an answer."""
    # TODO Workshop 6:
    # 1. Read the pipeline from ``_state(request).pipeline``.
    # 2. Call ``pipeline.answer(payload.question, k=payload.top_k,
    #    temperature=payload.temperature)``.
    # 3. Convert each source ``Chunk`` into a ``Source`` schema.
    # 4. Return an ``AskResponse`` with answer, sources, and the original question.
    raise NotImplementedError("Workshop 6: implement POST /ask")


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    """Re-run ingestion against a directory or file path."""
    # TODO Workshop 6:
    # 1. Resolve the target path (``payload.path`` or ``settings.data_dir``).
    # 2. If the path doesn't exist, raise ``HTTPException(status_code=404, ...)``.
    # 3. If ``payload.clear`` is True, call ``store.clear()`` first.
    # 4. Use ``DocumentLoader().load(target)`` and ``Chunker().chunk(documents)``,
    #    then ``store.add(chunks)``.
    # 5. Return an ``IngestResponse`` with the document/chunk counts and the
    #    final ``store.count()``.
    raise NotImplementedError("Workshop 6: implement POST /ingest")
