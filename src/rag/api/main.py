"""FastAPI application entrypoint — Workshop 6.

Run locally::

    uv run uvicorn rag.api.main:app --reload

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes import router
from rag.generation.pipeline import RAGPipeline
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.retriever import Retriever
from rag.utils.logging import get_logger
from rag.vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct the pipeline once at startup, tear it down at shutdown."""
    # TODO Workshop 6:
    # 1. Build embeddings and llm via ``get_embeddings()`` / ``get_llm()``.
    # 2. Build a ``ChromaStore(embeddings=embeddings)``.
    # 3. Build a ``Retriever(store=store)``.
    # 4. Build a ``RAGPipeline(retriever=retriever, llm=llm)``.
    # 5. Stash the pipeline (and any other useful objects) on ``app.state`` so
    #    your route handlers can read them via ``request.app.state``.
    # 6. ``yield`` to hand control back to FastAPI; on shutdown, log a message.
    raise NotImplementedError("Workshop 6: implement the FastAPI lifespan")
    yield  # unreachable, but keeps the function shape obvious to readers.


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Defined as a factory so tests can build a fresh app per session.
    """
    # TODO Workshop 6:
    # 1. Construct ``FastAPI(title=..., version=..., lifespan=lifespan)``.
    # 2. Add the CORS middleware (you can be permissive in dev; lock it down later).
    # 3. ``app.include_router(router)`` so the route handlers in routes.py are wired up.
    # 4. Return the app.
    raise NotImplementedError("Workshop 6: implement create_app")


try:
    app: FastAPI | None = create_app()
    """Module-level app instance used by ``uvicorn rag.api.main:app``."""
except NotImplementedError:
    # Starter: ``create_app`` is a Workshop 6 stub. Importing this module
    # should not crash before Workshop 6 (e.g. during test collection).
    app = None
