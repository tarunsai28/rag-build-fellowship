"""FastAPI application entrypoint — Workshop 6.

Run locally::

    uv run uvicorn rag.api.main:app --reload

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    embeddings = get_embeddings()
    llm = get_llm()
    store = ChromaStore(embeddings=embeddings)
    retriever = Retriever(store=store)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    app.state.store = store
    app.state.pipeline = pipeline
    app.state.retriever = retriever

    log.info("RAG pipeline ready")
    yield
    log.info("Shutting down")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Defined as a factory so tests can build a fresh app per session.
    """
    app = FastAPI(
        title="OAF RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

    return app


try:
    app: FastAPI | None = create_app()
except NotImplementedError:
    app = None