"""FastAPI application entrypoint — Workshop 6.

Run locally::

    uv run uvicorn rag.api.main:app --reload

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""

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
    log.info("Initialising RAG pipeline...")
    embeddings = get_embeddings()
    llm = get_llm()
    store = ChromaStore(embeddings=embeddings)
    retriever = Retriever(store=store)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    app.state.embeddings = embeddings
    app.state.llm = llm
    app.state.store = store
    app.state.retriever = retriever
    app.state.pipeline = pipeline
    log.info("RAG pipeline ready (indexed_chunks=%d).", store.count())
    try:
        yield
    finally:
        log.info("Shutting down RAG pipeline.")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Defined as a factory so tests can build a fresh app per session.
    """
    app = FastAPI(
        title="OAF RAG API",
        version="0.1.0",
        description="Document Q&A built on top of a Retrieval-Augmented Generation pipeline.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
"""Module-level app instance used by ``uvicorn rag.api.main:app``."""
