"""Integration tests for the FastAPI surface, using mock providers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import main as api_main
from rag.api.main import create_app
from rag.config import settings
from rag.generation.pipeline import RAGPipeline
from rag.ingestion.models import Chunk
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore
from tests.conftest import MockEmbeddings, MockLLM


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Spin up a FastAPI app whose lifespan uses mock providers and a temp store."""
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / ".chroma")
    monkeypatch.setattr(settings, "chroma_collection", "api_test")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "docs")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text(
        "RAG stands for Retrieval-Augmented Generation.", encoding="utf-8"
    )

    # Patch the factory functions where ``rag.api.main`` already imported them.
    monkeypatch.setattr(api_main, "get_llm", lambda: MockLLM("MOCK"))
    monkeypatch.setattr(api_main, "get_embeddings", lambda: MockEmbeddings())

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "indexed_chunks" in body


def test_ingest_then_ask_round_trip(client: TestClient) -> None:
    ingest = client.post("/ingest", json={"clear": True})
    assert ingest.status_code == 200, ingest.text
    body = ingest.json()
    assert body["documents_loaded"] == 1
    assert body["chunks_created"] >= 1

    ask = client.post("/ask", json={"question": "What does RAG stand for?"})
    assert ask.status_code == 200, ask.text
    answer = ask.json()
    assert answer["answer"]
    assert answer["question"] == "What does RAG stand for?"


def test_ask_with_empty_store_still_works(client: TestClient) -> None:
    """Asking a question against an empty store should not 500."""
    # Skip ingest. The pipeline should produce *some* answer from no context.
    response = client.post("/ask", json={"question": "anything?"})
    assert response.status_code == 200
    assert "answer" in response.json()


def test_ingest_validates_path(client: TestClient) -> None:
    response = client.post("/ingest", json={"path": "/no/such/path/exists"})
    assert response.status_code == 404


def test_ingest_handles_extra_chunks(client: TestClient) -> None:
    """Smoke check that direct add() also works (sanity, not via API)."""
    chunk = Chunk(text="RAG = Retrieval Augmented Generation", source="x", chunk_index=0)
    pipeline: RAGPipeline = client.app.state.pipeline
    retriever: Retriever = client.app.state.retriever
    store: ChromaStore = client.app.state.store
    store.add([chunk])
    results = retriever.search("Retrieval Augmented Generation", k=1)
    assert results
    assert pipeline.llm is not None
