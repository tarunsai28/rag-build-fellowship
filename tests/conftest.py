"""Pytest fixtures shared across the test suite.

Tests must never hit real Gemini / OpenAI / Anthropic — they use the
``MockLLM`` and ``MockEmbeddings`` providers below. These intentionally
satisfy the :class:`LLMProvider` and :class:`EmbeddingsProvider` Protocols
without inheriting from anything (Protocols are duck-typed).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.generation.pipeline import RAGPipeline
from rag.ingestion.models import Chunk, Document
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


class MockLLM:
    """Deterministic mock LLM. Echoes the first 80 chars of the prompt."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        if self.response is not None:
            return self.response
        return f"MOCK_ANSWER for prompt: {prompt[:80]!r}"


class MockEmbeddings:
    """Deterministic mock embeddings provider.

    Maps each input to a fixed-dimension vector seeded by an MD5 of the text.
    The vectors are NOT semantically meaningful; they're stable and unique,
    which is enough for vector-store and pipeline tests.
    """

    _DIMENSION = 16

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8")).digest()
            vec = [(b - 128) / 128.0 for b in digest[: self._DIMENSION]]
            vectors.append(vec)
        return vectors


@pytest.fixture
def mock_llm() -> MockLLM:
    """Fresh :class:`MockLLM` per test."""
    return MockLLM()


@pytest.fixture
def mock_embeddings() -> MockEmbeddings:
    """Fresh :class:`MockEmbeddings` per test."""
    return MockEmbeddings()


@pytest.fixture
def sample_document() -> Document:
    """A small, predictable document for chunker / loader tests."""
    return Document(
        text=(
            "Retrieval-Augmented Generation combines a vector search step with a "
            "language model. The retriever finds relevant context from a document "
            "store, and the language model uses that context to answer questions. "
            "ChromaDB is one popular vector store for RAG systems."
        ),
        source="sample.txt",
        metadata={"format": "txt"},
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    """A deterministic list of chunks built from :func:`sample_document`."""
    text = sample_document.text
    midpoint = len(text) // 2
    return [
        Chunk(
            text=text[:midpoint],
            source=sample_document.source,
            chunk_index=0,
            metadata={"chunk_index": 0, "char_start": 0, "char_end": midpoint},
        ),
        Chunk(
            text=text[midpoint:],
            source=sample_document.source,
            chunk_index=1,
            metadata={"chunk_index": 1, "char_start": midpoint, "char_end": len(text)},
        ),
    ]


@pytest.fixture
def chroma_store(tmp_path: Path, mock_embeddings: MockEmbeddings) -> Iterator[ChromaStore]:
    """A :class:`ChromaStore` rooted in a per-test temp directory."""
    store = ChromaStore(
        embeddings=mock_embeddings,
        persist_dir=tmp_path / ".chroma",
        collection_name="test_collection",
    )
    yield store


@pytest.fixture
def populated_store(chroma_store: ChromaStore, sample_chunks: list[Chunk]) -> ChromaStore:
    """A :class:`ChromaStore` pre-populated with :func:`sample_chunks`."""
    chroma_store.add(sample_chunks)
    return chroma_store


@pytest.fixture
def pipeline(populated_store: ChromaStore, mock_llm: MockLLM) -> RAGPipeline:
    """A :class:`RAGPipeline` built from the populated store and a mock LLM."""
    return RAGPipeline(retriever=Retriever(store=populated_store), llm=mock_llm)
