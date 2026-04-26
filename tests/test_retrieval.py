"""Tests for the Retriever facade.

Skipped in the starter — un-skip and complete in Workshop 7.
"""

from __future__ import annotations

import pytest

from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


@pytest.mark.skip(reason="Workshop 7: retriever returns chunks.")
def test_retriever_returns_chunks(populated_store: ChromaStore) -> None:
    results = Retriever(store=populated_store).search("RAG combines retrieval", k=2)
    assert 1 <= len(results) <= 2
    assert all(r.text for r in results)


@pytest.mark.skip(reason="Workshop 7: retriever empty-query handling.")
def test_retriever_empty_query_returns_empty(populated_store: ChromaStore) -> None:
    assert Retriever(store=populated_store).search("   ") == []


@pytest.mark.skip(reason="Workshop 7: retriever respects k.")
def test_retriever_respects_k(populated_store: ChromaStore) -> None:
    results = Retriever(store=populated_store).search("anything", k=1)
    assert len(results) == 1
