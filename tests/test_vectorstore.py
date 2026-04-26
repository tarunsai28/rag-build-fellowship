"""Tests for the ChromaStore wrapper.

Skipped in the starter — un-skip and complete in Workshop 7.
"""

from __future__ import annotations

import pytest

from rag.ingestion.models import Chunk
from rag.vectorstore.chroma_store import ChromaStore


@pytest.mark.skip(reason="Workshop 7: round-trip add() then query().")
def test_add_then_query_round_trips(chroma_store: ChromaStore) -> None:
    chunks = [
        Chunk(text="apples are red fruit", source="fruit.md", chunk_index=0),
        Chunk(text="bananas are yellow fruit", source="fruit.md", chunk_index=1),
    ]
    chroma_store.add(chunks)

    assert chroma_store.count() == 2
    results = chroma_store.query("apples are red fruit", k=1)
    assert results
    assert results[0].text == "apples are red fruit"
    assert results[0].source == "fruit.md"
    assert results[0].chunk_index == 0


@pytest.mark.skip(reason="Workshop 7: empty-query handling.")
def test_query_empty_string_returns_empty(chroma_store: ChromaStore) -> None:
    assert chroma_store.query("", k=4) == []


@pytest.mark.skip(reason="Workshop 7: clear() wipes the collection.")
def test_clear_wipes_collection(chroma_store: ChromaStore) -> None:
    chroma_store.add([Chunk(text="x", source="s.txt", chunk_index=0)])
    assert chroma_store.count() == 1
    chroma_store.clear()
    assert chroma_store.count() == 0


@pytest.mark.skip(reason="Workshop 7: metadata filtering.")
def test_metadata_filter_constrains_results(chroma_store: ChromaStore) -> None:
    chroma_store.add(
        [
            Chunk(text="alpha", source="a.txt", chunk_index=0, metadata={"topic": "a"}),
            Chunk(text="beta", source="b.txt", chunk_index=0, metadata={"topic": "b"}),
        ]
    )
    results = chroma_store.query("alpha", k=2, where={"topic": "b"})
    assert len(results) == 1
    assert results[0].source == "b.txt"
