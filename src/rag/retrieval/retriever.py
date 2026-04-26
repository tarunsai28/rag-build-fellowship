"""Retriever — Workshop 4.

A thin layer over the vector store. Today it just delegates to
``ChromaStore.query``; the abstraction earns its keep when retrieval grows
real-world features (metadata filters, reranking, hybrid search, query
rewriting).
"""

# ruff: noqa: F401  -- imports become used once Workshop 4 is implemented.

from __future__ import annotations

from typing import Any

from rag.config import settings
from rag.ingestion.models import Chunk
from rag.vectorstore.chroma_store import ChromaStore


class Retriever:
    """Returns the most-relevant chunks for a natural-language query."""

    def __init__(self, store: ChromaStore) -> None:
        """Construct a Retriever bound to a vector store.

        Args:
            store: The :class:`ChromaStore` to query against.
        """
        self.store = store

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Return up to ``k`` chunks most relevant to ``query``.

        Args:
            query: Natural-language query string.
            k: Number of chunks to return. Defaults to ``settings.top_k``.
            filters: Optional metadata filter, passed through to Chroma's
                ``where`` clause (e.g. ``{"source": "manual.pdf"}``).

        Returns:
            Ranked list of chunks (most relevant first), possibly empty.
        """
        # TODO Workshop 4:
        # 1. Guard against empty queries (return []).
        # 2. Resolve ``k`` (default: ``settings.top_k``).
        # 3. Delegate to ``self.store.query(query, k=..., where=filters)``.
        #
        # Stretch: log the number of results retrieved; add a simple score
        # threshold; integrate a reranker.
        raise NotImplementedError("Workshop 4: implement Retriever.search")
