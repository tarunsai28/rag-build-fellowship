"""ChromaDB-backed vector store — Workshop 3.

Wraps the file-backed ``PersistentClient``. Embeddings are computed by an
injected :class:`EmbeddingsProvider`, *not* by Chroma's built-in embedder —
this keeps the abstraction visible and lets students swap providers freely.
"""

# ruff: noqa: F401  -- imports become used once Workshop 3 is implemented.

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.config import settings
from rag.ingestion.models import Chunk
from rag.providers.base import EmbeddingsProvider
from rag.utils.logging import get_logger

log = get_logger(__name__)


def _scalarize(value: Any) -> str | int | float | bool:
    """Coerce a metadata value into a Chroma-allowed scalar."""
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


class ChromaStore:
    """File-backed ChromaDB collection for chunk embeddings."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        # save the provider — we'll use it in add() and query()
        self.embeddings = embeddings

        # fall back to settings if not provided, then make sure folder exists
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or settings.chroma_collection

        # connect to chroma on disk
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # get existing collection or create a new one with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]

        # sanitize metadata values and make sure source is always present
        metadatas = [
            {**{k: _scalarize(v) for k, v in c.metadata.items()}, "source": c.source}
            for c in chunks
        ]

        # embed all chunks then upsert so re-running doesn't create duplicates
        vectors = self.embeddings.embed(documents)
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not query_text or not query_text.strip():
            return []

        # embed the question using the same model we used at index time
        query_vector = self.embeddings.embed([query_text])[0]

        # chroma returns lists-of-lists (one per query) so we grab index 0
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # rebuild Chunk objects from what chroma gave back
        chunks = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            chunks.append(Chunk(
                text=doc,
                source=meta.get("source", ""),
                chunk_index=int(meta.get("chunk_index", i)),
                metadata=meta,
            ))
        return chunks

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        # wipe the collection and start fresh
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )