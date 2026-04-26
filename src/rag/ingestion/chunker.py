"""Text chunking — Workshop 2.

Fixed-size chunking with character-level overlap. This is the simplest strategy
that *works*: it doesn't respect semantics, but it's deterministic, predictable,
and a good baseline before students explore smarter splitters.
"""

from __future__ import annotations

from dataclasses import replace

from rag.config import settings
from rag.ingestion.models import Chunk, Document


class Chunker:
    """Splits Documents into fixed-size, overlapping :class:`Chunk` objects."""

    def __init__(
        self,
        size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        """Construct a chunker.

        Args:
            size: Maximum characters per chunk. Defaults to ``settings.chunk_size``.
            overlap: Characters of overlap between consecutive chunks.
                Defaults to ``settings.chunk_overlap``. Must be < ``size``.
        """
        self.size = size if size is not None else settings.chunk_size
        self.overlap = overlap if overlap is not None else settings.chunk_overlap
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split each document into fixed-size, overlapping chunks.

        Args:
            documents: Documents to chunk.

        Returns:
            All resulting chunks, in document → in-document order.
        """
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self._chunk_one(doc))
        return chunks

    def _chunk_one(self, doc: Document) -> list[Chunk]:
        """Chunk a single Document. Empty/whitespace docs return no chunks."""
        text = doc.text or ""
        if not text.strip():
            return []

        step = self.size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.size, n)
            piece = text[start:end].strip()
            if piece:
                meta = dict(doc.metadata)
                meta.update({"chunk_index": idx, "char_start": start, "char_end": end})
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        chunk_index=idx,
                        metadata=meta,
                    )
                )
                idx += 1
            if end >= n:
                break
            start += step
        return chunks

    @staticmethod
    def renumber(chunks: list[Chunk]) -> list[Chunk]:
        """Return ``chunks`` with ``chunk_index`` re-assigned globally (0..N-1).

        Useful when merging chunk lists from multiple chunkers.
        """
        return [
            replace(c, chunk_index=i, metadata={**c.metadata, "chunk_index": i})
            for i, c in enumerate(chunks)
        ]
