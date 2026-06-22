"""Text chunking — Workshop 2.

Fixed-size chunking with character-level overlap. This is the simplest strategy
that *works*: it doesn't respect semantics, but it's deterministic, predictable,
and a good baseline before students explore smarter splitters.
"""

# ruff: noqa: F401  -- imports become used once Workshop 2 is implemented.

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
        # Step 1: Start with an empty result list
        # Step 2: For each document, generate its chunks and add them to the list
        # Step 3: Return the complete flat list of all chunks
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self._chunk_one(doc))
        return all_chunks

    def _chunk_one(self, doc: Document) -> list[Chunk]:
        """Chunk a single Document. Empty/whitespace docs return no chunks."""
        # Step 1: Safely get the text — treat None as empty string
        text = doc.text or ""

        # Step 2: Skip documents that have no real content
        if not text.strip():
            return []

        # Step 3: Calculate how far to move forward after each chunk
        step = self.size - self.overlap
        total_length = len(text)

        result: list[Chunk] = []
        chunk_index = 0
        start = 0

        while start < total_length:
            # Step 4: Cap the end so we never slice past the text boundary
            end = min(start + self.size, total_length)
            piece = text[start:end].strip()

            # Step 5: Only keep the chunk if it has actual content
            if piece:
                # Step 6: Copy parent metadata and add chunk-specific info
                chunk_metadata = dict(doc.metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["char_start"] = start
                chunk_metadata["char_end"] = end

                # Step 7: Build the Chunk object and add to result
                result.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        chunk_index=chunk_index,
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

            # Step 8: Stop if we've reached the end
            if end >= total_length:
                break

            # Step 9: Move the window forward
            start += step

        return result

    @staticmethod
    def renumber(chunks: list[Chunk]) -> list[Chunk]:
        """Return ``chunks`` with ``chunk_index`` re-assigned globally (0..N-1).

        Useful when merging chunk lists from multiple chunkers.
        """
        return [
            replace(c, chunk_index=i, metadata={**c.metadata, "chunk_index": i})
            for i, c in enumerate(chunks)
        ]