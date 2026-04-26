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
        # TODO Workshop 2:
        # - For each document, call ``_chunk_one`` and accumulate the chunks.
        # - Return the flat list.
        raise NotImplementedError("Workshop 2: implement Chunker.chunk")

    def _chunk_one(self, doc: Document) -> list[Chunk]:
        """Chunk a single Document. Empty/whitespace docs return no chunks."""
        # TODO Workshop 2:
        # - If the document text is empty/whitespace-only, return [].
        # - Walk the text in steps of ``self.size - self.overlap``, slicing out
        #   ``self.size``-character windows.
        # - Strip each piece; skip empties.
        # - For each kept piece, build a Chunk with:
        #     * ``chunk_index`` increasing from 0
        #     * metadata copied from the parent doc, plus
        #       ``chunk_index``, ``char_start``, ``char_end``.
        # - Stop when the slice reaches the end of the text.
        raise NotImplementedError("Workshop 2: implement Chunker._chunk_one")

    @staticmethod
    def renumber(chunks: list[Chunk]) -> list[Chunk]:
        """Return ``chunks`` with ``chunk_index`` re-assigned globally (0..N-1).

        Useful when merging chunk lists from multiple chunkers.
        """
        return [
            replace(c, chunk_index=i, metadata={**c.metadata, "chunk_index": i})
            for i, c in enumerate(chunks)
        ]
