"""Plain dataclasses that flow through the ingestion → vectorstore → retrieval pipeline.

These types are deliberately small and simple. Students should be able to read
them in 30 seconds and have a complete mental model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single source document, before chunking.

    Attributes:
        text: Full text content of the document.
        source: Identifier for the source — usually the file path.
        metadata: Free-form key/value pairs (e.g. ``{"page_count": 12}``).
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a Document, post-chunking.

    Attributes:
        text: The chunk's text content.
        source: Source identifier inherited from the parent Document.
        chunk_index: 0-based position of this chunk within its source document.
        metadata: Free-form key/value pairs. Typically includes ``chunk_index``,
            ``char_start``, ``char_end``, plus anything inherited from the Document.
    """

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable identifier for this chunk: ``"<source>::<chunk_index>"``."""
        return f"{self.source}::{self.chunk_index}"
