"""Document loading from disk: PDF, plain text, and Markdown.

Workshop 2 deliverable. Students implement :meth:`DocumentLoader.load` and
the per-format helpers.
"""

# ruff: noqa: F401  -- imports become used once Workshop 2 is implemented.

from __future__ import annotations

from pathlib import Path

import pypdf

from rag.ingestion.models import Document
from rag.utils.logging import get_logger

log = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _PDF_SUFFIXES


class DocumentLoader:
    """Loads documents from a single file or recursively from a directory."""

    def load(self, path: Path) -> list[Document]:
        # Step 1: coerce to Path and check it exists
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Step 2: single file
        if path.is_file():
            doc = self._load_one(path)
            return [doc] if doc is not None else []

        # Step 3: directory — walk recursively
        docs = []
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix.lower() in _SUPPORTED_SUFFIXES:
                doc = self._load_one(file)
                if doc is not None:
                    docs.append(doc)

        # Step 4: return collected documents
        return docs

    def _load_one(self, path: Path) -> Document | None:
        try:
            suffix = path.suffix.lower()
            if suffix in _PDF_SUFFIXES:
                return self._load_pdf(path)
            elif suffix in _TEXT_SUFFIXES:
                return self._load_text(path)
            else:
                log.warning("Unsupported file type, skipping: %s", path)
                return None
        except Exception as e:
            log.error("Failed to load %s: %s", path, e)
            return None

    @staticmethod
    def _load_text(path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            text=text,
            source=str(path),
            metadata={
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            },
        )

    @staticmethod
    def _load_pdf(path: Path) -> Document:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\f\n".join(pages)
        return Document(
            text=text,
            source=str(path),
            metadata={
                "page_count": len(reader.pages),
                "size_bytes": path.stat().st_size,
            },
        )