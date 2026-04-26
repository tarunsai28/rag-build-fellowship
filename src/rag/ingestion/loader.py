"""Document loading from disk: PDF, plain text, and Markdown.

Workshop 2 deliverable. Students implement :meth:`DocumentLoader.load` and
the per-format helpers.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.models import Document
from rag.utils.logging import get_logger

log = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _PDF_SUFFIXES


class DocumentLoader:
    """Loads documents from a single file or recursively from a directory."""

    def load(self, path: Path) -> list[Document]:
        """Load documents from ``path`` (file or directory).

        - If ``path`` is a single file, returns a list with one Document.
        - If ``path`` is a directory, recursively loads every supported file.
        - Unsupported extensions are skipped with a warning (not an error).

        Args:
            path: File or directory path.

        Returns:
            List of Documents (possibly empty if no supported files found).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {path}")

        if path.is_file():
            doc = self._load_one(path)
            return [doc] if doc is not None else []

        # Directory — walk recursively, sorted for reproducibility.
        documents: list[Document] = []
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            if child.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            doc = self._load_one(child)
            if doc is not None:
                documents.append(doc)
        log.info("Loaded %d documents from %s", len(documents), path)
        return documents

    def _load_one(self, path: Path) -> Document | None:
        """Dispatch to the right format-specific loader. Returns None on failure."""
        suffix = path.suffix.lower()
        try:
            if suffix in _PDF_SUFFIXES:
                return self._load_pdf(path)
            if suffix in _TEXT_SUFFIXES:
                return self._load_text(path)
            log.warning("Skipping unsupported file type: %s", path)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Failed to load %s: %s", path, exc)
            return None

    @staticmethod
    def _load_text(path: Path) -> Document:
        """Load a plain text or Markdown file as a single Document."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            text=text,
            source=str(path),
            metadata={"format": path.suffix.lower().lstrip("."), "size_bytes": path.stat().st_size},
        )

    @staticmethod
    def _load_pdf(path: Path) -> Document:
        """Load a PDF as a single Document, joining pages with form-feeds."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_texts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            page_texts.append(extracted)
        text = "\n\f\n".join(page_texts)
        return Document(
            text=text,
            source=str(path),
            metadata={
                "format": "pdf",
                "page_count": len(reader.pages),
                "size_bytes": path.stat().st_size,
            },
        )
