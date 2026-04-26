"""CLI: ingest documents from a directory into the vector store.

Usage::

    uv run python scripts/ingest.py                 # uses settings.data_dir
    uv run python scripts/ingest.py --path data/    # custom path
    uv run python scripts/ingest.py --clear         # wipe and re-index
    uv run python scripts/ingest.py --dry-run       # chunk only, do not store

The ``--dry-run`` flag is the Workshop 2 success path: students can run it the
moment their loader + chunker work, before they've implemented the vector store.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import settings
from rag.ingestion.chunker import Chunker
from rag.ingestion.loader import DocumentLoader


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ingest script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=settings.data_dir,
        help=f"Path to ingest (default: {settings.data_dir}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Chunk size in characters (default: {settings.chunk_size}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap in characters (default: {settings.chunk_overlap}).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the collection before re-indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + chunk only; print a summary, do not embed or store.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, non-zero on error."""
    args = parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}")
        return 2

    print(f"Loading documents from {args.path}...")
    documents = DocumentLoader().load(args.path)
    print(f"Loaded {len(documents)} document(s).")

    chunker = Chunker(size=args.chunk_size, overlap=args.chunk_overlap)
    chunks = chunker.chunk(documents)
    print(f"Created {len(chunks)} chunk(s) (size={args.chunk_size}, overlap={args.chunk_overlap}).")

    if args.dry_run:
        for chunk in chunks[:3]:
            preview = chunk.text[:160].replace("\n", " ")
            print(f"  - {chunk.id}: {preview!r}")
        if len(chunks) > 3:
            print(f"  ... and {len(chunks) - 3} more.")
        print("Dry run — nothing stored.")
        return 0

    # Lazy import: only require the vectorstore + provider stack on real runs.
    from rag.providers.factory import get_embeddings
    from rag.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(embeddings=get_embeddings())
    if args.clear:
        print("Clearing existing collection...")
        store.clear()
    print("Embedding and indexing...")
    store.add(chunks)
    print(f"Done. Collection now contains {store.count()} chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
