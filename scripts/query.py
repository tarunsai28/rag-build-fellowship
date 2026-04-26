"""CLI: ask one question against the indexed corpus.

Usage::

    uv run python scripts/query.py "What is RAG?"
    uv run python scripts/query.py --top-k 6 "How does ChromaDB persist data?"
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the query script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (defaults to settings.top_k).",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved source chunks alongside the answer.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success."""
    args = parse_args()

    from rag.generation.pipeline import RAGPipeline
    from rag.providers.factory import get_embeddings, get_llm
    from rag.retrieval.retriever import Retriever
    from rag.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(embeddings=get_embeddings())
    if store.count() == 0:
        print("Vector store is empty. Run: uv run python scripts/ingest.py")
        return 2

    pipeline = RAGPipeline(retriever=Retriever(store=store), llm=get_llm())
    answer = pipeline.answer(args.question, k=args.top_k)

    print("\n=== Answer ===")
    print(answer.text)
    if args.show_sources:
        print("\n=== Sources ===")
        for i, chunk in enumerate(answer.sources, start=1):
            preview = chunk.text[:240].replace("\n", " ")
            print(f"[{i}] {chunk.source} (chunk {chunk.chunk_index}): {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
