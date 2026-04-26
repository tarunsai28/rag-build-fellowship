# Workshop guide

A per-week reading list and short reflection prompt. Pair this with
`ARCHITECTURE_SPEC.md` §6 for the full TODO list per week.

## Workshop 1 — Project Kickoff, Architecture, Environment Setup

- Read: `README.md`, `SETUP.md`, top half of `ARCHITECTURE_SPEC.md`.
- Do: install `uv`, run `uv sync`, paste your Gemini key into `.env`,
  run `uv run python scripts/verify_setup.py`.
- Reflect: where in the file tree do you expect the LLM call to live?
  Where do you expect the chunker to live?

## Workshop 2 — Document Ingestion and Chunking

- Read: `src/rag/ingestion/models.py`, `loader.py`, `chunker.py`.
- Do: implement `DocumentLoader.load` and `Chunker.chunk`. Run
  `uv run python scripts/ingest.py --dry-run`.
- Reflect: what happens to chunk boundaries when you raise `chunk_overlap`
  to half of `chunk_size`?

## Workshop 3 — Embeddings and Vector Database Indexing

- Read: `src/rag/vectorstore/chroma_store.py`, the ChromaDB section of
  `data/sample_docs/sample_2.txt`.
- Do: implement `ChromaStore.add` and `ChromaStore.query`. Run the full
  `scripts/ingest.py` (without `--dry-run`).
- Reflect: open `.chroma/`. What's in there? What do you think is recoverable
  if you delete that directory?

## Workshop 4 — Retrieval and Similarity Search

- Read: `src/rag/retrieval/retriever.py`.
- Do: implement `Retriever.search`. Try queries with and without metadata
  filters.
- Stretch: add a `filters={"source": ...}` argument and call it from a
  notebook.
- Reflect: why does this thin layer exist when it just calls the store?

## Workshop 5 — LLM Integration and Prompt Design

- Read: `src/rag/generation/prompts.py`, `pipeline.py`,
  `data/sample_docs/sample_3.md`.
- Do: implement `RAGPipeline.answer`. Run
  `uv run python scripts/query.py "What is RAG?"`.
- Reflect: try removing the "if you don't know, say so" line from the prompt.
  Ask a question your corpus can't answer. What changes?

## Workshop 6 — FastAPI Endpoints and API Design

- Read: `src/rag/api/main.py`, `routes.py`, `schemas.py`.
- Do: implement the FastAPI lifespan and the `POST /ask` route. Run
  `uv run uvicorn rag.api.main:app --reload` and test via `/docs`.
- Stretch: add request logging. Add a `top_k` query parameter to `GET /ask`.

## Workshop 7 — Testing, Refactoring, and Cleanup

- Read: `tests/conftest.py`. Note that no test hits a real LLM.
- Do: un-skip and complete tests in your two weakest modules.
- Stretch: implement `OpenAIProvider` from the stub. Change
  `LLM_PROVIDER=openai` in `.env`. Verify nothing else changes.

## Workshop 8 — Final Demo

Bring a 5-minute walkthrough. Show one query that works well, and one that
fails interestingly. Failure modes are the best part of demoing AI systems.
