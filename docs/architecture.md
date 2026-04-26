# Architecture

> The single source of truth is `ARCHITECTURE_SPEC.md` at the repo root.
> This file is the human-friendly summary that links back into it.

## System diagram

```
                ┌─────────────────────────────────────────────────────┐
                │                     FastAPI app                     │
                │   POST /ask     POST /ingest     GET /health        │
                └─────────────┬───────────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │       RAGPipeline           │
                │  (retrieve → format → gen)  │
                └─────┬───────────────┬───────┘
                      │               │
                      ▼               ▼
            ┌──────────────┐    ┌─────────────┐
            │  Retriever   │    │ LLMProvider │
            └──────┬───────┘    └──────┬──────┘
                   │                   │
                   ▼                   ▼
            ┌──────────────┐    ┌─────────────────────────────┐
            │  ChromaStore │    │  Gemini / OpenAI /          │
            │  (.chroma/)  │    │  Anthropic / Ollama         │
            └──────┬───────┘    └─────────────────────────────┘
                   ▲
                   │
            ┌──────┴────────────────────┐
            │  Chunker  ◄──  Loader     │     ingestion pipeline
            └────────────────┬──────────┘
                             ▼
                       data/sample_docs/
```

## Data flow

1. **Ingest** (run once, then on demand). `DocumentLoader` walks
   `data/sample_docs/`, returns `Document`s. `Chunker` splits them into
   overlapping `Chunk`s. `ChromaStore` embeds each chunk via the configured
   `EmbeddingsProvider` and persists vectors to `.chroma/`.
2. **Ask** (per request). `Retriever.search(question)` embeds the query and
   pulls the top-k most similar chunks. `format_context(chunks)` renders
   them into a prompt. `LLMProvider.generate(prompt)` returns the answer.

## The provider seam

Every external model call goes through `LLMProvider.generate(...)` or
`EmbeddingsProvider.embed(...)`. Concrete classes live in
`src/rag/providers/` and are selected by `factory.get_llm()` /
`factory.get_embeddings()` based on `.env` settings. Swapping providers is
one env-var change — see Workshop 7 for the exercise that proves it.
