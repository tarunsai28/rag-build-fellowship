# Stretch exercises

Once the core pipeline runs, these are good places to push further. None of
them are required for the final demo. Each is sized to fit a single weekend.

## Retrieval

- **Hybrid search.** Combine BM25 (keyword) and dense (vector) scores. The
  `rank-bm25` package is small enough to add. Run both retrieval paths and
  rerank by a weighted sum.
- **Metadata filters.** Extend `Retriever.search` to accept
  `filters={"source": "manual.pdf"}` and pass through to Chroma's `where`.
- **Reranking.** Run a small cross-encoder (e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top-20 candidates and
  return the top-4.

## Chunking

- **Sentence-boundary chunking.** Don't split mid-sentence. Use a regex
  splitter or `nltk.tokenize.sent_tokenize` and pack sentences up to the
  size limit.
- **Semantic chunking.** Group consecutive sentences whose embeddings are
  close enough; cut a chunk boundary where similarity drops.
- **Markdown-aware chunking.** Respect heading hierarchy: don't split a
  section header from its body.

## Generation

- **Streaming.** Replace `LLMProvider.generate` with `generate_stream`
  yielding tokens, and stream them through FastAPI as
  `text/event-stream` (SSE).
- **Citations.** Modify the prompt to request `[1]`, `[2]` citations, and
  in `AskResponse` return the cited chunks first.
- **Multi-question dispatch.** If a question contains "and" / "also",
  split it, retrieve per sub-question, then synthesise.

## Operations

- **Caching.** Hash `(question, top_k, llm_model)` and cache answers in
  SQLite or Redis. Surface cache hit/miss in `/health`.
- **Eval harness.** Build a tiny gold set of (question, expected substring)
  pairs and a `scripts/eval.py` that scores the pipeline.
- **Observability.** Wrap each request with a unique trace id; log
  retrieved-chunk ids, latency per stage, and final answer length.
