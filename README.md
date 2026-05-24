# OAF RAG Build Project

Build a working document Q&A system from scratch in 8 weeks. Loads documents,
chunks them, embeds them, indexes them in a vector database, retrieves
relevant context for a query, and uses an LLM to produce a grounded answer —
all behind a FastAPI surface.

This is a teaching codebase. It deliberately avoids LangChain / LlamaIndex
abstractions so you can see the actual mechanics of RAG.

```
data/  →  loader  →  chunker  →  embeddings  →  ChromaDB
                                                   │
                                                   ▼
   question  →  embed  →  similarity search  →  prompt  →  LLM  →  answer
```

## Quickstart

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh         # macOS / Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. Install dependencies
uv sync

# 3. Configure
cp .env.example .env
# Open .env, paste GEMINI_API_KEY (free at https://aistudio.google.com/apikey)

# 4. Verify
uv run python scripts/verify_setup.py

# 5. Ingest the sample corpus
uv run python scripts/ingest.py

# 6. Ask a question
uv run python scripts/query.py "What is RAG?"

# 7. Or run the API
uv run uvicorn rag.api.main:app --reload
# Open http://127.0.0.1:8000/docs
```

## Layout

```
src/rag/        Source code (config, providers, ingestion, vectorstore,
                retrieval, generation, api, utils)
data/           Sample corpus (replaceable in Workshop 7)
docs/           Per-topic deep-dives
tests/          Pytest suite — uses mock providers, never hits real APIs
scripts/        verify_setup.py, ingest.py, query.py
```

## Branches

This repository ships several branches, each at a different point in the
fellowship's progression:

| Branch | Purpose |
|--------|---------|
| `main` | Full reference solution. All modules, all providers, all tests pass. |
| `student-starter` | What students clone. Stubs throughout, complete only where the spec says ("fully implemented in starter"). |
| `week-1-end` … `week-8-end` | Snapshots — what the codebase should look like at the end of each workshop. |

See [`MIGRATION.md`](MIGRATION.md) for how the branches relate and how to
move between them.

## Configuration

Everything is `.env`-driven. The ones you'll touch most:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `gemini` (default) / `openai` / `anthropic` / `ollama` |
| `EMBEDDINGS_PROVIDER` | `gemini` (default) / `openai` / `ollama` / `local` |
| `LLM_MODEL` | e.g. `gemini-2.5-flash-lite`, `gpt-4o-mini`, `claude-3-5-haiku-latest` |
| `EMBEDDINGS_MODEL` | e.g. `gemini-embedding-2`, `text-embedding-3-small` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Only the active provider's key needs to be set. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Retrieval hyper-parameters. |

## Common commands

```bash
make sync       # uv sync
make verify     # run scripts/verify_setup.py
make ingest     # ingest data/sample_docs into the vector store
make query Q='Your question'
make api        # run the FastAPI server
make test       # pytest
make lint       # ruff check
make format     # ruff format
```

## Documentation

- [`SETUP.md`](SETUP.md) — detailed environment setup per platform
- [`ARCHITECTURE_SPEC.md`](ARCHITECTURE_SPEC.md) — single source of truth
  for the design
- [`docs/architecture.md`](docs/architecture.md) — visual summary
- [`docs/workshop-guide.md`](docs/workshop-guide.md) — week-by-week reading
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — known issues
- [`docs/extensions.md`](docs/extensions.md) — stretch exercises
- [`MIGRATION.md`](MIGRATION.md) — how the branches relate

## License

MIT. See `LICENSE` if added.
