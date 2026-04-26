# Branch map and migration guide

This repository ships **ten** branches. They share a common scaffold and
diverge by what's filled in. This document explains the map.

## The map

```
main                  Full reference solution. Source of truth.
                          │
                          │  derive: stub function bodies marked
                          │  "TODO Workshop N" per ARCHITECTURE_SPEC.md §6.
                          ▼
student-starter       What students initially clone.
                      Workshop 1 prep done. Stubs everywhere else.
                          │
                          │  Workshop 1: setup only — no new code.
                          ▼
week-1-end            Identical to student-starter.
                          │
                          │  Fill in: ingestion/loader.py + ingestion/chunker.py
                          ▼
week-2-end            Loader + chunker complete.
                          │
                          │  Fill in: vectorstore/chroma_store.py
                          ▼
week-3-end            Vector store complete. Sample docs are now indexable.
                          │
                          │  Fill in: retrieval/retriever.py
                          ▼
week-4-end            Retriever complete.
                          │
                          │  Fill in: generation/pipeline.py
                          ▼
week-5-end            Full RAG loop runs from CLI.
                          │
                          │  Fill in: api/main.py + api/routes.py
                          ▼
week-6-end            FastAPI surface is live.
                          │
                          │  Tests un-skipped + completed.
                          │  (Stretch:) openai_provider.py + anthropic_provider.py
                          ▼
week-7-end            Tests pass. Optional providers implemented.
                          │
                          │  Workshop 8: demo. No code change.
                          ▼
week-8-end            Identical to week-7-end.
```

## What's the same in every branch

- Top-level config: `pyproject.toml`, `.python-version`, `.env.example`,
  `.gitignore`, `Makefile`, `README.md`, `SETUP.md`,
  `ARCHITECTURE_SPEC.md`, `MIGRATION.md`, `CONTRIBUTING.md`, `docs/`.
- `data/sample_docs/` corpus.
- `src/rag/__init__.py`, `config.py`.
- `src/rag/utils/retry.py`, `logging.py`.
- `src/rag/providers/base.py` (Protocols), `gemini.py`, `ollama.py`,
  `local.py`, `factory.py`.
- `src/rag/ingestion/models.py` (the Document and Chunk dataclasses).
- `src/rag/generation/prompts.py`.
- `src/rag/api/schemas.py`.
- `tests/conftest.py` (mock providers and fixtures).
- `scripts/verify_setup.py`, `scripts/ingest.py`, `scripts/query.py`.

## What changes branch to branch

Function bodies in module files. Each file either contains the working
implementation (as on `main`), or contains:

```python
def foo(self, ...) -> ...:
    """Public docstring kept identical to main."""
    # TODO Workshop N: <one-paragraph description of what to do>
    raise NotImplementedError("Workshop N: implement <name>")
```

The signatures, docstrings, and imports remain identical so that consuming
code (and tests) compile against either form.

## How to switch between branches

```bash
# Save your work first.
git stash

# Jump to a checkpoint.
git checkout week-3-end

# Re-install in case dependencies changed (rare, but cheap).
uv sync

# Continue from there.
git checkout -b my-week-4 week-3-end
```

## How instructors regenerate the branches

```bash
# Starting from main (full reference):
git checkout main

# 1. Create student-starter by stubbing per spec §6.
git checkout -b student-starter
# … run the stubbing script / manual edits …
git commit -am "Stub workshop modules per ARCHITECTURE_SPEC.md §6"

# 2. Create week-N-end branches.
git checkout -b week-1-end student-starter            # identical
for week in 2 3 4 5 6 7; do
    git checkout -b "week-${week}-end" "week-$((week-1))-end"
    git checkout main -- <files implemented this week>
    git commit -m "Workshop ${week}: fill in <modules>"
done
git checkout -b week-8-end week-7-end                 # identical
```

The exact "files implemented this week" list per workshop comes from
`ARCHITECTURE_SPEC.md` §6 — that document is the source of truth, this
file is the navigational map.
