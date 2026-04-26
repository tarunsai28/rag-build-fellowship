# `data/` — corpus directory

Anything dropped in `sample_docs/` is fair game for ingestion via:

```bash
uv run python scripts/ingest.py
```

Supported file types: `.pdf`, `.txt`, `.md`, `.markdown`, `.rst`.

The starter ships a small placeholder set so the pipeline has something to
chew on out of the box. In Workshop 7 we swap the placeholder corpus for a
real one (the choice is deferred — see `ARCHITECTURE_SPEC.md` §10).

## Adding your own documents

```bash
cp ~/papers/my-paper.pdf data/sample_docs/
uv run python scripts/ingest.py --clear     # re-index from scratch
uv run python scripts/query.py "what does the paper say about X?"
```

> **Storage:** embeddings live in `.chroma/` at the repo root. That directory
> is git-ignored. Delete it any time to start fresh — the next ingest run
> rebuilds it.
