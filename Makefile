# Cross-platform Makefile for the OAF RAG build project.
# All targets shell out to `uv run`, so they work identically on
# macOS, Linux, and Windows (PowerShell with GNU Make installed).

.PHONY: help sync verify ingest query api test lint format clean

help:
	@echo "Common commands:"
	@echo "  make sync      Install dependencies (uv sync)"
	@echo "  make verify    Run scripts/verify_setup.py"
	@echo "  make ingest    Ingest data/sample_docs into the vector store"
	@echo "  make query Q='What is X?'   Ask one question via CLI"
	@echo "  make api       Run the FastAPI server (uvicorn)"
	@echo "  make test      Run pytest"
	@echo "  make lint      Run ruff check"
	@echo "  make format    Run ruff format"
	@echo "  make clean     Remove caches and the local Chroma store"

sync:
	uv sync

verify:
	uv run python scripts/verify_setup.py

ingest:
	uv run python scripts/ingest.py

query:
	uv run python scripts/query.py "$(Q)"

api:
	uv run uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .chroma .pytest_cache .ruff_cache .mypy_cache **/__pycache__
