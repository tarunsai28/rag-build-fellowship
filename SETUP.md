# Setup

This document is the authoritative environment-setup guide for the project.
The same workflow works on macOS, Linux, and Windows (PowerShell) thanks to
`uv` handling all the platform differences.

## Prerequisites

- A terminal you're comfortable in (Terminal.app, GNOME Terminal, PowerShell).
- Git installed and on your PATH.
- A free Gemini API key — get one at <https://aistudio.google.com/apikey>.

You do **not** need to install Python yourself. `uv` handles that.

---

## 1. Install `uv`

`uv` is a single binary that handles Python install, virtual environments,
and dependency resolution. One command:

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If PowerShell complains about scripts being disabled, set the execution
policy once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Restart your shell, then verify:

```bash
uv --version
```

---

## 2. Clone and pick a branch

```bash
git clone <repo-url>
cd oaf-rag-build-project

# If you're a student, start from the starter branch:
git checkout student-starter

# Or pick any week's checkpoint:
git checkout week-3-end
```

---

## 3. Install dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `.python-version`, downloads Python 3.11 if
you don't have it, creates `.venv/`, and installs everything pinned in
`uv.lock`. First run takes a couple of minutes; subsequent runs are
near-instant.

> **Optional extras.** The `local` extra installs `sentence-transformers`
> for offline embeddings; this pulls in PyTorch (~2 GB on disk). Only
> install it if you actually plan to use local embeddings:
>
> ```bash
> uv sync --extra local
> ```

---

## 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in `GEMINI_API_KEY=...`. The other keys
(OpenAI, Anthropic) can stay empty unless you switch providers.

---

## 5. Verify

```bash
uv run python scripts/verify_setup.py
```

You should see four `[OK]` lines and a final "Setup is ready."

If anything fails, the script prints a specific message — see
[`docs/troubleshooting.md`](docs/troubleshooting.md).

---

## 6. (Optional) Set up Ollama for fully-local operation

Use this if Gemini is blocked in your region or you want full offline
operation.

```bash
# Install (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a chat model (~5 GB) and an embedding model (~300 MB)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Make sure ollama serve is running (it usually starts as a daemon).
# Then point this project at it:
echo "LLM_PROVIDER=ollama"           >> .env
echo "EMBEDDINGS_PROVIDER=ollama"    >> .env
echo "LLM_MODEL=llama3.1:8b"         >> .env
echo "EMBEDDINGS_MODEL=nomic-embed-text" >> .env

uv run python scripts/verify_setup.py
```

Windows: download the Ollama installer from <https://ollama.com/download>.

---

## Platform notes

### macOS

- Apple Silicon: PyTorch (used only by the optional `local` embeddings)
  installs the MPS-accelerated build automatically via `uv`. No action
  needed.

### Linux

- Reference platform. No special steps.

### Windows

- Use PowerShell, not `cmd.exe`.
- Long path support may need to be enabled — see
  [`docs/troubleshooting.md`](docs/troubleshooting.md).
- Ollama on Windows requires WSL2 for the GPU path. CPU-only works without it.

---

## Daily workflow

After setup, you'll prefix everything with `uv run`:

```bash
uv run python scripts/ingest.py
uv run python scripts/query.py "What is RAG?"
uv run pytest
uv run uvicorn rag.api.main:app --reload
```

Or use the `Makefile` shortcuts (`make ingest`, `make api`, `make test`,
etc.) if you have GNU Make installed.
