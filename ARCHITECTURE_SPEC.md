# Build a Real Retrieval-Augmented AI System from Scratch
## Architecture Specification & Codebase Scaffolding Plan

**Fellow:** Mayank Gowda (Mike)
**Project:** OAF Build Fellowship — 8-week instructor-led project
**Audience:** Beginner-to-intermediate students with some Python, basic API familiarity, and curiosity about AI
**Document purpose:** Single source of truth for codebase generation. This document is the input to Claude Code for scaffolding the repository.

---

## 1. Project context

Students build a working document Q&A API in Python over 8 weeks. The system loads documents, chunks and embeds them, stores embeddings in a vector database, retrieves relevant context for user queries, and uses an LLM to generate grounded answers. The final deliverable is exposed through a FastAPI backend.

The codebase is designed to teach the concepts of RAG without forcing students to wrestle with framework abstractions. Wherever possible, students see and write the actual mechanics — chunking, similarity search, prompt construction — rather than calling a black-box `RetrievalQA.from_chain_type()` and hoping for the best.

---

## 2. Locked architecture decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ | Modern type hints, Pydantic v2, broad compatibility |
| Package manager | `uv` | Fast, single tool, cross-platform, handles Python install + venv + deps |
| LLM (default) | Gemini 2.5 Flash-Lite | Free tier, no card, available in 200+ countries |
| LLM (fallback) | Ollama (Llama 3.1 8B) | Local, offline, for blocked regions or privacy needs |
| Embeddings (default) | Gemini `text-embedding-004` | Free, no separate setup |
| Embeddings (fallback) | `sentence-transformers/all-MiniLM-L6-v2` | Local, CPU-friendly, free |
| Vector DB | ChromaDB | File-backed, zero ops, beginner-friendly |
| API framework | FastAPI | Modern, type-safe, auto-generated docs |
| Provider abstraction | Custom Protocol-based | Pedagogical clarity over framework convenience |
| Repo structure | Single repo, multiple branches | Easier to maintain, simpler mental model |
| Cross-platform | Mac/Linux/Windows (PowerShell) | Single setup workflow with platform-specific notes |

**Explicitly rejected:**
- LangChain / LlamaIndex (too much hidden magic for a teaching codebase)
- Pinecone / Weaviate (cloud-only, requires accounts)
- OpenAI / Anthropic as default (paid, friction for students)
- Two separate repos for reference and starter (maintenance overhead)

---

## 3. Repository layout

```
oaf-rag-build-project/
├── README.md                      # Overview, quickstart, links to deeper docs
├── SETUP.md                       # Detailed environment setup per platform
├── CONTRIBUTING.md                # For students contributing fixes back
├── pyproject.toml                 # uv-managed project config
├── uv.lock                        # Pinned dependency versions
├── .env.example                   # Template environment file
├── .gitignore
├── .python-version                # Pins to 3.11
├── Makefile                       # Common commands (cross-platform via uv)
│
├── docs/
│   ├── architecture.md            # System diagram + data flow explanation
│   ├── workshop-guide.md          # Per-workshop notes for students
│   ├── troubleshooting.md         # Known issues & fixes (Gemini 429, etc.)
│   ├── extensions.md              # Stretch exercises for advanced students
│   └── images/
│       └── rag-architecture.png   # Diagram referenced from architecture.md
│
├── data/
│   ├── sample_docs/               # Placeholder corpus (replaced Workshop 7)
│   │   ├── sample_1.pdf
│   │   ├── sample_2.txt
│   │   └── sample_3.md
│   └── README.md                  # How to add your own documents
│
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py              # Loads .env, central settings (Workshop 1)
│       │
│       ├── providers/             # Provider abstraction (visible from Day 1)
│       │   ├── __init__.py
│       │   ├── base.py            # LLMProvider, EmbeddingsProvider Protocols
│       │   ├── gemini.py          # GeminiLLM, GeminiEmbeddings (default impl)
│       │   ├── openai_provider.py # OpenAILLM, OpenAIEmbeddings (stub for W7)
│       │   ├── anthropic_provider.py  # AnthropicLLM (stub for W7)
│       │   ├── ollama.py          # OllamaLLM, OllamaEmbeddings (fallback)
│       │   ├── local.py           # SentenceTransformersEmbeddings (fallback)
│       │   └── factory.py         # Reads env, returns concrete impl
│       │
│       ├── ingestion/             # Workshop 2
│       │   ├── __init__.py
│       │   ├── models.py          # Document, Chunk dataclasses
│       │   ├── loader.py          # PDF / txt / md loading
│       │   └── chunker.py         # Text splitting strategies
│       │
│       ├── vectorstore/           # Workshop 3
│       │   ├── __init__.py
│       │   └── chroma_store.py    # ChromaDB wrapper (add, query, persist)
│       │
│       ├── retrieval/             # Workshop 4
│       │   ├── __init__.py
│       │   └── retriever.py       # Top-k retrieval, ranking, filters
│       │
│       ├── generation/            # Workshop 5
│       │   ├── __init__.py
│       │   ├── prompts.py         # Prompt templates
│       │   └── pipeline.py        # End-to-end RAG: query → retrieve → generate
│       │
│       ├── api/                   # Workshop 6
│       │   ├── __init__.py
│       │   ├── main.py            # FastAPI app, lifespan, CORS
│       │   ├── routes.py          # Endpoint handlers
│       │   └── schemas.py         # Pydantic request/response models
│       │
│       └── utils/
│           ├── __init__.py
│           ├── retry.py           # Exponential backoff (429 handling)
│           └── logging.py         # Structured logging setup
│
├── tests/                         # Workshop 7 (mostly)
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures (mock providers, sample data)
│   ├── test_providers.py
│   ├── test_ingestion.py
│   ├── test_vectorstore.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
│
├── scripts/
│   ├── ingest.py                  # CLI: ingest documents into vector store
│   ├── query.py                   # CLI: query the system
│   └── verify_setup.py            # Workshop 1: confirms env is working
│
└── notebooks/                     # Optional exploration notebooks
    ├── 01_explore_chunking.ipynb
    ├── 02_explore_embeddings.ipynb
    └── 03_explore_retrieval.ipynb
```

---

## 4. Module-by-module specification

### 4.1 `src/rag/config.py`
Central configuration loaded from `.env`. Implemented in Workshop 1, used everywhere.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Provider selection
    llm_provider: str = "gemini"          # gemini|openai|anthropic|ollama
    embeddings_provider: str = "gemini"   # gemini|openai|ollama|local

    # Model selection (provider-specific)
    llm_model: str = "gemini-2.5-flash-lite"
    embeddings_model: str = "text-embedding-004"

    # API keys (only the active provider's key needs to be set)
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Paths
    data_dir: Path = Path("data/sample_docs")
    chroma_persist_dir: Path = Path(".chroma")

    # Retrieval defaults
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # Ollama (when used)
    ollama_base_url: str = "http://localhost:11434"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 4.2 `src/rag/providers/base.py`
The two Protocols students will implement against.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can take a prompt and return a string answer."""

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...

@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Anything that can turn text into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...
```

### 4.3 `src/rag/providers/gemini.py` (DEFAULT — fully implemented in starter)
Uses the official `google-genai` SDK. Wrapped with retry/backoff. Students don't write this — they use it as the working baseline.

### 4.4 `src/rag/providers/openai_provider.py` and `anthropic_provider.py`
**In starter branch:** stub files with class signatures and `raise NotImplementedError("Workshop 7 stretch exercise")`.
**In main branch:** fully implemented.
Workshop 7 stretch task: implement these and verify the pipeline works by changing `LLM_PROVIDER` in `.env`.

### 4.5 `src/rag/providers/ollama.py`
Wraps the local Ollama HTTP API. Used by students in regions blocked from Gemini, or those who want to run fully offline.

### 4.6 `src/rag/providers/local.py`
`SentenceTransformersEmbeddings` — wraps `sentence-transformers` library. CPU-friendly. ~80MB model downloads on first use.

### 4.7 `src/rag/providers/factory.py`
Reads `settings.llm_provider` and `settings.embeddings_provider`, returns the right implementation. One function per protocol. Students never modify this; they just change env vars.

```python
def get_llm() -> LLMProvider:
    match settings.llm_provider:
        case "gemini":   return GeminiLLM(...)
        case "openai":   return OpenAILLM(...)
        case "anthropic":return AnthropicLLM(...)
        case "ollama":   return OllamaLLM(...)
        case _: raise ValueError(...)
```

### 4.8 `src/rag/ingestion/models.py`
Two simple dataclasses students see early and use throughout.

```python
@dataclass
class Document:
    text: str
    source: str          # e.g., file path
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.9 `src/rag/ingestion/loader.py` (Workshop 2)
`DocumentLoader.load(path: Path) -> list[Document]`
- Detects file type by extension
- Supports `.pdf` (via `pypdf`), `.txt`, `.md`
- Returns one `Document` per file with text and source metadata
- Handles common errors (file not found, unreadable PDF) gracefully

### 4.10 `src/rag/ingestion/chunker.py` (Workshop 2)
`Chunker.chunk(documents: list[Document], size: int, overlap: int) -> list[Chunk]`
- Fixed-size chunking with overlap (simplest strategy that works)
- Optional improvement: respect sentence boundaries
- Each chunk carries source + position metadata for citation later

### 4.11 `src/rag/vectorstore/chroma_store.py` (Workshop 3)
Wraps ChromaDB. Three methods:
- `add(chunks: list[Chunk]) -> None` — embeds chunks and stores them
- `query(query_text: str, k: int) -> list[Chunk]` — semantic search
- `clear() -> None` — wipes the collection (useful in dev)

Persists to `.chroma/` directory by default. Embeddings provider is injected (uses whatever the factory returned).

### 4.12 `src/rag/retrieval/retriever.py` (Workshop 4)
Thin layer over the vector store. Students might think "why does this exist if it just calls `chroma_store.query()`?" — the answer is that retrieval grows in real systems (filtering, reranking, hybrid search) and the abstraction gives them a place to extend.

```python
class Retriever:
    def search(self, query: str, k: int = 4) -> list[Chunk]: ...
```

Workshop 4 stretch: add metadata filtering (`source: "manual.pdf"`).

### 4.13 `src/rag/generation/prompts.py` (Workshop 5)
Just one prompt template at the start. Students experiment with it.

```python
RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY
the context provided. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
```

### 4.14 `src/rag/generation/pipeline.py` (Workshop 5)
The end-to-end glue.

```python
class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMProvider): ...

    def answer(self, question: str) -> Answer:
        chunks = self.retriever.search(question)
        context = format_context(chunks)
        prompt = RAG_PROMPT.format(context=context, question=question)
        text = self.llm.generate(prompt)
        return Answer(text=text, sources=chunks)
```

### 4.15 `src/rag/api/` (Workshop 6)
- `main.py` — FastAPI app instance, CORS, lifespan that initializes the pipeline once at startup
- `routes.py` — `POST /ask` (returns answer + sources), `GET /health`, `POST /ingest` (re-runs ingestion)
- `schemas.py` — `AskRequest`, `AskResponse`, `Source`

### 4.16 `src/rag/utils/retry.py`
Decorator for exponential backoff on 429 / transient errors. Used inside the Gemini and OpenAI providers. **Critical for free-tier reliability.**

```python
@retry_with_backoff(retries=5, base_delay=1.0)
def generate(self, prompt: str, ...) -> str:
    ...
```

### 4.17 `tests/` (Workshop 7)
- Fixtures: a `MockLLMProvider` and `MockEmbeddingsProvider` so tests don't hit real APIs
- One small test file per module — focus on the highest-value tests, not coverage for its own sake
- `pytest` + `pytest-asyncio` for async API tests

### 4.18 `scripts/`
- `verify_setup.py` — Workshop 1 deliverable. Checks Python version, env vars, makes one Gemini call, prints "✓ ready". This is the success criterion for Week 1.
- `ingest.py` — CLI to ingest a directory of docs into the vector store
- `query.py` — CLI to ask one-off questions without running the API

---

## 5. Provider abstraction in detail

This is the architectural centerpiece. Walking through it carefully because students will see the concept before they understand its purpose.

**The "why" we'll teach:** different providers have different APIs. OpenAI uses `client.chat.completions.create(messages=[...])`. Anthropic uses `client.messages.create(messages=[...])`. Gemini uses `model.generate_content(prompt)`. Without an abstraction, swapping providers means changing every call site. With the abstraction, swapping is one env var change.

**The "how" we'll teach:** the Protocol class. We *don't* use abstract base classes (ABC) — Protocols are duck-typed and feel more Pythonic. Students learn that "anything with a `generate(prompt) -> str` method is an `LLMProvider`."

**The pedagogical sequence:**
- **Workshop 1:** Students see `providers/base.py` and `providers/gemini.py`. They run `from rag.providers.factory import get_llm; llm = get_llm(); print(llm.generate("Hello"))`. They don't write provider code yet — they use it.
- **Workshops 2–6:** Students build features that *consume* the provider abstraction. Their chunker code calls `embeddings.embed(...)`. Their pipeline calls `llm.generate(...)`. They never see provider-specific code.
- **Workshop 7 stretch:** Students implement `OpenAIProvider` from the stub. They follow the existing `GeminiLLM` as a model. They verify by changing `LLM_PROVIDER=openai` in `.env` and running the same pipeline. Aha moment: the rest of the code didn't change.

This is a deliberate teaching arc: introduce the seam → use it implicitly → understand it explicitly.

---

## 6. Week-by-week scaffolding plan

For each workshop, this section specifies:
- **Already in starter** — what students inherit (working code, structure, stubs)
- **TODO this week** — what students fill in
- **End state** — what the codebase should look like at the end (= contents of `week-N-end` branch)

### Workshop 1 — Project Kickoff, Architecture, Environment Setup

**Already in starter:**
- Full repo structure (all dirs, all `__init__.py` files)
- `pyproject.toml`, `.python-version`, `.env.example`, `Makefile`
- `config.py` (fully implemented)
- `providers/base.py` (Protocols defined)
- `providers/gemini.py` (fully implemented — students need a working baseline)
- `providers/factory.py` (fully implemented)
- `providers/openai_provider.py` (stub with `NotImplementedError`)
- `providers/anthropic_provider.py` (stub with `NotImplementedError`)
- `providers/ollama.py` (fully implemented for the fallback path)
- `providers/local.py` (fully implemented for the fallback path)
- `utils/retry.py` (fully implemented)
- `utils/logging.py` (fully implemented)
- `scripts/verify_setup.py` (fully implemented)
- `data/sample_docs/` with 3 placeholder documents

**TODO this week:**
- Install `uv`
- Run `uv sync`
- Get a Gemini API key (in-workshop walkthrough)
- Copy `.env.example` to `.env`, paste key
- Run `python scripts/verify_setup.py` and see ✓
- Make one customization: change `LLM_MODEL` in `.env` and verify the change took effect

**End state (`week-1-end`):**
- `verify_setup.py` exits 0
- Student understands the file tree at a high level
- Student has run their first Gemini call

### Workshop 2 — Document Ingestion and Chunking

**Already in starter:** all of Week 1's content, plus:
- `ingestion/models.py` (fully implemented — dataclasses)
- `ingestion/loader.py` with stub:
  ```python
  class DocumentLoader:
      def load(self, path: Path) -> list[Document]:
          # TODO: detect file type from extension
          # TODO: dispatch to _load_pdf / _load_text / _load_markdown
          # TODO: return list of Document
          raise NotImplementedError
  ```
- `ingestion/chunker.py` with stub for `Chunker.chunk(...)`

**TODO this week:**
- Implement `DocumentLoader.load` for `.pdf`, `.txt`, `.md`
- Implement `Chunker.chunk` with fixed-size + overlap strategy
- Run `python scripts/ingest.py --dry-run` to see chunks printed without storing them yet (script is fully implemented and gracefully handles the "no vector store" case)

**End state (`week-2-end`):** ingest pipeline produces clean chunks. No storage yet.

### Workshop 3 — Embeddings and Vector Database Indexing

**Already in starter:** all of Week 2's content, plus:
- `vectorstore/chroma_store.py` with stub for `ChromaStore.add`, `query`, `clear`

**TODO this week:**
- Implement `ChromaStore.__init__` (collection setup, persistence path)
- Implement `ChromaStore.add(chunks)` — embed via injected `EmbeddingsProvider`, store in Chroma
- Implement `ChromaStore.query(text, k)` — embed query, retrieve top-k
- Run `python scripts/ingest.py` and verify the `.chroma/` directory is populated

**End state (`week-3-end`):** sample docs are embedded and stored. A direct call to `chroma_store.query("...")` returns relevant chunks.

### Workshop 4 — Retrieval and Similarity Search

**Already in starter:** all of Week 3's content, plus:
- `retrieval/retriever.py` with stub

**TODO this week:**
- Implement `Retriever.search(query, k)` — for now, just delegates to `chroma_store.query`, but introduce the abstraction
- Test with example queries from a notebook
- Discuss why this thin layer exists (extensibility for filtering, reranking)

**Stretch:** add metadata filtering — `Retriever.search(query, k, filters={"source": "manual.pdf"})`.

**End state (`week-4-end`):** clean retrieval interface exists, returns ranked chunks.

### Workshop 5 — LLM Integration and Prompt Design

**Already in starter:** all of Week 4's content, plus:
- `generation/prompts.py` with the `RAG_PROMPT` template (filled in)
- `generation/pipeline.py` with stub for `RAGPipeline.answer`

**TODO this week:**
- Implement `RAGPipeline.__init__(retriever, llm)`
- Implement `RAGPipeline.answer(question) -> Answer`:
  - call retriever, format context, build prompt, call llm
- Implement `format_context(chunks) -> str` helper
- Test end-to-end: `python scripts/query.py "What is X?"`
- Discuss prompt iteration — what happens when context is empty? Conflicting? Long?

**End state (`week-5-end`):** working CLI Q&A. The full RAG loop runs.

### Workshop 6 — FastAPI Endpoints and API Design

**Already in starter:** all of Week 5's content, plus:
- `api/schemas.py` (fully implemented — Pydantic models)
- `api/main.py` with stub
- `api/routes.py` with stub

**TODO this week:**
- Implement FastAPI lifespan that initializes pipeline once at startup
- Implement `POST /ask` endpoint
- Implement `GET /health` endpoint
- Run `uvicorn rag.api.main:app --reload` and test via Swagger UI

**Stretch:** streaming responses, request logging, basic auth header.

**End state (`week-6-end`):** running API. `curl localhost:8000/ask` returns answers.

### Workshop 7 — Testing, Refactoring, and Cleanup

**Already in starter:** all of Week 6's content, plus:
- `tests/conftest.py` (fully implemented — fixtures, mocks)
- `tests/test_*.py` files with several stub tests marked `@pytest.mark.skip`

**TODO this week:**
- Un-skip and complete 1–2 tests per module
- Refactor any rough spots in their own code
- Update README with their additions

**Stretch (this is where the provider abstraction lesson lands):**
- Implement `OpenAIProvider` from the stub
- Get an OpenAI key (or use the OAF-allocated credits if available)
- Change `LLM_PROVIDER=openai` in `.env`
- Verify the same pipeline works without changing any other code
- Reflect on why this was easy

**End state (`week-7-end`):** tests pass, code is presentation-ready, repo README is complete.

### Workshop 8 — Final Demo

No new code. Students present their projects.

---

## 7. Branch strategy

- **`main`** — full reference solution. All modules complete. All providers complete. Maintained by the Fellow (you). Source of truth.
- **`student-starter`** — what students fork/clone. Stubs everywhere. Workshop 1 prep already done. This is what students initially clone.
- **`week-1-end`, `week-2-end`, ..., `week-8-end`** — checkpoint snapshots. A student who falls behind in Week 3 can `git checkout week-3-end` and rejoin in Week 4 without being lost.

**How they're generated:**
1. `main` is generated first (full reference)
2. `student-starter` is derived by replacing function bodies with stubs (mechanical script + manual review)
3. `week-N-end` branches are derived by selectively filling in specific modules from `main` into `student-starter`

This is a clear, repeatable process. Claude Code can execute it given this spec.

---

## 8. Setup workflow (single workflow, all platforms)

The setup is intentionally identical on macOS, Linux, and Windows because `uv` handles the platform differences. The only divergence is shell syntax for environment activation (which `uv run` makes unnecessary anyway).

```bash
# 1. Install uv (one-line installer, all platforms)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone repo, switch to student starter
git clone https://github.com/<oaf>/<repo>.git
cd <repo>
git checkout student-starter

# 3. Install dependencies (uv reads pyproject.toml + .python-version)
uv sync

# 4. Set up environment
cp .env.example .env
# Edit .env: paste GEMINI_API_KEY

# 5. Verify setup
uv run python scripts/verify_setup.py

# 6. Done. Throughout the project, prefix commands with 'uv run':
uv run python scripts/ingest.py
uv run python scripts/query.py "What is X?"
uv run pytest
uv run uvicorn rag.api.main:app --reload
```

**Windows-specific notes:**
- PowerShell required (not cmd.exe)
- May need to `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once
- Long path support may need to be enabled (one registry tweak — documented in SETUP.md)

**Mac-specific notes:**
- Apple Silicon: PyTorch (used by sentence-transformers) installs the MPS-accelerated build automatically via uv

**Linux-specific notes:**
- None expected. Reference platform.

---

## 9. Testing strategy

Three layers, in order of importance:

1. **Smoke tests (highest priority).** Tests that catch "the pipeline doesn't run at all." `verify_setup.py` is the user-facing version. There's also a `test_smoke.py` that runs the whole RAG flow with mock providers and asserts it returns *something*.

2. **Unit tests on critical pure functions.** Chunker (deterministic input/output), prompt formatting, retriever ranking. Ten tests, fast, no external dependencies.

3. **API integration tests.** Spin up FastAPI in test mode, send `POST /ask`, assert structure of response.

**What we explicitly skip:**
- Hitting real Gemini in tests (rate-limit risk, flakiness, requires keys in CI)
- LLM output quality testing (this is the user's job at demo time, not a unit test)
- Coverage targets (this is a teaching repo, not a production system)

---

## 10. Sample data corpus

**Phase 1 (Workshops 1–6):** placeholder corpus. Three documents totaling under 50 pages. Need to be:
- Public domain or permissively licensed (no copyright issues)
- Topically coherent (so retrieval results feel sensible)
- Mix of formats (one PDF, one txt, one markdown)

**Suggested placeholder set** (final selection deferred to Week 7):
- A short Wikipedia article exported as markdown
- An open-source software README (your own GitHub project, or something like the FastAPI README)
- One short public-domain PDF (a government report excerpt, an arXiv abstract collection)

**Phase 2 (Workshop 7):** swap in the "real" corpus. Options:
- Synthetic company handbook (you write it, ~10 pages, fictional company)
- Build Fellowship FAQ document (real, useful, scoped)
- Topic of student interest, voted on Week 5

The corpus does not affect the architecture — only the demo experience. Defer the choice without blocking.

---

## 11. The Claude Code generation prompt

This is the prompt to hand to Claude Code at the start of Phase B (codebase generation). Save it as `claude_code_prompt.md` in your local workspace.

```
You are scaffolding a Python codebase for a teaching project: an 8-week
Retrieval-Augmented Generation system for the OAF Build Fellowship.

The full architecture specification is in ARCHITECTURE_SPEC.md (attached).
Read it before generating any code.

Your task:
1. Generate the complete `main` branch (full reference solution) according
   to the spec. All modules implemented. All providers implemented. Tests
   passing. README complete.
2. From `main`, derive the `student-starter` branch by replacing function
   bodies with `raise NotImplementedError("Workshop N: <description>")`
   and adding a `# TODO:` comment block above each describing what the
   student needs to do. Modules listed as "fully implemented in starter"
   in section 6 of the spec are NOT stubbed — they ship complete.
3. From `student-starter`, derive the `week-1-end`, `week-2-end`, ...,
   `week-8-end` branches by filling in the modules specified in section 6
   for each week.

Constraints:
- Python 3.11+, uv for package management
- Default LLM: Gemini 2.5 Flash-Lite via google-genai SDK
- Default embeddings: Gemini text-embedding-004
- Vector DB: ChromaDB (file-backed, persisted to .chroma/)
- API framework: FastAPI
- All API calls wrapped with retry/backoff (see utils/retry.py spec)
- Provider abstraction via typing.Protocol (NOT abc.ABC)
- Cross-platform (macOS / Linux / Windows PowerShell)
- No LangChain, no LlamaIndex

Quality bar:
- Every public function has a docstring
- Type hints on every function signature
- `ruff check` and `ruff format` pass cleanly
- `uv run pytest` passes on the `main` branch
- `uv run python scripts/verify_setup.py` succeeds with a valid Gemini key
- Each workshop's "TODO this week" tasks are achievable in 30-45 minutes
  of guided coding by an intermediate student

Deliverables:
- Generated repository in /current working directory
- All branches created and pushed (or ready to push)
- ARCHITECTURE_SPEC.md and SETUP.md committed
- A short MIGRATION.md describing how the branches relate

Start by listing the file tree you intend to create, then generate
files in dependency order: config → providers → ingestion → vectorstore
→ retrieval → generation → api → tests → scripts → docs.
```

---

## 12. Open items & things to watch

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| OAF API credit allocation | Pending Slack ask | Mike | Gemini default unaffected |
| Final corpus selection | Defer to Workshop 7 | Mike + students | Architecture-independent |
| Screening questions for student application | Step 2 doc TODO | Mike | Not blocking codebase |
| Peer review pairing | Step 3 optional task | Mike | Schedule after spec freeze |
| Gemini free tier policy changes | Monitor monthly | Mike | Tier shrunk twice in past 6 months — env-var fallback to Ollama mitigates |
| Slide deck generation | After codebase | Mike + Claude | Reference codebase = source of truth for screenshots |

---

## 13. Document control

- **v1.0** — Initial spec (this document). Architecture frozen. Ready for codebase generation.
- Update this document if architecture decisions change. The codebase derives from this spec, so a stale spec creates drift.
- All future modifications go through PR review.

---

**End of specification.**
