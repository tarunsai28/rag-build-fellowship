"""Workshop 1 deliverable: ``uv run python scripts/verify_setup.py``.

Runs in three stages:

1. Check the Python interpreter version.
2. Check that the configured provider's API key is present.
3. Make a one-token round-trip call to the LLM and the embeddings model.

Exits with status 0 on success, 1 on any failure. On failure, prints a
specific, actionable error message — no stack traces.
"""

from __future__ import annotations

import sys


def _emit(ok: bool, label: str, detail: str = "") -> None:
    icon = "[OK]  " if ok else "[FAIL]"
    suffix = f" — {detail}" if detail else ""
    print(f"{icon} {label}{suffix}")


def check_python() -> bool:
    """Confirm Python 3.11+."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    _emit(ok, "Python version", f"{major}.{minor}.{sys.version_info.micro}")
    if not ok:
        print("    Need Python 3.11 or newer. Run: uv sync")
    return ok


def check_settings() -> bool:
    """Confirm settings load and the active provider's key is set."""
    try:
        from rag.config import settings
    except Exception as exc:
        _emit(False, "Settings", f"failed to import: {exc}")
        return False

    provider = settings.llm_provider.lower()
    key_attr = {
        "gemini": "gemini_api_key",
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "ollama": None,  # No key needed.
    }.get(provider)

    _emit(True, "Settings loaded", f"llm_provider={provider}, model={settings.llm_model}")
    if key_attr is None:
        return True
    if not getattr(settings, key_attr, None):
        _emit(False, "API key present", f"{key_attr.upper()} is empty")
        print(f"    Add {key_attr.upper()}=... to your .env file.")
        return False
    _emit(True, "API key present", f"{key_attr.upper()} configured")
    return True


def check_llm() -> bool:
    """Round-trip a tiny prompt through the configured LLM."""
    try:
        from rag.providers.factory import get_llm

        llm = get_llm()
        reply = llm.generate("Reply with the single word: ready", temperature=0.0, max_tokens=8)
    except Exception as exc:
        _emit(False, "LLM round-trip", str(exc).splitlines()[0])
        return False
    _emit(True, "LLM round-trip", f"reply: {reply[:60]!r}")
    return True


def check_embeddings() -> bool:
    """Round-trip a single string through the configured embeddings provider."""
    try:
        from rag.providers.factory import get_embeddings

        embeddings = get_embeddings()
        vectors = embeddings.embed(["hello world"])
    except Exception as exc:
        _emit(False, "Embeddings round-trip", str(exc).splitlines()[0])
        return False
    if not vectors or not vectors[0]:
        _emit(False, "Embeddings round-trip", "empty vector")
        return False
    _emit(True, "Embeddings round-trip", f"dim={len(vectors[0])}")
    return True


def main() -> int:
    """Run all setup checks. Returns 0 on success, 1 on failure."""
    print("Verifying OAF RAG setup...\n")
    steps = [check_python, check_settings, check_llm, check_embeddings]
    results = [step() for step in steps]
    print()
    if all(results):
        print("Setup is ready. You can move on to Workshop 2.")
        return 0
    print("Setup is NOT ready. See FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
