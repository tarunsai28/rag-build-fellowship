"""Tests for the provider Protocols and the factory dispatch.

Skipped in the starter — un-skip and complete in Workshop 7.
"""

from __future__ import annotations

import pytest

from rag.providers.base import EmbeddingsProvider, LLMProvider
from rag.providers.factory import get_embeddings, get_llm
from tests.conftest import MockEmbeddings, MockLLM
def test_mock_llm_satisfies_protocol() -> None:
    """A duck-typed mock should isinstance-match the runtime-checkable Protocol."""
    assert isinstance(MockLLM(), LLMProvider)
def test_mock_embeddings_satisfies_protocol() -> None:
    assert isinstance(MockEmbeddings(), EmbeddingsProvider)
def test_factory_rejects_unknown_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factories should raise ValueError on unknown provider names."""
    from rag.config import settings

    monkeypatch.setattr(settings, "llm_provider", "no-such-provider")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        get_llm()
def test_factory_rejects_unknown_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag.config import settings

    monkeypatch.setattr(settings, "embeddings_provider", "no-such-provider")
    with pytest.raises(ValueError, match="Unknown embeddings_provider"):
        get_embeddings()

