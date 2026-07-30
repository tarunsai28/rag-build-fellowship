"""Smoke test: the whole RAG pipeline runs end-to-end with mock providers.

Skipped in the starter — un-skip and complete in Workshop 7.
"""

from __future__ import annotations

import pytest

from rag.generation.pipeline import RAGPipeline
def test_pipeline_runs_end_to_end(pipeline: RAGPipeline) -> None:
    """Asking a question returns an answer string and at least one source chunk."""
    answer = pipeline.answer("What is RAG?")
    assert isinstance(answer.text, str)
    assert answer.text  # non-empty
    assert answer.sources, "expected at least one source chunk"
    assert answer.question == "What is RAG?"
def test_empty_question_returns_safely(pipeline: RAGPipeline) -> None:
    """Empty input does not raise."""
    answer = pipeline.answer("")
    assert "empty" in answer.text.lower()
    assert answer.sources == []

