"""Tests for prompt formatting and the RAGPipeline orchestration.

Skipped in the starter — un-skip and complete in Workshop 7.
"""

from __future__ import annotations

import pytest

from rag.generation.pipeline import RAGPipeline, format_context
from rag.generation.prompts import RAG_PROMPT
from rag.ingestion.models import Chunk


@pytest.mark.skip(reason="Workshop 7: format_context renders chunks.")
def test_format_context_renders_each_chunk() -> None:
    chunks = [
        Chunk(text="alpha", source="a.txt", chunk_index=0),
        Chunk(text="beta", source="b.txt", chunk_index=3),
    ]
    rendered = format_context(chunks)
    assert "[1]" in rendered and "alpha" in rendered
    assert "[2]" in rendered and "beta" in rendered
    assert "a.txt" in rendered and "b.txt" in rendered


@pytest.mark.skip(reason="Workshop 7: format_context handles empty input.")
def test_format_context_handles_empty() -> None:
    assert "no context" in format_context([]).lower()


@pytest.mark.skip(reason="Workshop 7: prompt template has the right placeholders.")
def test_rag_prompt_has_required_placeholders() -> None:
    formatted = RAG_PROMPT.format(context="CTX", question="Q")
    assert "CTX" in formatted
    assert "Q" in formatted


@pytest.mark.skip(reason="Workshop 7: pipeline injects retrieved context into the prompt.")
def test_pipeline_passes_context_to_llm(pipeline: RAGPipeline) -> None:
    pipeline.answer("about RAG")
    sent_prompt = pipeline.llm.calls[-1]  # type: ignore[attr-defined]
    assert "about RAG" in sent_prompt
    assert "Retrieval-Augmented" in sent_prompt or "RAG" in sent_prompt
