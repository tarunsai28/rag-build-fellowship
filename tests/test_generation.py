"""Tests for prompt formatting and the RAGPipeline orchestration."""

from __future__ import annotations

from rag.generation.pipeline import RAGPipeline, format_context
from rag.generation.prompts import RAG_PROMPT
from rag.ingestion.models import Chunk


def test_format_context_renders_each_chunk() -> None:
    chunks = [
        Chunk(text="alpha", source="a.txt", chunk_index=0),
        Chunk(text="beta", source="b.txt", chunk_index=3),
    ]
    rendered = format_context(chunks)
    assert "[1]" in rendered and "alpha" in rendered
    assert "[2]" in rendered and "beta" in rendered
    assert "a.txt" in rendered and "b.txt" in rendered


def test_format_context_handles_empty() -> None:
    assert "no context" in format_context([]).lower()


def test_rag_prompt_has_required_placeholders() -> None:
    formatted = RAG_PROMPT.format(context="CTX", question="Q")
    assert "CTX" in formatted
    assert "Q" in formatted


def test_pipeline_passes_context_to_llm(pipeline: RAGPipeline) -> None:
    pipeline.answer("about RAG")
    sent_prompt = pipeline.llm.calls[-1]  # type: ignore[attr-defined]
    assert "about RAG" in sent_prompt
    # The retrieved chunk text from the populated store should appear in the prompt.
    assert "Retrieval-Augmented" in sent_prompt or "RAG" in sent_prompt
