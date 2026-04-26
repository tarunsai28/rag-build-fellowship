"""End-to-end RAG: query → retrieve → generate. Workshop 5."""

from rag.generation.pipeline import Answer, RAGPipeline, format_context
from rag.generation.prompts import RAG_PROMPT

__all__ = ["Answer", "RAGPipeline", "RAG_PROMPT", "format_context"]
