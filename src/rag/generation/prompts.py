"""Prompt templates used by the RAG pipeline.

Kept intentionally minimal in the starter so students can iterate on it. The
shape — explicit "use ONLY the context" instruction, explicit "say so when you
don't know" — is the bare minimum for a grounded RAG answer; without it the
model will happily hallucinate from prior knowledge.
"""

from __future__ import annotations

RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context provided. \
If the context doesn't contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
"""Default RAG prompt template. Format with ``context`` and ``question`` kwargs."""
