"""End-to-end RAG pipeline: question → retrieved chunks → grounded answer.

Workshop 5 deliverable. Composes a :class:`Retriever` and an :class:`LLMProvider`
with the prompt template from :mod:`rag.generation.prompts`.
"""

# ruff: noqa: F401  -- imports become used once Workshop 5 is implemented.

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import settings
from rag.generation.prompts import RAG_PROMPT
from rag.ingestion.models import Chunk
from rag.providers.base import LLMProvider
from rag.retrieval.retriever import Retriever
from rag.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Answer:
    """A grounded answer produced by the RAG pipeline.

    Attributes:
        text: The model-generated answer text.
        sources: The chunks the model was given as context.
        question: The original question, echoed for convenience.
    """

    text: str
    sources: list[Chunk] = field(default_factory=list)
    question: str = ""


def format_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a single string for the prompt.

    Each chunk is rendered as::

        [1] (source: <source>, chunk: <index>)
        <text>

    Numbering starts at 1 to match what students would naturally cite.
    """
    # TODO Workshop 5:
    # - If ``chunks`` is empty, return a placeholder string like
    #   ``"(no context retrieved)"``.
    # - Otherwise, render each chunk as a numbered block: a header line with
    #   ``[i]`` plus source + chunk index, then the text on the next line.
    # - Join blocks with a blank line.
    raise NotImplementedError("Workshop 5: implement format_context")


class RAGPipeline:
    """Orchestrates retrieval + generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        prompt_template: str = RAG_PROMPT,
    ) -> None:
        """Construct a RAGPipeline.

        Args:
            retriever: Source of relevant chunks for a query.
            llm: Provider used to generate the final answer.
            prompt_template: Prompt template with ``{context}`` and ``{question}``
                placeholders. Defaults to :data:`RAG_PROMPT`.
        """
        # TODO Workshop 5:
        # - Store ``retriever``, ``llm``, and ``prompt_template`` on ``self``.
        raise NotImplementedError("Workshop 5: implement RAGPipeline.__init__")

    def answer(
        self,
        question: str,
        *,
        k: int | None = None,
        temperature: float | None = None,
    ) -> Answer:
        """Run the full RAG loop and return a grounded :class:`Answer`.

        Args:
            question: The user's natural-language question.
            k: Optional override for the number of chunks to retrieve.
            temperature: Optional override for the LLM sampling temperature.

        Returns:
            An :class:`Answer` carrying the generated text plus the source chunks.
        """
        # TODO Workshop 5:
        # 1. Guard against empty/whitespace questions (return an Answer that says so).
        # 2. Call ``self.retriever.search(question, k=k)`` to get the context chunks.
        # 3. Build the prompt by formatting ``self.prompt_template`` with
        #    ``context=format_context(chunks)`` and ``question=question``.
        # 4. Call ``self.llm.generate(prompt, temperature=..., max_tokens=...)``.
        # 5. Wrap the result in :class:`Answer` and return it.
        raise NotImplementedError("Workshop 5: implement RAGPipeline.answer")
