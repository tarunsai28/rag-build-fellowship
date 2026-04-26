# Retrieval-Augmented Generation: a one-page primer

Retrieval-Augmented Generation (RAG) is a technique that pairs a vector-search
retrieval step with a large language model. Instead of asking the model to
answer purely from the parameters baked in at training time, we first fetch
relevant text from a document store and hand it to the model alongside the
question.

## Why bother

Foundation models have two well-known failure modes that RAG addresses:

1. **Knowledge cut-offs.** A model trained on data up to a particular date
   knows nothing later. A retriever can pull from a live index.
2. **Hallucination.** Asked a question outside its training distribution, a
   model often invents plausible-sounding answers. Grounding generation in
   retrieved context — and instructing the model to refuse if the context is
   silent — measurably reduces hallucination rates.

## The basic loop

```
   user question
        │
        ▼
   embed → vector search → top-k chunks
        │
        ▼
   prompt = "Answer using ONLY this context: {chunks}\nQuestion: {q}"
        │
        ▼
   LLM.generate(prompt) → answer
```

## What it does *not* fix

- Reasoning errors in the LLM itself.
- Bad source documents in the corpus.
- Queries that require multi-hop synthesis across many distant chunks (you
  need agentic retrieval for that).

## Where to look next

- Lewis et al. 2020 — the original "Retrieval-Augmented Generation" paper.
- The `chromadb` documentation for vector store internals.
- Anthropic's guidance on grounded prompt design.
