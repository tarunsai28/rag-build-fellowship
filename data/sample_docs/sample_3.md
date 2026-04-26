# Prompt design notes for grounded answering

A grounded answer is one whose claims are supported by the provided context.
A few patterns that consistently work in this project:

## Be explicit about the source-of-truth

Instructions like "Use only the context below" outperform vaguer phrasings
("Try to use the context"). Models follow strong negative constraints
("don't make things up") more reliably than soft positive ones.

## Make the refusal path attractive

If the model is told that "I don't know based on the provided context" is a
fully acceptable answer, it picks that path far more often than the alternative
(making something up to look helpful). This single sentence in the system
prompt drops hallucination dramatically.

## Number your sources

When chunks are numbered `[1]`, `[2]`, `[3]` in the context block, the model
will cite back into them in its answer almost for free. This is the same
trick used by Anthropic's citations API and most production RAG systems.

## Keep the context window honest

Stuffing 30 chunks into the prompt rarely helps. After the first few highly
relevant chunks, additional context dilutes attention and worsens both
latency and answer quality. Default `top_k=4` for short answers; raise it
only when the corpus is dense or the question is broad.

## Iterate on the prompt, not the model

When answers are wrong, the first thing to inspect is the prompt and the
retrieved chunks — not the model. Most "the model is bad" complaints in
practice trace back to either no context being retrieved or the wrong
context being retrieved.
