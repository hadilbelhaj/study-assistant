"""Generate grounded answers from retrieved course context."""

import re

from ollama import chat

from src.config import get_config
from src.context import build_context

NO_ANSWER = "Ce n'est pas dans vos documents fournis."

PROMPT_TEMPLATE = """You are a university study assistant.

Answer the question using only the provided course material.

Each context block has a source label such as [S1], [S2], or [S3].

For factual statements supported by the context, cite the relevant source label.
Use only the source labels provided in the context.
Do not invent source labels or citation information.

If the provided context does not contain enough information to answer the question reliably, say:
"Ce n'est pas dans vos documents fournis."

Context:

{context}

Question:
{query}

Answer:
"""


def generate(query: str, chunks: list[dict]) -> dict:
    config = get_config()

    if not chunks:
        return {"answer": NO_ANSWER, "sources": []}

    context_data = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(
        context=context_data["context"],
        query=query,
    )

    response = chat(
        model=config.generation.model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"].strip()

    if NO_ANSWER in answer:
        return {"answer": answer, "sources": []}

    # Matches S1, S2... inside [S1], [S1, S2], [S1][S3], etc.
    cited_ids = {f"S{n}" for n in re.findall(r"\bS(\d+)\b", answer)}
    sources = [s for s in context_data["sources"] if s["id"] in cited_ids]

    # Model forgot to cite: fall back to everything that was in the context
    if not sources:
        sources = context_data["sources"]

    return {"answer": answer, "sources": sources}