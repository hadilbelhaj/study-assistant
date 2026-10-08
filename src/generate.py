"""Generate grounded answers from retrieved course context."""

import re

from ollama import chat

from src.config import get_config
from src.context import build_context

NO_ANSWER = "Ce n'est pas dans vos documents fournis."

PROMPT_TEMPLATE = """You are a university study assistant.

FIRST, check whether the message is chitchat (greeting, thanks, goodbye,
acknowledgement) or unrelated to the course.
- If yes: reply in one short friendly sentence, in the user's language,
  with NO citations, and ignore the context completely.
  Example: "merci" -> "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."
- If no: follow the rules below.
Answer the question using ONLY the provided course material.

IMPORTANT RULES:

1. Do not use external knowledge.
2. Do not invent facts, explanations, APIs, code, examples, or definitions.
3. Every factual statement about the course must be supported by the provided context.
4. Cite factual statements using exactly [S1], [S2], [S3], etc.
5. Put the citation immediately after the sentence it supports.
6. Never write "Source", "Source Label", "Source Label:", or any other citation format.
7. Never invent a source label.
8. If the context does not contain enough information to answer reliably, say:
   "Ce n'est pas dans vos documents fournis."
9. If the question asks for an example or code and the provided context does not contain enough information to construct it reliably, do not invent one.
10. Preserve the terminology used in the course material.

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
    return {"answer": answer, "sources": sources}