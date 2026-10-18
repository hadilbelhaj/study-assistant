"""Generate grounded answers from retrieved course context."""

import re

from ollama import chat

from src.config import get_config
from src.context import build_context

NO_ANSWER = "Ce n'est pas dans vos documents fournis."

SYSTEM_PROMPT = """You are a university study assistant.
Answer using ONLY the course context provided by the user.

RULES:
1. Do not use external knowledge. Do not invent facts, explanations, APIs, code, examples, or definitions.
2. You MAY reuse code or examples that appear in the context. You may NOT write new ones.
3. Cite with exactly [S1], [S2], etc., at the end of each sentence that uses the context.
   Never write "According to [S1]" and never invent a source label.Example of the expected style:"En Java, une classe hérite d'une autre avec le mot-clé extends [S1]
4. If the context answers only part of the question, answer that part and say clearly
   which part is missing from the course documents.
5. Reply "Ce n'est pas dans vos documents fournis." ONLY if none of the context
   passages is related to the question. If some passages are related, answer with
   what they say, and note what is missing.
6. Preserve the terminology used in the course material.
7. If the message is not about the course (greeting, thanks...), reply in one short sentence
   with no citations.
"""

USER_TEMPLATE = """Context:

{context}

Question:

{query}"""


def generate(query: str, chunks: list[dict]) -> dict:
    config = get_config()

    if not chunks:
        return {"answer": NO_ANSWER, "sources": []}

    context_data = build_context(chunks)
    """prompt = PROMPT_TEMPLATE.format(
        context=context_data["context"],
        query=query,
    )"""

    response = chat(
    model=config.generation.model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=context_data["context"], query=query)},
    ],
    options={"temperature": 0, "num_ctx": 8192},)
    print("PROMPT TOKENS:", response.get("prompt_eval_count"))
    answer = response["message"]["content"].strip()

    if NO_ANSWER in answer:
        cleaned = answer.replace(NO_ANSWER, "").strip()
        if len(cleaned) < 20:          # the answer is only "not in your documents"
            return {"answer": NO_ANSWER, "sources": []}
        else:
            answer = cleaned               # otherwise keep the real answer and its source

    # Matches S1, S2... inside [S1], [S1, S2], [S1][S3], etc.
    cited_ids = {f"S{n}" for n in re.findall(r"\bS(\d+)\b", answer)}
    sources = [s for s in context_data["sources"] if s["id"] in cited_ids]
    return {"answer": answer, "sources": sources}