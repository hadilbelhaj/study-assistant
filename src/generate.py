"""Generate grounded answers from retrieved course context."""

import re

from ollama import chat

from src.config import get_config
from src.context import build_context

NO_ANSWER = "Ce n'est pas dans vos documents fournis."

SYSTEM_PROMPT = """You are a study assistant for university courses. You answer strictly from the course excerpts given in the user message.

HOW TO ANSWER
- - Read all the excerpts, then use only the ones that directly answer the question. Ignore excerpts about other topics, even if they share some words.Read all the excerpts and use every excerpt that helps answer the question.
- Write short, direct sentences that restate what the excerpts say, with the course's own terms.
- End every sentence with the excerpt(s) it comes from, like [S1] or [S1][S2].
- Reply in the language of the question. If the question asks for a language (for example "in French"), use that language.

STRICT LIMITS
- Use only what the excerpts say. No outside knowledge, no extra explanations, no benefits or uses that are not written in them.
- Code: copy only code that appears in the excerpts, exactly as written. Never write new code or new examples. If the question asks "how" and an excerpt has a code example, include it.
- Keep every relationship in the direction the excerpts state it. Never reverse or generalize it (no "and vice versa").
- No introduction, no conclusion, no summary. Stop after the last sentence the excerpts support.
- Never say that something is "the only" difference or point, or that nothing else exists. Say only what the excerpts state.

WHEN INFORMATION IS MISSING
- Only if the question asks for something that no excerpt contains, finish with one sentence saying the course documents do not cover it.
- If no excerpt is related to the question, reply with exactly this sentence and nothing else: Ce n'est pas dans vos documents fournis.

EXAMPLE
Excerpts:
[S1] TCP est un protocole orienté connexion. Il utilise un handshake en trois étapes.
[S2] UDP n'établit pas de connexion.
Question: Quelle est la différence entre TCP et UDP ?
Answer: TCP est orienté connexion et utilise un handshake en trois étapes [S1]. UDP n'établit pas de connexion [S2].
"""

USER_TEMPLATE = """Course excerpts:

{context}

Question: {query}

Answer only from the excerpts above. Cite every sentence."""


def generate(query: str, chunks: list[dict]) -> dict:
    config = get_config()

    if not chunks:
        return {"answer": NO_ANSWER, "sources": []}

    context_data = build_context(chunks)
    response = chat(
    model=config.generation.model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=context_data["context"], query=query)},
    ],
    options={"temperature": 0, "num_ctx": 8192, "repeat_penalty": 1.0},)
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