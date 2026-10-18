
"""Rewrite follow-up questions using recent conversation history."""

import json
import logging

from ollama import chat

from src.config import get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You rewrite questions for a university course RAG system.

Your task is to produce one standalone question from the user's latest message.

Rules:
1. Use conversation history only to resolve references and understand follow-ups.
2. If the latest message is a follow-up, rewrite it so it can be understood
   without seeing the conversation.
3. Preserve the user's intended question, including requests for additional
   information, comparisons, explanations, or examples.
4. If the latest message introduces a new topic or is already self-contained,
   return it unchanged.
5. Do not mix an unrelated previous topic into a new question.
6. Ignore greetings, thanks, acknowledgements, and goodbyes when resolving
   course-related context.
7. Do not answer the question or add facts, assumptions, or explanations.
8. Preserve the language of the latest user message.
9. Output only the resulting question, without quotes or commentary.
10. Remove filler such as "i meant", "I mean", "actually", "je veux dire".
    Always include the main subject (for example "Java") if the conversation makes it clear.
"""


def rewrite_query(query: str, history: list[dict]) -> str:
    """Resolve follow-ups into standalone questions."""
    if not history:
        return query

    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-10:]
        if message.get("role") in {"user", "assistant"}
    ]

    if not recent_history:
        return query

    prompt = f"""Recent conversation (JSON):
{json.dumps(recent_history, ensure_ascii=False)}

Latest user question:
{query}

Return the standalone version of the latest question according to your rules."""

    try:
        response = chat(
            model=get_config().generation.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        rewritten = response["message"]["content"].strip().strip('"')
        return rewritten or query

    except Exception:
        logger.exception("Query rewriting failed; using the original question.")
        return query
