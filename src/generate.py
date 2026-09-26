"""Builds the grounded prompt (roadmap section 2) and calls the LLM.

Isolated to one file on purpose: swapping providers later means
editing only this file, not retrieve.py or rag.py.

Ollama runs fully locally -- no API key, no internet call, no cost.
It must be running (the Ollama app / `ollama serve`) and the model
must already be pulled once via `ollama pull <model>`.
"""
from pathlib import Path
from src.config import get_config
import yaml
from ollama import chat

PROMPT_TEMPLATE = """Contexte (extrait de {source_file}, page {page}):
{retrieved_chunks}

Question: {query}

Réponds uniquement à partir du contexte ci-dessus. Cite la source
(fichier + page) pour chaque affirmation. Si le contexte ne
contient pas la réponse, dis-le clairement plutôt que d'inventer.
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['source_file']}, p.{c['page']}] {c['text']}" for c in chunks
    )
    first = chunks[0] if chunks else {"source_file": "?", "page": "?"}
    return PROMPT_TEMPLATE.format(
        source_file=first["source_file"],
        page=first["page"],
        retrieved_chunks=context,
        query=query,
    )


def generate(query: str, chunks: list[dict]) -> str:
    config = get_config()
    if not chunks:
        return "Ce n'est pas dans vos documents fournis."
    prompt = build_prompt(query, chunks)
    response = chat(
        model=config.generation.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]