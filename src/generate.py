"""Builds the grounded prompt (roadmap section 2) and calls the LLM.

Isolated to one file on purpose: swapping providers later means
editing only this file, not retrieve.py or rag.py.
"""
import os
from pathlib import Path

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
CONFIG = yaml.safe_load(Path("config.yaml").read_text())

PROMPT_TEMPLATE = """Contexte (extrait de {source_file}, page {page}):
{retrieved_chunks}

Question: {query}

Réponds uniquement à partir du contexte ci-dessus. Cite la source
(fichier + page) pour chaque affirmation. Si le contexte ne
contient pas la réponse, dis-le clairement plutôt que d'inventer.
"""

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def build_prompt(query: str, chunks: list[dict]) -> str:
    # Each chunk keeps its own source_file/page so the model can cite
    # per-claim even when chunks come from different lectures.
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
    if not chunks:
        return "Ce n'est pas dans vos documents fournis."

    prompt = build_prompt(query, chunks)
    response = _client.messages.create(
        model=CONFIG["generation"]["model"],
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
