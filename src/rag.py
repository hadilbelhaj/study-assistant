"""The one function scripts/run_cli.py and notebooks call."""
from src.generate import generate
from src.retrieve import retrieve


def rag_query(query: str, metadata_filter: dict | None = None) -> dict:
    chunks = retrieve(query, metadata_filter=metadata_filter)

    # Phase 2 will replace this with a real similarity threshold check
    # (see config.yaml: retrieval.min_similarity_threshold) once FAISS
    # distances are calibrated against your own corpus.
    return generate(query, chunks)