"""Qwen reranker loading and inference."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import get_config


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Load the reranker once per process."""

    config = get_config()
    return CrossEncoder(config.reranker.model)


def rerank(query: str,chunks: list[dict],top_k: int,) -> list[dict]:

    model = get_reranker_model()

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    ranked_chunks = []

    for chunk, score in zip(chunks, scores):
        ranked_chunks.append({**chunk,"rerank_score": float(score),})

    ranked_chunks.sort(key=lambda chunk: chunk["rerank_score"],reverse=True)

    return ranked_chunks[:top_k]