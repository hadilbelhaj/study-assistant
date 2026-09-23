"""Rerank retrieved chunks using Qwen3-Reranker-0.6B."""

from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = _model().predict(pairs)
    ranked_chunks = []
    for chunk, score in zip(chunks, scores):
        ranked_chunks.append({
            **chunk,
            "rerank_score": float(score),
        })
    ranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )
    return ranked_chunks[:top_k]