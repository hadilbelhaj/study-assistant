"""Retrieval using Qdrant hybrid search and reranking."""

from rag_study_assistant.config import get_config
from rag_study_assistant.qdrant_storage import hybrid_search
from rag_study_assistant.reranker import rerank


def retrieve_stages(query: str, k: int | None = None, metadata_filter: dict | None = None) -> dict[str, list[dict]]:
    config = get_config()
    top_k = k if k is not None else config.retrieval.top_k
    candidate_k = config.retrieval.candidate_k

    hybrid_results = hybrid_search(
        query=query,
        limit=candidate_k,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
    )

    candidates = [
        {**result.payload, "id": str(result.id), "hybrid_score": float(result.score)}
        for result in hybrid_results
    ]

    reranked_results = rerank(query=query, chunks=candidates, top_k=top_k)

    return {
        "hybrid": candidates[:top_k],
        "reranked": reranked_results,
    }


def retrieve(query: str, k: int | None = None, metadata_filter: dict | None = None) -> list[dict]:
    return retrieve_stages(query=query, k=k, metadata_filter=metadata_filter)["reranked"]