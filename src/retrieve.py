"""Hybrid retrieval: dense + sparse + RRF + reranking."""

import numpy as np

from src.config import get_config
from src.embed import embed_query
from src.reranker import rerank
from src.storage import LocalStore


def retrieve_dense(query: str, store: LocalStore, k: int, metadata_filter: dict | None = None) -> list[dict]:
    query_vector = np.asarray(embed_query(query), dtype="float32")
    return store.search_dense(query_vector=query_vector, k=k, metadata_filter=metadata_filter)


def retrieve_sparse(query: str, store: LocalStore, k: int, metadata_filter: dict | None = None) -> list[dict]:
    return store.search_sparse(query=query, k=k, metadata_filter=metadata_filter)


def reciprocal_rank_fusion(result_lists: list[list[dict]], rrf_k: int) -> list[dict]:
    scores, documents = {}, {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
            documents[chunk_id] = result

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [{**documents[chunk_id], "rrf_score": scores[chunk_id]} for chunk_id in ranked_ids]


def retrieve(query: str, name: str = "index", k: int | None = None, metadata_filter: dict | None = None) -> list[dict]:
    config = get_config()
    top_k = k if k is not None else config.retrieval.top_k
    candidate_k = config.retrieval.candidate_k
    rrf_k = config.retrieval.rrf_k

    store = LocalStore(name)
    store.load()

    dense_results = retrieve_dense(query=query, store=store, k=candidate_k, metadata_filter=metadata_filter)
    sparse_results = retrieve_sparse(query=query, store=store, k=candidate_k, metadata_filter=metadata_filter)

    fused_results = reciprocal_rank_fusion([dense_results, sparse_results], rrf_k=rrf_k)
    candidates = fused_results[:candidate_k]

    return rerank(query=query, chunks=candidates, top_k=top_k)