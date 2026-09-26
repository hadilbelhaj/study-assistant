"""Hybrid retrieval: dense + sparse + RRF + reranking."""

import numpy as np

from src.config import get_config
from src.embed import embed_query
from src.index import load_index
from src.reranker import rerank
from src.sparse import search_sparse


def retrieve_dense(query: str, index, chunks: list[dict], k: int, metadata_filter: dict | None = None) -> list[dict]:
    query_vec = np.array([embed_query(query)]).astype("float32")
    search_k = k * 5 if metadata_filter else k
    distances, ids = index.search(query_vec, search_k)

    results = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        if metadata_filter and not all(chunk.get(key) == value for key, value in metadata_filter.items()):
            continue

        results.append({**chunk, "dense_score": float(dist)})
        if len(results) == k:
            break

    return results


def retrieve_sparse(query: str, bm25, chunks: list[dict], k: int, metadata_filter: dict | None = None) -> list[dict]:
    search_k = k * 5 if metadata_filter else k
    results = search_sparse(bm25, chunks, query, search_k)

    if metadata_filter:
        results = [result for result in results if all(result.get(key) == value for key, value in metadata_filter.items())]

    return results[:k]


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

    index, chunks, bm25 = load_index(name)

    dense_results = retrieve_dense(query=query, index=index, chunks=chunks, k=candidate_k, metadata_filter=metadata_filter)
    sparse_results = retrieve_sparse(query=query, bm25=bm25, chunks=chunks, k=candidate_k, metadata_filter=metadata_filter)

    fused_results = reciprocal_rank_fusion([dense_results, sparse_results], rrf_k=rrf_k)
    candidates = fused_results[:candidate_k]

    return rerank(query=query, chunks=candidates, top_k=top_k)