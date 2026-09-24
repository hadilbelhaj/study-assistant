from pathlib import Path

import numpy as np
import yaml
from src.reranker import rerank
from src.embed import embed_query
from src.index import load_index
from src.sparse import search_sparse

CONFIG = yaml.safe_load(Path("config.yaml").read_text())


def retrieve_dense(
    query: str,
    index,
    chunks: list[dict],
    k: int,
    metadata_filter: dict | None = None,
) -> list[dict]:

    query_vec = np.array(
        [embed_query(query)]
    ).astype("float32")

    search_k = k * 5 if metadata_filter else k

    distances, ids = index.search(
        query_vec,
        search_k
    )

    results = []

    for dist, idx in zip(distances[0], ids[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]

        if (
            metadata_filter
            and not all(
                chunk.get(key) == val
                for key, val in metadata_filter.items()
            )
        ):
            continue

        results.append({
            **chunk,
            "dense_score": float(dist),
        })

        if len(results) == k:
            break

    return results
def retrieve_sparse(
    query: str,
    bm25,
    chunks: list[dict],
    k: int,
    metadata_filter: dict | None = None,
) -> list[dict]:

    search_k = k * 5 if metadata_filter else k

    results = search_sparse(
        bm25,
        chunks,
        query,
        search_k,
    )

    if metadata_filter:
        results = [
            result
            for result in results
            if all(
                result.get(key) == val
                for key, val in metadata_filter.items()
            )
        ]

    return results[:k]
def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    rrf_k: int = 60,
) -> list[dict]:

    scores = {}
    documents = {}

    for results in result_lists:

        for rank, result in enumerate(results, start=1):

            chunk_id = result["id"]

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            documents[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_ids:
        fused_results.append({
            **documents[chunk_id],
            "rrf_score": scores[chunk_id],
        })

    return fused_results
def retrieve(
    query: str,
    name: str = "index",
    k: int = 5,
    metadata_filter: dict | None = None,
) -> list[dict]:

    index, chunks, bm25 = load_index(name)

    candidate_k = k * 4

    dense_results = retrieve_dense(
        query=query,
        index=index,
        chunks=chunks,
        k=candidate_k,
        metadata_filter=metadata_filter,
    )

    sparse_results = retrieve_sparse(
        query=query,
        bm25=bm25,
        chunks=chunks,
        k=candidate_k,
        metadata_filter=metadata_filter,
    )

    fused_results = reciprocal_rank_fusion(
        [
            dense_results,
            sparse_results,
        ]
    )

    candidates = fused_results[:candidate_k]

    return rerank(
        query=query,
        chunks=candidates,
        top_k=k,
    )