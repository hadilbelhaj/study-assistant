"""Query -> top-k chunks, with an optional metadata filter.

Phase 2 will add BM25 + reranking here; for the MVP this is plain
dense retrieval only.
"""
from pathlib import Path

import numpy as np
import yaml

from src.embed import embed_query
from src.index import load_index

CONFIG = yaml.safe_load(Path("config.yaml").read_text())


def retrieve(query: str, name: str = "index", k: int | None = None,
             metadata_filter: dict | None = None) -> list[dict]:
    """metadata_filter example: {"course": "Reseaux", "doc_type": "exam"}"""
    k = k or CONFIG["retrieval"]["top_k"]
    index, chunks = load_index(name)

    query_vec = np.array([embed_query(query)]).astype("float32")
    # Over-fetch so filtering doesn't leave us short of k results.
    distances, ids = index.search(query_vec, k * 5 if metadata_filter else k)

    results = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if metadata_filter and not all(chunk.get(key) == val for key, val in metadata_filter.items()):
            continue
        results.append({**chunk, "distance": float(dist)})
        if len(results) == k:
            break
    return results
