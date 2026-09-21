import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_sparse_index(
    chunks: list[dict],
    path: Path
) -> None:

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    with path.open("wb") as f:
        pickle.dump(bm25, f)


def load_sparse_index(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def search_sparse(
    bm25,
    chunks: list[dict],
    query: str,
    k: int
) -> list[dict]:

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for idx in top_indices:
        results.append({
            **chunks[idx],
            "sparse_score": float(scores[idx]),
        })

    return results