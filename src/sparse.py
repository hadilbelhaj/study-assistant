from functools import lru_cache
import pickle
import re
from pathlib import Path
import json
from config import get_config
from rank_bm25 import BM25Okapi




@lru_cache(maxsize=1)
def get_bm25_resources():
    config = get_config()
    bm25_path = config.paths.vectorstore_dir / "index.bm25.pkl"
    bm25 = load_sparse_index(bm25_path)

    chunks = []
    for path in sorted(Path(config.paths.processed_dir).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            chunks.extend(json.loads(line) for line in file)

    return bm25, chunks
    
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
