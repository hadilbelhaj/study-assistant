"""Local knowledge-base storage using FAISS + BM25."""

import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from src.config import get_config
from src.embed import embed_texts
from src.sparse import build_sparse_index, load_sparse_index, search_sparse


class LocalStore:
    """Persistent local storage for chunks and retrieval indexes."""
    def __init__(self, name: str = "index"):
        config = get_config()
        self.vectorstore_dir = config.paths.vectorstore_dir
        self.name = name
        self.faiss_path = self.vectorstore_dir / f"{name}.faiss"
        self.metadata_path = self.vectorstore_dir / f"{name}.meta.pkl"
        self.bm25_path = self.vectorstore_dir / f"{name}.bm25.pkl"
        self.index = None
        self.chunks = None
        self.bm25 = None

    def build(self, chunks: list[dict]) -> None:
        """Build and save FAISS, metadata, and BM25."""
        if not chunks:
            raise ValueError("Cannot build storage from empty chunks.")

        vectors = np.asarray(embed_texts([chunk["text"] for chunk in chunks]), dtype="float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_path))

        with self.metadata_path.open("wb") as file:
            pickle.dump(chunks, file)

        build_sparse_index(chunks, self.bm25_path)
        # Keep the newly built data available in memory.
        self.index = index
        self.chunks = chunks

    def load(self) -> None:
        """Load the persisted indexes into memory."""   
        self.index = faiss.read_index(str(self.faiss_path))
        with self.metadata_path.open("rb") as file:
            self.chunks = pickle.load(file)
        self.bm25 = load_sparse_index(self.bm25_path)

    def search_dense(self, query_vector: np.ndarray, k: int, metadata_filter: dict | None = None) -> list[dict]:
        """Search the dense FAISS index."""
        if self.index is None:
            self.load()

        search_k = k * 5 if metadata_filter else k
        distances, ids = self.index.search(np.array([query_vector], dtype="float32"), search_k)

        results = []
        for distance, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx]
            if metadata_filter and not all(chunk.get(key) == value for key, value in metadata_filter.items()):
                continue

            results.append({**chunk, "dense_score": float(distance)})
            if len(results) == k:
                break

        return results

    def search_sparse(self, query: str, k: int, metadata_filter: dict | None = None) -> list[dict]:
        """Search the sparse BM25 index."""
        if self.bm25 is None:
            self.load()

        search_k = k * 5 if metadata_filter else k
        results = search_sparse(self.bm25, self.chunks, query, search_k)

        if metadata_filter:
            results = [result for result in results if all(result.get(key) == value for key, value in metadata_filter.items())]

        return results[:k]


def load_chunks_from_jsonl(processed_dir: Path) -> list[dict]:
    """Load all processed chunks from JSONL files."""
    chunks = []
    for jsonl_path in Path(processed_dir).rglob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as file:
            chunks.extend(json.loads(line) for line in file)

    return chunks