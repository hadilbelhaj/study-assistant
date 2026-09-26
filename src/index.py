"""FAISS index build/load and chunk metadata persistence."""
import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from src.config import get_config
from src.embed import embed_texts
from src.sparse import build_sparse_index, load_sparse_index


def _get_index_paths(name: str) -> tuple[Path, Path, Path]:
    """Return paths for FAISS, metadata, and BM25 indexes."""
    vectorstore_dir = get_config().paths.vectorstore_dir
    return (
        vectorstore_dir / f"{name}.faiss",
        vectorstore_dir / f"{name}.meta.pkl",
        vectorstore_dir / f"{name}.bm25.pkl",
    )

def build_index(chunks: list[dict], name: str = "index") -> None:
    """Build and persist the FAISS + metadata + BM25 indexes."""
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")

    faiss_path, metadata_path, bm25_path = _get_index_paths(name)

    # Generate embeddings and build FAISS index.
    vectors = np.asarray(embed_texts([chunk["text"] for chunk in chunks]), dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Ensure the storage directory exists and save FAISS.
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))

    # Save chunk metadata.
    with metadata_path.open("wb") as file:
        pickle.dump(chunks, file)

    # Build and save BM25.
    build_sparse_index(chunks, bm25_path)


def load_index(name: str = "index"):
    """Load FAISS, chunk metadata, and BM25."""
    faiss_path, metadata_path, bm25_path = _get_index_paths(name)

    index = faiss.read_index(str(faiss_path))
    with metadata_path.open("rb") as file:
        chunks = pickle.load(file)
    bm25 = load_sparse_index(bm25_path)

    return index, chunks, bm25


def load_chunks_from_jsonl(processed_dir: Path) -> list[dict]:
    """Load all processed chunks from JSONL files."""
    chunks = []
    for jsonl_path in Path(processed_dir).rglob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as file:
            chunks.extend(json.loads(line) for line in file)
    return chunks