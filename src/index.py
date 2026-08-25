"""FAISS IndexFlatL2 build/save/load, plus the metadata sidecar.

FAISS only stores vectors — chunk text and metadata (course, lecture,
page, ...) live in a parallel list, saved next to the index so a
result's vector row number maps back to its chunk.
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import yaml

from src.embed import embed_texts

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
VECTORSTORE_DIR = Path(CONFIG["paths"]["vectorstore_dir"])


def build_index(chunks: list[dict], name: str = "index") -> None:
    vectors = np.array(embed_texts([c["text"] for c in chunks])).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTORSTORE_DIR / f"{name}.faiss"))
    with (VECTORSTORE_DIR / f"{name}.meta.pkl").open("wb") as f:
        pickle.dump(chunks, f)


def load_index(name: str = "index"):
    index = faiss.read_index(str(VECTORSTORE_DIR / f"{name}.faiss"))
    with (VECTORSTORE_DIR / f"{name}.meta.pkl").open("rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def load_chunks_from_jsonl(processed_dir: Path) -> list[dict]:
    """Gather every chunk written by ingest.py under data/processed/<course>/."""
    chunks = []
    for jsonl_path in Path(processed_dir).rglob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f)
    return chunks
