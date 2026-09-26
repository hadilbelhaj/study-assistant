"""Thin wrapper around the embedding model named in config.yaml.

Kept to one function on purpose — swapping models means editing
config.yaml, not this file.
"""
from functools import lru_cache
from pathlib import Path
from src.config import get_config
import yaml
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _model():
    config = get_config()
    return SentenceTransformer(config.embedding.model)
    
def embed_texts(texts: list[str]):
    """Returns a numpy array of shape (len(texts), dim)."""
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_query(query: str):
    return _model().encode([query], normalize_embeddings=True)[0]
