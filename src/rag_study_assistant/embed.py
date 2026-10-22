"""Embedding model loading and inference."""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import get_config

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per process."""

    config = get_config()
    return SentenceTransformer(config.embedding.model)


def embed_texts(texts: list[str]):
    """Embed multiple texts using the shared model."""

    model = get_embedding_model()
    return model.encode(texts,normalize_embeddings=True,show_progress_bar=True,)


def embed_query(query: str):
    """Embed one query using the shared model."""
    model = get_embedding_model()
    return model.encode([query],normalize_embeddings=True,)[0]