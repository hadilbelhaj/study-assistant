"""Qdrant storage for course chunks."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams

from src.config import get_config
from src.embed import embed_texts, embed_query


FILTER_FIELDS = ["course", "study_year", "semester"]


def get_qdrant_client() -> QdrantClient:
    config = get_config()
    return QdrantClient(url=config.qdrant.url)


def ensure_collection() -> None:
    config = get_config()
    client = get_qdrant_client()

    if not client.collection_exists(config.qdrant.collection):
        client.create_collection(
            collection_name=config.qdrant.collection,
            vectors_config=VectorParams(size=config.embedding.dimension, distance=Distance.COSINE),
        )

    create_payload_indexes()


def create_payload_indexes() -> None:
    config = get_config()
    client = get_qdrant_client()

    for field_name in FILTER_FIELDS:
        client.create_payload_index(
            collection_name=config.qdrant.collection,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def upsert_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    config = get_config()
    client = get_qdrant_client()
    vectors = embed_texts([chunk["text"] for chunk in chunks])

    points = [
        PointStruct(id=str(chunk["id"]), vector=vector.tolist(), payload=chunk)
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=config.qdrant.collection, points=points)


def search_chunks(query: str, limit: int = 5, metadata_filter: dict | None = None) -> list:
    config = get_config()
    client = get_qdrant_client()
    query_vector = embed_query(query)

    query_filter = None
    if metadata_filter:
        query_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in metadata_filter.items()
            ]
        )

    return client.query_points(
        collection_name=config.qdrant.collection,
        query=query_vector.tolist(),
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    ).points