"""Qdrant storage and hybrid retrieval."""

from qdrant_client import QdrantClient, models

from src.config import get_config
from src.embed import embed_query, embed_texts


FILTER_FIELDS = ["course", "study_year", "semester"]


def get_qdrant_client() -> QdrantClient:
    config = get_config()
    return QdrantClient(url=config.qdrant.url)


def create_collection() -> None:
    config = get_config()
    client = get_qdrant_client()

    client.create_collection(
        collection_name=config.qdrant.collection,
        vectors_config={
            "dense": models.VectorParams(
                size=config.embedding.dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    for field_name in FILTER_FIELDS:
        client.create_payload_index(
            collection_name=config.qdrant.collection,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def ensure_collection() -> None:
    config = get_config()
    client = get_qdrant_client()

    if not client.collection_exists(config.qdrant.collection):
        create_collection()


def build_filter(metadata_filter: dict | None = None):
    if not metadata_filter:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in metadata_filter.items()
        ]
    )


def upsert_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    config = get_config()
    client = get_qdrant_client()

    dense_vectors = embed_texts([chunk["text"] for chunk in chunks])

    points = [
        models.PointStruct(
            id=str(chunk["id"]),
            vector={
                "dense": dense_vector.tolist(),
                "sparse": models.Document(
                    text=chunk["text"],
                    model="Qdrant/bm25",
                ),
            },
            payload=chunk,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors)
    ]

    client.upsert(
        collection_name=config.qdrant.collection,
        points=points,
    )


def hybrid_search(query: str, limit: int = 5, candidate_k: int = 20, metadata_filter: dict | None = None) -> list:
    config = get_config()
    client = get_qdrant_client()
    query_filter = build_filter(metadata_filter)

    dense_vector = embed_query(query)

    return client.query_points(
        collection_name=config.qdrant.collection,
        prefetch=[
            models.Prefetch(
                query=dense_vector.tolist(),
                using="dense",
                limit=candidate_k,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=candidate_k,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        query_filter=query_filter,
        with_payload=True,
        limit=limit,
    ).points