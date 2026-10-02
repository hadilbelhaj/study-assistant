from src.config import get_config
from src.qdrant_storage import get_qdrant_client, hybrid_search
from src.retrieve import retrieve


def main() -> None:
    config = get_config()
    client = get_qdrant_client()
    info = client.get_collection(config.qdrant.collection)

    print("QDRANT")
    print("=" * 60)
    print(f"Collection: {config.qdrant.collection}")
    print(f"Points: {info.points_count}")
    print(f"Vector config: {info.config.params}")

    query = "Quelle est la différence entre une transmission synchrone et asynchrone ?"

    print("\nHYBRID SEARCH")
    print("=" * 60)

    hybrid_results = hybrid_search(
        query=query,
        limit=config.retrieval.top_k,
        candidate_k=config.retrieval.candidate_k,
    )

    print(f"Retrieved: {len(hybrid_results)}")

    for rank, result in enumerate(hybrid_results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"ID: {result.id}")
        print(f"Course: {result.payload.get('course')}")
        print(f"Semester: {result.payload.get('semester')}")
        print(f"Section: {result.payload.get('section')}")
        print(f"Text: {result.payload.get('text', '')[:250]}")

    print("\nRERANKED RETRIEVAL")
    print("=" * 60)

    reranked_results = retrieve(query=query)

    print(f"Retrieved: {len(reranked_results)}")

    for rank, result in enumerate(reranked_results, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"Section: {result.get('section')}")
        print(f"Course: {result.get('course')}")
        print(f"Text: {result['text'][:250]}")

    print("\nFILTERED HYBRID SEARCH")
    print("=" * 60)

    filtered_results = hybrid_search(
        query=query,
        limit=config.retrieval.top_k,
        candidate_k=config.retrieval.candidate_k,
        metadata_filter={
            "course": "communication-numerique",
            "semester": "S1",
        },
    )

    print(f"Retrieved with filter: {len(filtered_results)}")

    for rank, result in enumerate(filtered_results, start=1):
        print(f"{rank}. {result.payload.get('course')} | {result.payload.get('semester')} | {result.payload.get('section')}")

    print("\nQDRANT PIPELINE TEST COMPLETED")


if __name__ == "__main__":
    main()