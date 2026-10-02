from src.qdrant_storage import search_chunks


def print_results(title: str, results: list) -> None:
    print(f"\n{title}")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"ID: {result.id}")
        print(f"Course: {result.payload.get('course')}")
        print(f"Semester: {result.payload.get('semester')}")
        print(f"Section: {result.payload.get('section')}")
        print(f"Text: {result.payload.get('text', '')[:250]}")


def main() -> None:
    query = "Quels sont les principaux codes utilisés pour la transmission en bande de base ?"

    results = search_chunks(query=query, limit=3)
    print_results("Without metadata filter", results)

    filtered_results = search_chunks(
        query=query,
        limit=3,
        metadata_filter={
            "course": "communication-numerique",
            "semester": "S1",
        },
    )
    print_results("With course + semester filter", filtered_results)

    wrong_filter_results = search_chunks(
        query=query,
        limit=3,
        metadata_filter={
            "course": "java",
        },
    )
    print_results("With wrong course filter", wrong_filter_results)


if __name__ == "__main__":
    main()