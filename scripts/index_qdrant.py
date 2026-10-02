import json
from pathlib import Path

from src.config import get_config
from src.qdrant_storage import ensure_collection, get_qdrant_client, upsert_chunks


BATCH_SIZE = 32


def load_chunks(processed_dir: Path) -> list[dict]:
    chunks = []

    for path in sorted(processed_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                chunks.append(json.loads(line))

    return chunks


def main() -> None:
    config = get_config()
    ensure_collection()

    chunks = load_chunks(config.paths.processed_dir)

    if not chunks:
        raise ValueError(f"No JSONL chunks found in {config.paths.processed_dir}")

    print(f"Found {len(chunks)} chunks.")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Qdrant collection: {config.qdrant.collection}")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        upsert_chunks(batch)

        end = min(start + BATCH_SIZE, len(chunks))
        print(f"Indexed {end}/{len(chunks)} chunks.")

    client = get_qdrant_client()
    info = client.get_collection(config.qdrant.collection)

    print("\nQdrant indexing completed.")
    print(f"Points in collection: {info.points_count}")


if __name__ == "__main__":
    main()