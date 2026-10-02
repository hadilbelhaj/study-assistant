from src.config import get_config
from src.qdrant_storage import create_collection


def main() -> None:
    config = get_config()
    from src.qdrant_storage import get_qdrant_client

    client = get_qdrant_client()

    if client.collection_exists(config.qdrant.collection):
        client.delete_collection(config.qdrant.collection)
        print(f"Deleted collection: {config.qdrant.collection}")

    create_collection()
    print(f"Created hybrid collection: {config.qdrant.collection}")


if __name__ == "__main__":
    main()