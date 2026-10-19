from src.config import get_config
from src.qdrant_storage import get_qdrant_client

client = get_qdrant_client()
name = get_config().qdrant.collection

if client.collection_exists(name):
    client.delete_collection(name)
    print(f"Collection '{name}' deleted.")
else:
    print(f"Collection '{name}' does not exist.")