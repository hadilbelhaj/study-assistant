
"""CLI for the RAG study assistant.

Run from the project root:
    python -m scripts.run_cli
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversation_store import ConversationStore
from src.rag import rag_query


def format_pages(source: dict) -> str:
    start, end = source.get("page_start"), source.get("page_end")
    if start is None:
        return "?"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def format_source(source: dict) -> str:
    details = " ".join(
        part for part in (source.get("lecture"), source.get("section")) if part
    )
    line = f"  [{source['id']}] {source.get('source_file', '?')} p.{format_pages(source)}"
    return f"{line} — {details}" if details else line


def main() -> None:
    store = ConversationStore(max_messages=20, ttl_seconds=86400)
    conversation_id = str(uuid4())

    try:
        store.ping()
        print("RAG study assistant — type a question, or 'quit' to exit.")
        print(f"Conversation ID: {conversation_id}\n")

        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue

            try:
                result = rag_query(
                    query,
                    conversation_id=conversation_id,
                    conversation_store=store,
                )
            except Exception as exc:
                print(f"\nError: {exc}\n")
                continue

            print(f"\n{result['answer']}\n")

            if result["sources"]:
                print("Sources:")
                for source in result["sources"]:
                    print(format_source(source))
                print()
    finally:
        store.close()


if __name__ == "__main__":
    main()
