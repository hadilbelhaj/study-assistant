"""Plain CLI loop: Phase 1 interface, no UI yet.

    python scripts/run_cli.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    print("RAG study assistant — type a question, or 'quit' to exit.\n")

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
            result = rag_query(query)
        except Exception as exc:
            print(f"\nError: {exc}\n")
            continue

        print(f"\n{result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            for source in result["sources"]:
                print(format_source(source))
            print()


if __name__ == "__main__":
    main()