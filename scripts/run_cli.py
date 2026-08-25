"""Plain CLI loop — Phase 1 interface, no UI yet.

    python scripts/run_cli.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import rag_query

if __name__ == "__main__":
    print("RAG study assistant — type a question, or 'quit' to exit.\n")
    while True:
        query = input("> ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        result = rag_query(query)
        print(f"\n{result['answer']}\n")
        for source in result["sources"]:
            print(f"  - {source['source_file']} p.{source['page']}")
        print()
