"""Build structured LLM context from retrieved chunks."""

from typing import Any


def build_context(chunks: list[dict]) -> dict[str, Any]:
    """Format retrieved chunks and assign temporary source IDs."""
    seen_ids = set()
    context_parts = []
    sources = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk["id"])

        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        source_id = f"S{len(sources) + 1}"

        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        if page_start is None:
            pages = "Unknown"
        elif page_end is None or page_end == page_start:
            pages = str(page_start)
        else:
            pages = f"{page_start}-{page_end}"

        context_parts.append(
            f"[{source_id}]\n"
            f"Course: {chunk.get('course', 'Unknown')}\n"
            f"Lecture: {chunk.get('lecture', 'Unknown')}\n"
            f"Section: {chunk.get('section') or 'Unknown'}\n"
            f"Pages: {pages}\n"
            f"Content:\n{chunk.get('text', '').strip()}"
        )

        sources.append(
            {
                "id": source_id,
                "chunk_id": chunk_id,
                "course": chunk.get("course"),
                "lecture": chunk.get("lecture"),
                "section": chunk.get("section"),
                "source_file": chunk.get("source_file"),
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }