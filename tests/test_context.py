from src.context import build_context


def test_build_context_assigns_source_ids():
    chunks = [
        {
            "id": "chunk-1",
            "text": "First chunk.",
            "course": "communication-numerique",
            "lecture": "Chapitre 1",
            "section": "Introduction",
            "source_file": "lecture1.pdf",
            "page_start": 2,
            "page_end": 3,
        },
        {
            "id": "chunk-2",
            "text": "Second chunk.",
            "course": "communication-numerique",
            "lecture": "Chapitre 1",
            "section": "Transmission",
            "source_file": "lecture1.pdf",
            "page_start": 4,
            "page_end": 4,
        },
    ]

    result = build_context(chunks)

    assert "[S1]" in result["context"]
    assert "[S2]" in result["context"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["id"] == "S1"
    assert result["sources"][1]["id"] == "S2"


def test_build_context_removes_duplicate_chunks():
    chunk = {
        "id": "chunk-1",
        "text": "Same chunk.",
        "course": "java",
        "lecture": "Chapitre 1",
        "section": "Introduction",
        "source_file": "lecture1.pdf",
        "page_start": 1,
        "page_end": 1,
    }

    result = build_context([chunk, chunk])

    assert result["context"].count("[S1]") == 1
    assert len(result["sources"]) == 1