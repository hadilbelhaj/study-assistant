from rag_study_assistant.context import build_context


def test_build_context_preserves_source_metadata():
    chunk = {
        "id": "chunk-123",
        "text": "Java collections are containers for objects.",
        "course": "java",
        "lecture": "Chapitre 6 - Les collections",
        "section": "Présentation du framework collection",
        "source_file": "lecture6-les-collections.pdf",
        "page_start": 2,
        "page_end": 2,
    }

    result = build_context([chunk])
    source = result["sources"][0]

    assert source["course"] == "java"
    assert source["lecture"] == "Chapitre 6 - Les collections"
    assert source["section"] == "Présentation du framework collection"
    assert source["source_file"] == "lecture6-les-collections.pdf"
    assert source["page_start"] == 2
    assert source["page_end"] == 2