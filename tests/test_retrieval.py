
"""Unit tests for Qdrant hybrid retrieval and reranking."""

from types import SimpleNamespace

import rag_study_assistant.retrieve as retrieval


def make_result(chunk_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        payload={
            "text": f"Content of chunk {chunk_id}",
            "course": "java",
        },
        score=score,
    )


def make_config(top_k: int = 2, candidate_k: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            top_k=top_k,
            candidate_k=candidate_k,
        )
    )


def test_retrieve_stages_uses_hybrid_search_and_reranks(monkeypatch):
    monkeypatch.setattr(
        retrieval, "get_config", lambda: make_config()
    )

    results = [
        make_result("A", 0.95),
        make_result("B", 0.90),
        make_result("C", 0.85),
    ]

    calls = {}

    def fake_hybrid_search(query, limit, candidate_k, metadata_filter):
        calls["query"] = query
        calls["limit"] = limit
        calls["candidate_k"] = candidate_k
        calls["metadata_filter"] = metadata_filter
        return results

    monkeypatch.setattr(retrieval, "hybrid_search", fake_hybrid_search)

    def fake_rerank(query, chunks, top_k):
        calls["rerank_query"] = query
        calls["rerank_chunks"] = chunks
        calls["rerank_top_k"] = top_k
        return chunks[:top_k]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    metadata_filter = {"course": "java"}

    stages = retrieval.retrieve_stages(
        query="Explain Java collections",
        metadata_filter=metadata_filter,
    )

    assert calls["query"] == "Explain Java collections"
    assert calls["limit"] == 20
    assert calls["candidate_k"] == 20
    assert calls["metadata_filter"] == metadata_filter

    assert len(calls["rerank_chunks"]) == 3
    assert calls["rerank_top_k"] == 2

    assert len(stages["hybrid"]) == 2
    assert len(stages["reranked"]) == 2

    assert stages["hybrid"][0]["id"] == "A"
    assert stages["hybrid"][0]["hybrid_score"] == 0.95


def test_retrieve_returns_reranked_results(monkeypatch):
    expected = [
        {"id": "B", "text": "Relevant chunk"},
        {"id": "A", "text": "Another chunk"},
    ]

    def fake_retrieve_stages(query, k=None, metadata_filter=None):
        return {
            "hybrid": [{"id": "A"}, {"id": "B"}],
            "reranked": expected,
        }

    monkeypatch.setattr(
        retrieval, "retrieve_stages", fake_retrieve_stages
    )

    results = retrieval.retrieve("test query")

    assert results == expected


def test_retrieve_stages_handles_metadata_filter(monkeypatch):
    monkeypatch.setattr(
        retrieval, "get_config", lambda: make_config()
    )

    received = {}

    def fake_hybrid_search(query, limit, candidate_k, metadata_filter):
        received["metadata_filter"] = metadata_filter
        return []

    monkeypatch.setattr(retrieval, "hybrid_search", fake_hybrid_search)

    monkeypatch.setattr(
        retrieval,
        "rerank",
        lambda query, chunks, top_k: [],
    )

    retrieval.retrieve_stages(
        query="test query",
        metadata_filter={
            "course": "java",
            "semester": "S2",
        },
    )

    assert received["metadata_filter"] == {
        "course": "java",
        "semester": "S2",
    }
