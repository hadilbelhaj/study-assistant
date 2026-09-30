import numpy as np

from src import retrieve as retrieval


def make_chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text for {chunk_id}"}


def test_reciprocal_rank_fusion_ranks_shared_documents_higher():
    results_a = [
        make_chunk("A"),
        make_chunk("B"),
        make_chunk("C"),
    ]
    results_b = [
        make_chunk("B"),
        make_chunk("C"),
        make_chunk("D"),
    ]

    fused = retrieval.reciprocal_rank_fusion(
        [results_a, results_b],
        rrf_k=60,
    )

    ranked_ids = [result["id"] for result in fused]

    assert ranked_ids[0] == "B"
    assert set(ranked_ids) == {"A", "B", "C", "D"}


def test_reciprocal_rank_fusion_keeps_all_documents():
    results_a = [make_chunk("A"), make_chunk("B")]
    results_b = [make_chunk("C"), make_chunk("D")]

    fused = retrieval.reciprocal_rank_fusion(
        [results_a, results_b],
        rrf_k=60,
    )

    ranked_ids = [result["id"] for result in fused]

    assert len(ranked_ids) == 4
    assert set(ranked_ids) == {"A", "B", "C", "D"}


def test_retrieve_uses_candidate_k_and_reranks(monkeypatch):
    config = type(
        "Config",
        (),
        {
            "retrieval": type(
                "RetrievalConfig",
                (),
                {"top_k": 5, "candidate_k": 20, "rrf_k": 60},
            )()
        },
    )()

    monkeypatch.setattr(retrieval, "get_config", lambda: config)

    class FakeStore:
        def __init__(self, name: str):
            self.name = name

        def load(self):
            pass

        def search_dense(self, query_vector, k, metadata_filter=None):
            assert k == 20
            return [make_chunk("A"), make_chunk("B")]

        def search_sparse(self, query, k, metadata_filter=None):
            assert k == 20
            return [make_chunk("B"), make_chunk("C")]

    monkeypatch.setattr(retrieval, "LocalStore", FakeStore)
    monkeypatch.setattr(
        retrieval,
        "embed_query",
        lambda query: np.array([0.1, 0.2, 0.3], dtype="float32"),
    )

    def fake_rerank(query, chunks, top_k):
        assert query == "test query"
        assert top_k == 5
        assert len(chunks) == 3
        return chunks[:top_k]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    results = retrieval.retrieve("test query")

    assert len(results) == 3
    assert all("id" in result for result in results)