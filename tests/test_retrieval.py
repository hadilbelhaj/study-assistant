from src.index import load_index
from src.retrieve import retrieve_dense, retrieve_sparse
from src.retrieve import reciprocal_rank_fusion


index, chunks, bm25 = load_index()

query = "Quelles sont les principales collections en Java ?"

dense_results = retrieve_dense(
    query,
    index,
    chunks,
    k=20,
)

sparse_results = retrieve_sparse(
    query,
    bm25,
    chunks,
    k=20,
)

fused_results = reciprocal_rank_fusion(
    [dense_results, sparse_results]
)

print("\n===== DENSE =====")
for rank, result in enumerate(dense_results[:10], start=1):
    print(
        rank,
        result["id"],
        result.get("dense_score"),
        result["text"][:100].replace("\n", " ")
    )

print("\n===== SPARSE =====")
for rank, result in enumerate(sparse_results[:10], start=1):
    print(
        rank,
        result["id"],
        result.get("sparse_score"),
        result["text"][:100].replace("\n", " ")
    )

print("\n===== RRF =====")
for rank, result in enumerate(fused_results[:10], start=1):
    print(
        rank,
        result["id"],
        result["rrf_score"],
        result["text"][:100].replace("\n", " ")
    )