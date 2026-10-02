import json
from pathlib import Path

from src.retrieve import retrieve_qdrant_stages, retrieve_stages

K_VALUES = [1, 3, 5, 10]


K_VALUES = [1, 3, 5, 10]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(ranked_results: list[dict], relevant_ids: set[str]) -> dict:
    ranked_ids = [str(result["id"]) for result in ranked_results]
    ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, start=1)}
    relevant_ranks = [ranks[chunk_id] for chunk_id in relevant_ids if chunk_id in ranks]
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None

    metrics = {
        "first_relevant_rank": first_relevant_rank,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    }

    for k in K_VALUES:
        retrieved_ids = set(ranked_ids[:k])
        hits = len(relevant_ids & retrieved_ids)
        metrics[f"hit@{k}"] = 1 if hits > 0 else 0
        metrics[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0

    return metrics


def qdrant_results_to_dicts(results: list) -> list[dict]:
    return [
        {
            "id": str(result.id),
            "text": result.payload.get("text", ""),
        }
        for result in results
    ]


def evaluate_question(question: dict, max_k: int = 10) -> dict:
    relevant_ids = {str(chunk_id) for chunk_id in question["relevant_chunk_ids"]}

    faiss_stages = retrieve_stages(
        query=question["question"],
        max_k=max_k,
    )

    qdrant_stages = retrieve_qdrant_stages(
        query=question["question"],
        k=max_k,
    )

    return {
        "id": question["id"],
        "question": question["question"],
        "faiss": {
            method: calculate_metrics(faiss_stages[method], relevant_ids)
            for method in ["dense", "sparse", "rrf", "reranked"]
        },
        "qdrant": {
            method: calculate_metrics(qdrant_stages[method], relevant_ids)
            for method in ["dense", "sparse", "rrf", "reranked"]
        },
    }


def average_metric(results: list[dict], system: str, stage: str, metric: str) -> float:
    return sum(result[system][stage][metric] for result in results) / len(results)


def print_summary(results: list[dict]) -> None:
    header = (
        f"{'System / Stage':<24}"
        f"{'Hit@1':>10}"
        f"{'Recall@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Recall@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>10}"
        f"{'Hit@10':>10}"
        f"{'Recall@10':>10}"
        f"{'MRR':>10}"
    )

    print(f"\nEvaluated {len(results)} questions\n")
    print(header)
    print("-" * len(header))

    for system in ["faiss", "qdrant"]:
        for stage in ["dense", "sparse", "rrf", "reranked"]:
            label = f"{system} / {stage}"

            print(
                f"{label:<24}"
                f"{average_metric(results, system, stage, 'hit@1'):>10.3f}"
                f"{average_metric(results, system, stage, 'recall@1'):>10.3f}"
                f"{average_metric(results, system, stage, 'hit@3'):>10.3f}"
                f"{average_metric(results, system, stage, 'recall@3'):>10.3f}"
                f"{average_metric(results, system, stage, 'hit@5'):>10.3f}"
                f"{average_metric(results, system, stage, 'recall@5'):>10.3f}"
                f"{average_metric(results, system, stage, 'hit@10'):>10.3f}"
                f"{average_metric(results, system, stage, 'recall@10'):>10.3f}"
                f"{average_metric(results, system, stage, 'mrr'):>10.3f}"
            )

def print_disagreements(results: list[dict]) -> None:
    print("\nQUESTIONS WHERE FINAL FAISS AND QDRANT RESULTS DIFFER")
    print("=" * 80)

    for result in results:
        faiss = result["faiss"]["reranked"]
        qdrant = result["qdrant"]["reranked"]

        if (
            faiss["first_relevant_rank"] != qdrant["first_relevant_rank"]
            or faiss["recall@10"] != qdrant["recall@10"]
        ):
            print(f"\n{result['id']}: {result['question']}")
            print(
                f"  FAISS   rank={faiss['first_relevant_rank']}, "
                f"Recall@10={faiss['recall@10']:.3f}"
            )
            print(
                f"  Qdrant  rank={qdrant['first_relevant_rank']}, "
                f"Recall@10={qdrant['recall@10']:.3f}"
            )


def main() -> None:
    questions = load_questions(Path("evaluation/questions.json"))

    results = [
        evaluate_question(question)
        for question in questions
    ]

    print_summary(results)
    print_disagreements(results)


if __name__ == "__main__":
    main()