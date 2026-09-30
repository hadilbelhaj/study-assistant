import json
from pathlib import Path

from src.retrieve import retrieve_stages

K_VALUES = [1, 3, 5, 10]
METHODS = ["dense", "sparse", "rrf", "reranked"]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(ranked_results: list[dict], relevant_ids: set[str]) -> dict:
    ranked_ids = [result["id"] for result in ranked_results]
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


def evaluate_question(question: dict) -> dict:
    stages = retrieve_stages(query=question["question"], max_k=max(K_VALUES))
    relevant_ids = set(question["relevant_chunk_ids"])

    return {
        "id": question["id"],
        "question": question["question"],
        "metrics": {method: calculate_metrics(stages[method], relevant_ids) for method in METHODS},
    }


def average_metric(results: list[dict], method: str, metric: str) -> float:
    return sum(result["metrics"][method][metric] for result in results) / len(results)


def print_summary(results: list[dict]) -> None:
    print(f"\nEvaluated {len(results)} questions\n")

    header = (
        f"{'Method':<12}"
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

    print(header)
    print("-" * len(header))

    for method in METHODS:
        print(
            f"{method:<12}"
            f"{average_metric(results, method, 'hit@1'):>10.3f}"
            f"{average_metric(results, method, 'recall@1'):>10.3f}"
            f"{average_metric(results, method, 'hit@3'):>10.3f}"
            f"{average_metric(results, method, 'recall@3'):>10.3f}"
            f"{average_metric(results, method, 'hit@5'):>10.3f}"
            f"{average_metric(results, method, 'recall@5'):>10.3f}"
            f"{average_metric(results, method, 'hit@10'):>10.3f}"
            f"{average_metric(results, method, 'recall@10'):>10.3f}"
            f"{average_metric(results, method, 'mrr'):>10.3f}"
        )


def print_failures(results: list[dict]) -> None:
    print("\nQUESTIONS WHERE FINAL RERANKED RETRIEVAL MISSED ALL RELEVANT CHUNKS")
    print("=" * 80)

    for result in results:
        metrics = result["metrics"]["reranked"]
        if metrics["hit@10"] == 0:
            print(f"\n{result['id']}: {result['question']}")
            for method in METHODS:
                method_metrics = result["metrics"][method]
                print(
                    f"  {method:<10} "
                    f"first_rank={method_metrics['first_relevant_rank']}, "
                    f"Recall@10={method_metrics['recall@10']:.3f}"
                )


def main() -> None:
    questions = load_questions(Path("evaluation/questions.json"))
    results = [evaluate_question(question) for question in questions]

    print_summary(results)
    print_failures(results)


if __name__ == "__main__":
    main()