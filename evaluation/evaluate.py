import json
from pathlib import Path

from src.retrieve import retrieve

K_VALUES = [1, 3, 5, 10]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_question(question: dict, max_k: int = 10) -> dict:
    results = retrieve(query=question["question"], k=max_k)
    relevant_ids = set(question["relevant_chunk_ids"])
    ranked_ids = [result["id"] for result in results]

    ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, start=1)}
    relevant_ranks = [ranks[chunk_id] for chunk_id in relevant_ids if chunk_id in ranks]
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None

    metrics = {
        "id": question["id"],
        "first_relevant_rank": first_relevant_rank,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    }

    for k in K_VALUES:
        retrieved_ids = set(ranked_ids[:k])
        hits = len(relevant_ids & retrieved_ids)
        metrics[f"hit@{k}"] = 1 if hits > 0 else 0
        metrics[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0

    return metrics


def main() -> None:
    questions = load_questions(Path("evaluation/questions.json"))
    results = [evaluate_question(question) for question in questions]

    print(f"Evaluated {len(results)} questions\n")

    for question, result in zip(questions, results):
        print(f"{'=' * 80}")
        print(f"{result['id']}: {question['question']}")
        print(f"First relevant rank: {result['first_relevant_rank']}")
        print(f"Hit@1: {result['hit@1']}")
        print(f"Hit@3: {result['hit@3']}")
        print(f"Hit@5: {result['hit@5']}")
        print(f"Hit@10: {result['hit@10']}")
        print(f"Recall@1: {result['recall@1']:.3f}")
        print(f"Recall@3: {result['recall@3']:.3f}")
        print(f"Recall@5: {result['recall@5']:.3f}")
        print(f"Recall@10: {result['recall@10']:.3f}")
        print(f"MRR: {result['mrr']:.3f}")

    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")

    for k in K_VALUES:
        hit = sum(result[f"hit@{k}"] for result in results) / len(results)
        recall = sum(result[f"recall@{k}"] for result in results) / len(results)
        print(f"Hit@{k}:    {hit:.3f}")
        print(f"Recall@{k}: {recall:.3f}")

    mrr = sum(result["mrr"] for result in results) / len(results)
    print(f"MRR:        {mrr:.3f}")


if __name__ == "__main__":
    main()