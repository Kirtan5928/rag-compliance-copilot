import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from hybrid_retrieval import hybrid_search
from query import generate_answer

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "golden_questions.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT = RESULTS_DIR / "latest_results.json"


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def answer_matches(actual, expected):
    if expected is None:
        return actual.strip() == "This information is not available in the provided report context."
    a = normalize(actual)
    e = normalize(expected)
    return e in a or a in e


def evidence_matches(retrieved, keywords):
    blob = "\n".join(item["text"] for item in retrieved).lower()
    return all(keyword.lower() in blob for keyword in keywords)


def first_relevant_rank(retrieved, keywords):
    for rank, item in enumerate(retrieved, start=1):
        text = item["text"].lower()
        if all(keyword.lower() in text for keyword in keywords):
            return rank
    return None


def run():
    golden = json.loads(DATASET.read_text(encoding="utf-8"))
    results = []

    for item in golden:
        retrieved = hybrid_search(
            item["question"],
            n_results=5,
            alpha=0.5,
            document_id=None,
        )

        answer = generate_answer(item["question"], retrieved)
        evidence = evidence_matches(retrieved, item.get("evidence_keywords", []))
        rank = first_relevant_rank(retrieved, item.get("evidence_keywords", []))

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item.get("expected_answer"),
            "actual_answer": answer,
            "answer_correct": answer_matches(answer, item.get("expected_answer")),
            "evidence_found": evidence,
            "relevant_rank": rank,
            "retrieved": [
                {
                    "chunk_id": x["chunk_id"],
                    "document_name": x["document_name"],
                    "page_start": x["page_start"],
                    "page_end": x["page_end"],
                    "score": round(x["score"], 4),
                }
                for x in retrieved
            ],
            "abstention_expected": not item["answerable"],
            "abstention_correct": (
                not item["answerable"]
                and answer.strip() == "This information is not available in the provided report context."
            ) if not item["answerable"] else None,
        })

    answerable = [r for r in results if r["expected_answer"] is not None]
    unanswerable = [r for r in results if r["expected_answer"] is None]
    retrieval_hits = [r for r in answerable if r["evidence_found"]]
    reciprocal_ranks = [
        1.0 / r["relevant_rank"] for r in answerable if r["relevant_rank"] is not None
    ]

    summary = {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "recall_at_5": round(len(retrieval_hits) / len(answerable), 4) if answerable else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(answerable), 4) if answerable else 0.0,
        "answer_accuracy": round(sum(r["answer_correct"] for r in answerable) / len(answerable), 4) if answerable else 0.0,
        "abstention_accuracy": round(sum(r["abstention_correct"] for r in unanswerable) / len(unanswerable), 4) if unanswerable else 0.0,
    }

    payload = {
        "summary": summary,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {OUTPUT}")


if __name__ == "__main__":
    run()
