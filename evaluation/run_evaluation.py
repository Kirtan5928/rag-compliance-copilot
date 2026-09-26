import sys
import json
import re
from pathlib import Path

# Add project root to Python import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from hybrid_retrieval import hybrid_search
from query import generate_answer


# Load environment variables from .env
load_dotenv()

# Paths
EVALUATION_DIR = Path(__file__).resolve().parent
DATASET = EVALUATION_DIR / "golden_questions.json"

RESULTS_DIR = EVALUATION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT = RESULTS_DIR / "latest_results.json"


ABSTENTION_MESSAGE = (
    "This information is not available in the provided report context."
)


def normalize(text):
    """
    Normalize text for answer comparison.
    Removes spaces, punctuation and capitalization differences.
    """
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def answer_matches(actual, expected):
    """
    Check whether the expected answer appears in the generated answer.

    For unanswerable questions, the answer must match
    the application's exact abstention message.
    """
    if expected is None:
        return actual.strip() == ABSTENTION_MESSAGE

    actual_normalized = normalize(actual)
    expected_normalized = normalize(expected)

    return (
        expected_normalized in actual_normalized
        or actual_normalized in expected_normalized
    )


def evidence_matches(retrieved, keywords):
    """
    Check whether all expected evidence keywords appear
    somewhere in the retrieved chunks.
    """
    if not keywords:
        return True

    blob = "\n".join(
        item.get("text", "")
        for item in retrieved
    ).lower()

    return all(
        keyword.lower() in blob
        for keyword in keywords
    )


def first_relevant_rank(retrieved, keywords):
    """
    Return the rank of the first retrieved chunk containing
    all expected evidence keywords.
    """
    if not keywords:
        return None

    for rank, item in enumerate(retrieved, start=1):
        text = item.get("text", "").lower()

        if all(
            keyword.lower() in text
            for keyword in keywords
        ):
            return rank

    return None


def run():
    print("Loading evaluation dataset...")

    golden = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    # golden_questions.json contains metadata + a questions list
    questions = golden["questions"]

    print(f"Dataset: {golden.get('dataset', 'Unknown')}")
    print(f"Questions: {len(questions)}")
    print("-" * 60)

    results = []

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        # Retrieve top-5 chunks
        retrieved = hybrid_search(
            question,
            n_results=5,
            alpha=0.5,
            document_id=None,
        )

        # Generate answer using retrieved context
        answer = generate_answer(
            question,
            retrieved
        )

        # Evaluate retrieval evidence
        evidence = evidence_matches(
            retrieved,
            item.get("evidence_keywords", [])
        )

        rank = first_relevant_rank(
            retrieved,
            item.get("evidence_keywords", [])
        )

        expected_answer = item.get("expected_answer")

        # Evaluate generated answer
        answer_correct = answer_matches(
            answer,
            expected_answer
        )

        # Evaluate abstention
        abstention_expected = not item["answerable"]

        abstention_correct = None

        if abstention_expected:
            abstention_correct = (
                answer.strip() == ABSTENTION_MESSAGE
            )

        # Store result
        results.append({
            "id": question_id,
            "category": item["category"],
            "question": question,

            "expected_answer": expected_answer,
            "actual_answer": answer,

            "answer_correct": answer_correct,

            "evidence_found": evidence,
            "relevant_rank": rank,

            "retrieved": [
                {
                    "chunk_id": x.get("chunk_id"),
                    "document_name": x.get("document_name"),
                    "page_start": x.get("page_start"),
                    "page_end": x.get("page_end"),
                    "score": round(
                        x.get("score", 0.0),
                        4
                    ),
                }
                for x in retrieved
            ],

            "abstention_expected": abstention_expected,
            "abstention_correct": abstention_correct,
        })

        print(
            f"    Answer correct: {answer_correct}"
        )

        print(
            f"    Evidence found: {evidence}"
        )

        print(
            f"    Relevant rank: {rank}"
        )

        print()

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    answerable = [
        r
        for r in results
        if r["expected_answer"] is not None
    ]

    unanswerable = [
        r
        for r in results
        if r["expected_answer"] is None
    ]

    # Retrieval Recall@5
    retrieval_hits = [
        r
        for r in answerable
        if r["evidence_found"]
    ]

    recall_at_5 = (
        len(retrieval_hits) / len(answerable)
        if answerable
        else 0.0
    )

    # Mean Reciprocal Rank
    reciprocal_ranks = [
        1.0 / r["relevant_rank"]
        for r in answerable
        if r["relevant_rank"] is not None
    ]

    mrr = (
        sum(reciprocal_ranks) / len(answerable)
        if answerable
        else 0.0
    )

    # Answer accuracy
    answer_accuracy = (
        sum(
            r["answer_correct"]
            for r in answerable
        ) / len(answerable)
        if answerable
        else 0.0
    )

    # Abstention accuracy
    abstention_accuracy = (
        sum(
            r["abstention_correct"]
            for r in unanswerable
        ) / len(unanswerable)
        if unanswerable
        else 0.0
    )

    summary = {
        "questions": len(results),

        "answerable_questions": len(answerable),

        "unanswerable_questions": len(unanswerable),

        "recall_at_5": round(
            recall_at_5,
            4
        ),

        "mrr": round(
            mrr,
            4
        ),

        "answer_accuracy": round(
            answer_accuracy,
            4
        ),

        "abstention_accuracy": round(
            abstention_accuracy,
            4
        ),
    }

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------

    payload = {
        "dataset": golden.get("dataset"),
        "source": golden.get("source"),
        "summary": summary,
        "results": results,
    }

    OUTPUT.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Print final summary
    # ---------------------------------------------------------

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print()
    print(
        f"Detailed results saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    run()
