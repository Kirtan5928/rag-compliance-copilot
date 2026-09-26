import json
from pathlib import Path

from embeddings import embed_texts
from hybrid_retrieval import hybrid_search, build_bm25_index, expand_lexical_query
from vectorstore import get_all_chunks, query_similar


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).resolve().parent / "golden_questions.json"
OUTPUT = Path(__file__).resolve().parent / "results" / "retrieval_benchmark.json"


def evidence_matches(retrieved, keywords):
    if not keywords:
        return True
    blob = "\n".join(item.get("text", "") for item in retrieved).lower()
    return all(keyword.lower() in blob for keyword in keywords)


def relevant_rank(retrieved, keywords):
    if not keywords:
        return None
    for rank, item in enumerate(retrieved, start=1):
        text = item.get("text", "").lower()
        if all(keyword.lower() in text for keyword in keywords):
            return rank
    return None


def evaluate_run(results):
    answerable = [item for item in results if item["answerable"]]
    hits = [item for item in answerable if item["evidence_found"]]
    reciprocal_ranks = [
        1.0 / item["relevant_rank"]
        for item in answerable
        if item["relevant_rank"] is not None
    ]

    return {
        "recall_at_5": round(len(hits) / len(answerable), 4) if answerable else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(answerable), 4) if answerable else 0.0,
    }


def retrieve_dense(question, records):
    return query_similar(
        embed_texts([question])[0],
        n_results=5,
        document_id=None,
    )


def retrieve_bm25(question, records):
    index = build_bm25_index(records)
    scores = index.get_scores(expand_lexical_query(question))
    order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)[:5]
    return [{**records[i], "score": float(scores[i])} for i in order]


def main():
    golden = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = [q for q in golden["questions"] if q["answerable"]]
    records = get_all_chunks()

    if not records:
        raise RuntimeError("No indexed chunks found.")

    runs = {
        "dense": [],
        "bm25": [],
        "hybrid_rrf_structured": [],
    }

    for item in questions:
        question = item["question"]
        keywords = item.get("evidence_keywords", [])

        retrievers = {
            "dense": retrieve_dense(question, records),
            "bm25": retrieve_bm25(question, records),
            "hybrid_rrf_structured": hybrid_search(
                question,
                n_results=5,
                alpha=0.5,
                document_id=None,
            ),
        }

        for name, retrieved in retrievers.items():
            runs[name].append({
                "id": item["id"],
                "question": question,
                "evidence_found": evidence_matches(retrieved, keywords),
                "relevant_rank": relevant_rank(retrieved, keywords),
                "retrieved": [
                    {
                        "chunk_id": x.get("chunk_id"),
                        "page_start": x.get("page_start"),
                        "score": round(float(x.get("score", 0.0)), 6),
                    }
                    for x in retrieved
                ],
            })

    summary = {
        name: evaluate_run(results)
        for name, results in runs.items()
    }

    payload = {
        "dataset": golden.get("dataset"),
        "questions": len(questions),
        "summary": summary,
        "runs": runs,
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Detailed results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
