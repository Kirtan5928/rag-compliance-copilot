from embeddings import embed_texts
from vectorstore import get_all_chunks, query_similar
from rank_bm25 import BM25Okapi

def build_bm25_index(records):
    return BM25Okapi([record["text"].lower().split() for record in records])

def normalize_scores(scores):
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def hybrid_search(question, n_results=5, alpha=0.5, collection_name="brsr_docs", document_id=None):
    records = get_all_chunks(collection_name, document_id)
    if not records:
        return []

    dense = query_similar(embed_texts([question])[0], n_results=len(records), collection_name=collection_name, document_id=document_id)
    dense_scores = normalize_scores([item["score"] for item in dense])
    dense_map = {item["chunk_id"]: score for item, score in zip(dense, dense_scores)}

    bm25_scores = normalize_scores(list(build_bm25_index(records).get_scores(question.lower().split())))
    bm25_map = {item["chunk_id"]: score for item, score in zip(records, bm25_scores)}

    combined = []
    for record in records:
        score = alpha * dense_map.get(record["chunk_id"], 0.0) + (1 - alpha) * bm25_map.get(record["chunk_id"], 0.0)
        combined.append({**record, "score": score})
    combined.sort(key=lambda item: item["score"], reverse=True)
    return combined[:n_results]
