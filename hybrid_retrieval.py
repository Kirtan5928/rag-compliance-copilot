from embeddings import embed_texts
from vectorstore import get_all_chunks, query_similar
from rank_bm25 import BM25Okapi

# BRSR/ESG reports often use formal variants such as "females" and
# "percentage of females" while users ask with "female" or "percent".
# Expanding only the lexical query improves exact table matching without
# changing the semantic embedding query.
QUERY_ALIASES = {
    "female": ["females", "women"],
    "females": ["female", "women"],
    "woman": ["women", "female", "females"],
    "women": ["female", "females"],
    "percent": ["percentage", "%"],
    "percentage": ["percent", "%"],
    "director": ["directors"],
    "directors": ["director"],
}

def tokenize(text):
    return text.lower().split()

def expand_lexical_query(question):
    tokens = tokenize(question)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_ALIASES.get(token.strip(".,:;()"), []))
    return expanded

def build_bm25_index(records):
    return BM25Okapi([tokenize(record["text"]) for record in records])

def reciprocal_rank_fusion(dense, bm25_scores, records, alpha=0.5, k=60):
    dense_rank = {
        item["chunk_id"]: rank
        for rank, item in enumerate(dense, start=1)
    }

    bm25_order = sorted(
        range(len(records)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )
    bm25_rank = {
        records[index]["chunk_id"]: rank
        for rank, index in enumerate(bm25_order, start=1)
    }

    combined = []
    for record in records:
        chunk_id = record["chunk_id"]
        score = 0.0

        if chunk_id in dense_rank:
            score += alpha / (k + dense_rank[chunk_id])
        if chunk_id in bm25_rank:
            score += (1 - alpha) / (k + bm25_rank[chunk_id])

        combined.append({**record, "score": score})

    combined.sort(key=lambda item: item["score"], reverse=True)
    return combined

def hybrid_search(question, n_results=5, alpha=0.5, collection_name="brsr_docs", document_id=None):
    records = get_all_chunks(collection_name, document_id)
    if not records:
        return []

    # Dense retrieval preserves the user's original natural-language intent.
    dense = query_similar(
        embed_texts([question])[0],
        n_results=len(records),
        collection_name=collection_name,
        document_id=document_id,
    )

    # BM25 gets a small lexical expansion to improve table/header matching.
    bm25 = build_bm25_index(records)
    bm25_scores = bm25.get_scores(expand_lexical_query(question))

    # Rank-based fusion avoids mixing incomparable dense/BM25 score scales.
    # alpha=0.5 keeps both retrieval signals equally weighted.
    combined = reciprocal_rank_fusion(
        dense,
        bm25_scores,
        records,
        alpha=alpha,
    )
    return combined[:n_results]
