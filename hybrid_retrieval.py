import chromadb
from embeddings import embed_texts
from rank_bm25 import BM25Okapi

def load_all_chunks(collection_name="brsr_docs"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    all_data = collection.get()
    return all_data["ids"], all_data["documents"]

def build_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def normalize_scores(scores):
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

def hybrid_search(question, n_results=5, alpha=0.5, collection_name="brsr_docs"):
    ids, chunks = load_all_chunks(collection_name)

    # ---- Vector search scores ----
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    question_embedding = embed_texts([question])

    vector_results = collection.query(query_embeddings=question_embedding, n_results=len(chunks))
    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_similarities = [1 - d for d in vector_distances]

    vector_score_map = dict(zip(vector_ids, normalize_scores(vector_similarities)))

    # ---- BM25 keyword scores ----
    bm25 = build_bm25_index(chunks)
    tokenized_question = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_question)
    bm25_score_map = dict(zip(ids, normalize_scores(list(bm25_scores))))

    # ---- Combine scores ----
    combined = []
    for chunk_id, chunk_text in zip(ids, chunks):
        v_score = vector_score_map.get(chunk_id, 0)
        b_score = bm25_score_map.get(chunk_id, 0)
        final_score = alpha * v_score + (1 - alpha) * b_score
        combined.append((chunk_id, chunk_text, final_score))

    combined.sort(key=lambda x: x[2], reverse=True)

    return combined[:n_results]