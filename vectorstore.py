import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

VECTOR_SIZE = 384  # embedding dimension for all-MiniLM-L6-v2

_client = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def ensure_collection(collection_name="brsr_docs"):
    client = get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def store_chunks(chunk_ids, texts, embeddings, collection_name="brsr_docs"):
    """
    chunk_ids: list of string ids like 'chunk_0', 'chunk_1', ...
    texts: list of chunk text strings
    embeddings: list of embedding vectors
    """
    ensure_collection(collection_name)
    client = get_client()

    points = [
        PointStruct(
            id=i,  # Qdrant requires int or UUID point ids, not arbitrary strings
            vector=embeddings[i],
            payload={"chunk_id": chunk_ids[i], "text": texts[i]},
        )
        for i in range(len(chunk_ids))
    ]

    client.upsert(collection_name=collection_name, points=points)


def get_all_chunks(collection_name="brsr_docs"):
    """Returns (chunk_ids, texts) for every stored chunk -- needed for BM25 indexing."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return [], []

    all_points = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(results)
        if offset is None:
            break

    chunk_ids = [p.payload["chunk_id"] for p in all_points]
    texts = [p.payload["text"] for p in all_points]
    return chunk_ids, texts


def query_similar(query_embedding, n_results=5, collection_name="brsr_docs"):
    """Returns list of (chunk_id, text, similarity_score), highest similarity first."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return []

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=n_results,
    ).points

    return [(r.payload["chunk_id"], r.payload["text"], r.score) for r in results]