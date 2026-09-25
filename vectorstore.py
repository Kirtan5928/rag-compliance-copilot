import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VECTOR_SIZE = 384
COLLECTION_NAME = "brsr_docs"
_client = None

def get_client():
    global _client
    if _client is None:
        if not QDRANT_URL:
            raise RuntimeError("QDRANT_URL is not configured.")
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client

def ensure_collection(collection_name=COLLECTION_NAME):
    client = get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))

def _document_filter(document_id):
    if not document_id:
        return None
    return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

def store_chunks(chunks, embeddings, document_id, document_name, collection_name=COLLECTION_NAME):
    ensure_collection(collection_name)
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk['chunk_index']}:{chunk['page_start']}:{chunk['text'][:80]}"))
        points.append(PointStruct(id=point_id, vector=embedding, payload={
            "chunk_id": f"{document_id}_p{chunk['page_start']}_c{chunk['chunk_index']}",
            "document_id": document_id, "document_name": document_name,
            "page_start": chunk["page_start"], "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"], "text": chunk["text"],
        }))
    get_client().upsert(collection_name=collection_name, points=points)

def get_all_chunks(collection_name=COLLECTION_NAME, document_id=None):
    client = get_client()
    if not client.collection_exists(collection_name):
        return []
    records, offset = [], None
    while True:
        results, offset = client.scroll(collection_name=collection_name, limit=100, offset=offset, with_payload=True, scroll_filter=_document_filter(document_id))
        for point in results:
            p = point.payload or {}
            records.append({k: p.get(k) for k in ["chunk_id","document_id","document_name","page_start","page_end","chunk_index","text"]})
        if offset is None:
            break
    return records

def query_similar(query_embedding, n_results=5, collection_name=COLLECTION_NAME, document_id=None):
    client = get_client()
    if not client.collection_exists(collection_name):
        return []
    results = client.query_points(collection_name=collection_name, query=query_embedding, query_filter=_document_filter(document_id), limit=n_results).points
    output = []
    for result in results:
        p = result.payload or {}
        output.append({k: p.get(k) for k in ["chunk_id","document_id","document_name","page_start","page_end","chunk_index","text"]} | {"score": result.score})
    return output

def delete_document(document_id, collection_name=COLLECTION_NAME):
    client = get_client()
    if not client.collection_exists(collection_name):
        return
    client.delete(collection_name=collection_name, points_selector=_document_filter(document_id))
