import os
import requests
from dotenv import load_dotenv

load_dotenv()

# "api" uses Hugging Face Inference API and is the deployment default.
# "local" requires requirements-local.txt and is intended for development.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "api")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
HF_TOKEN = os.getenv("HF_TOKEN")
_local_embedder = None

def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_embedder

def embed_texts(texts):
    if EMBEDDING_PROVIDER == "api":
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is not configured.")
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    return _get_local_embedder().encode(texts).tolist()
