import os
import requests
from dotenv import load_dotenv

load_dotenv()

# "local" = sentence-transformers running on this machine (needs torch, heavy RAM).
# "api"   = Hugging Face Inference API (no torch loaded locally, used for deployment).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

# Only created if/when the local provider is actually used -- this keeps
# torch from ever being imported (and loaded into memory) on deployments
# that use the API provider.
_local_embedder = None


def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_embedder


def embed_texts(texts):
    """
    Takes a list of strings, returns a list of embedding vectors.
    Works identically regardless of which provider is active.
    """
    if EMBEDDING_PROVIDER == "api":
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    else:
        embedder = _get_local_embedder()
        return embedder.encode(texts).tolist()