import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import load_pdf_text, chunk_text, build_vector_store
from hybrid_retrieval import hybrid_search
from query import generate_answer

app = FastAPI(title="RAG Compliance Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    hybrid_threshold: float = 0.8


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, extracts text, chunks it, embeds it,
    and stores it in the vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_pdf_text(save_path)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    chunks = chunk_text(text)
    build_vector_store(chunks)

    return {
        "filename": file.filename,
        "characters_extracted": len(text),
        "chunks_stored": len(chunks)
    }


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Runs hybrid retrieval + grounded generation for a given question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    all_retrieved = hybrid_search(request.question, n_results=5, alpha=0.5)
    retrieved = [r for r in all_retrieved if r[2] >= request.hybrid_threshold]

    if not retrieved:
        retrieved = all_retrieved[:1]  # fallback: single best match

    answer = generate_answer(request.question, retrieved)

    sources = [
        SourceChunk(chunk_id=cid, text=text, score=round(score, 3))
        for cid, text, score in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources)