import os
import shutil
import uuid
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ingest import extract_pdf_pages, chunk_pages, build_vector_store
from hybrid_retrieval import hybrid_search
from query import generate_answer
from vectorstore import get_all_chunks, delete_document

app = FastAPI(title="RAG Compliance Copilot API", version="2.0.0")
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in frontend_url.split(",") if x.strip()], allow_methods=["*"], allow_headers=["*"])
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    hybrid_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    document_id: str | None = None

class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-compliance-copilot"}

@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    document_id = uuid.uuid4().hex
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(DATA_DIR, f"{document_id}_{safe_name}")
    try:
        with open(save_path, "wb") as output:
            shutil.copyfileobj(file.file, output)
        pages = extract_pdf_pages(save_path)
        if not pages:
            raise HTTPException(status_code=422, detail="No extractable text found in PDF.")
        chunks = chunk_pages(pages)
        build_vector_store(chunks, document_id=document_id, document_name=safe_name)
        return {"document_id": document_id, "filename": safe_name, "pages_extracted": len(pages), "characters_extracted": sum(len(t) for _, t in pages), "chunks_stored": len(chunks)}
    finally:
        if os.path.exists(save_path): os.remove(save_path)

@app.get("/documents")
def list_documents():
    grouped = defaultdict(lambda: {"document_id": "", "document_name": "", "pages": set(), "chunks": 0})
    for record in get_all_chunks():
        item = grouped[record["document_id"]]
        item["document_id"] = record["document_id"]
        item["document_name"] = record["document_name"]
        item["chunks"] += 1
        if record["page_start"]: item["pages"].add(record["page_start"])
    return [{"document_id": x["document_id"], "document_name": x["document_name"], "pages": len(x["pages"]), "chunks": x["chunks"]} for x in grouped.values()]

@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}

@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    question = request.question.strip()
    if not question: raise HTTPException(status_code=400, detail="Question cannot be empty.")
    all_retrieved = hybrid_search(question, n_results=5, alpha=0.5, document_id=request.document_id)
    retrieved = [item for item in all_retrieved if item["score"] >= request.hybrid_threshold]
    answer = generate_answer(question, retrieved)
    sources = [SourceChunk(**{**item, "score": round(item["score"], 3)}) for item in retrieved]
    return QueryResponse(answer=answer, sources=sources)
