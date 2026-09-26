# System Architecture

## End-to-end flow

```text
PDF upload
   ↓
PDF validation
   ↓
Page-aware text extraction
   ↓
Page-aware chunking
   ↓
Dense embeddings
   ↓
Qdrant storage
   ↓
User question
   ├── Dense semantic retrieval
   └── BM25 lexical retrieval
             ↓
      Reciprocal Rank Fusion
             ↓
      Structured reranking
             ↓
           Top-5
             ↓
      Grounded LLM generation
             ↓
      Answer + source chunks
```

## Ingestion

`ingest.py` extracts `(page_number, text)` pairs and creates chunks while preserving page boundaries. Each stored chunk carries document and page metadata.

## Retrieval

`hybrid_retrieval.py` runs dense retrieval and BM25 over the same corpus. RRF combines their ranked lists without directly adding incompatible score scales.

For selected structured percentage queries, a small deterministic boost favors chunks containing the requested category, gender term, and percentage marker together.

## Storage

`vectorstore.py` stores embeddings and metadata in the Qdrant `brsr_docs` collection. A keyword payload index on `document_id` supports document-specific filtering.

## Generation

`query.py` sends only retrieved evidence to the configured LLM. The generation prompt requires exact numerical preservation, correct year/table selection, and abstention when the evidence does not support the answer.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /documents` | List indexed documents |
| `POST /ingest` | Validate, parse, chunk, embed, and store a PDF |
| `DELETE /documents/{document_id}` | Delete a document and its indexed chunks |
| `POST /query` | Retrieve evidence and generate a grounded answer |

## Frontend

The React/Vite frontend provides PDF upload, document selection, deletion, question submission, answer display, and source display. It defaults to the local backend at `http://localhost:8000` and can use `VITE_API_URL` for deployment.

## Deployment

```text
Browser → Vercel frontend → Render FastAPI backend
                               ├── Qdrant Cloud
                               ├── Hugging Face embeddings
                               └── Groq LLM
```
