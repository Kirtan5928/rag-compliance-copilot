# RAG Compliance Copilot

A retrieval-augmented generation system for querying SEBI BRSR/ESG reports with grounded, source-backed answers. The system is designed for compliance-style questions where exact figures, reporting years, table rows, and traceable evidence matter.

## Current status

**Core implementation complete. Project packaging and final evaluation are in progress.**

### What the system does

1. Uploads a BRSR/ESG PDF.
2. Extracts text with page numbers preserved.
3. Splits the report into page-aware searchable chunks.
4. Generates dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores chunks and metadata in Qdrant.
6. Retrieves candidates with:
   - dense semantic retrieval
   - BM25 lexical retrieval
   - Reciprocal Rank Fusion (RRF)
   - targeted structured reranking for table-heavy percentage questions
7. Sends only retrieved evidence to the LLM.
8. Generates a grounded answer with an explicit abstention response when the report does not support the requested information.
9. Returns source chunks with document and page metadata.
10. Supports document-level filtering and deletion.

## Architecture

```text
                         INGESTION
PDF
 │
 ▼
PDF validation
 │
 ▼
Page-aware extraction
 │
 ▼
Page-aware chunking ───────────────► text + page metadata
 │
 ▼
MiniLM embeddings
 │
 ▼
Qdrant vector store


                          QUERY
User question
 │
 ├──────────────► Dense retrieval ───┐
 │                                   │
 └──────────────► BM25 retrieval ────┤
                                     ▼
                            Reciprocal Rank Fusion
                                     │
                                     ▼
                         Targeted structured reranking
                                     │
                                     ▼
                              Top-5 evidence
                                     │
                                     ▼
                            Grounded LLM generation
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
                   Supported answer          Abstention
                         │
                         ▼
                  Answer + page sources
```

## Key engineering features

### Page-aware evidence

Each chunk retains:

- `document_id`
- `document_name`
- `page_start`
- `page_end`
- `chunk_id`
- source text

This makes generated answers auditable against the original report.

### Hybrid retrieval

Dense retrieval captures semantic similarity and paraphrased questions. BM25 captures exact terminology, table labels, identifiers, and numerical expressions. RRF combines ranked lists without assuming that dense and BM25 raw scores share the same scale.

### Structured reranking

BRSR reports contain structured tables where exact category and percentage evidence can matter more than broad semantic similarity. A narrow deterministic boost is applied only to specific percentage questions targeting female representation in categories such as Board of Directors or Key Management Personnel.

### Grounded generation and abstention

The generation prompt instructs the LLM to:

- use only supplied report context;
- preserve numerical values exactly;
- respect the requested reporting year and table row;
- avoid unsupported inference;
- return a fixed abstention response when the context is insufficient.

## Evaluation

The project contains a fixed 20-question golden dataset based on the RIL BRSR FY 2024-25 report:

- **18 answerable questions**
- **2 unanswerable questions**
- numerical extraction and table-oriented cases
- reporting-year disambiguation
- evidence-keyword validation
- explicit abstention cases

### Retrieval benchmark

| Retrieval strategy | Recall@5 | MRR |
|---|---:|---:|
| Dense | 77.78% | 72.22% |
| BM25 | 94.44% | 83.52% |
| Hybrid + structured reranking | **100.00%** | 82.13% |

These are retrieval-only measurements. The hybrid configuration achieved complete Top-5 evidence coverage on this benchmark, while BM25 had slightly higher MRR. The two metrics capture different properties.

### Previous end-to-end baseline

A previous clean end-to-end run before the final structured-query fix recorded:

| Metric | Result |
|---|---:|
| Recall@5 | 94.44% |
| MRR | 79.35% |
| Answer accuracy | 94.44% |
| Abstention accuracy | 100.00% |

The final post-fix 20-question LLM evaluation remains pending because the available LLM quota was exhausted during the previous attempt. No final post-fix generation metric is claimed until that run completes.

Run it with:

```powershell
python evaluation\run_evaluation.py
```

The retrieval-only benchmark can be run repeatedly without calling the LLM:

```powershell
python evaluation\benchmark_retrieval.py
```

## Documentation

- [System architecture](docs/ARCHITECTURE.md)
- [End-to-end workflow](docs/WORKFLOW.md)
- [Retrieval pipeline](docs/RETRIEVAL_PIPELINE.md)
- [Project report draft](docs/PROJECT_REPORT.md)
- [Evaluation methodology](docs/EVALUATION.md)

## Tech stack

### Backend

- Python
- FastAPI
- pypdf
- Qdrant
- sentence-transformers / Hugging Face Inference API
- rank-bm25
- Groq

### Frontend

- React
- Vite
- JavaScript
- CSS

### Deployment

- Backend: Render
- Frontend: Vercel
- Vector database: Qdrant Cloud
- LLM: Groq

## Local development

### Backend

```powershell
cd rag-compliance-copilot
python -m compileall -q .
uvicorn main:app --reload
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### Frontend

```powershell
cd rag-frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The frontend defaults to `http://localhost:8000` when `VITE_API_URL` is not configured.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health |
| `/documents` | GET | List indexed documents |
| `/ingest` | POST | Upload and index a PDF |
| `/documents/{document_id}` | DELETE | Delete a document and its chunks |
| `/query` | POST | Retrieve evidence and generate a grounded answer |

## Defensive handling

The backend validates:

- PDF file type
- **100 MB upload limit**
- corrupt/unreadable PDFs
- PDFs without extractable text
- PDFs that produce no searchable chunks
- empty/invalid questions
- retrieval failures
- answer-generation failures

The frontend mirrors the upload restriction and surfaces API errors to the user.

## Project roadmap

- [x] **M1 — Core RAG:** PDF ingestion, chunking, embeddings, vector retrieval, grounded generation
- [x] **M2 — Evidence grounding:** page-aware sources, document IDs, filtering, abstention
- [x] **M3 — Retrieval:** BM25 + dense hybrid search, RRF, structured reranking
- [x] **M4 — Application:** FastAPI backend, React frontend, deployment
- [x] **M5 — Hardening:** validation, error handling, 100 MB upload limit, deletion UX, CI checks
- [ ] **M6 — Final evaluation:** complete post-fix golden-set evaluation
- [ ] **M7 — Finalization:** diagrams, screenshots, final report, presentation and viva preparation

## Limitations

- The current extraction path is text-based; scanned/image-only PDFs are rejected.
- The golden dataset currently represents one BRSR report, so broader multi-company evaluation is still required.
- The LLM is not treated as an independent source of facts.
- Retrieval scores are ranking signals, not probabilities of answer correctness.
- Full end-to-end evaluation depends on available LLM quota.

## Research contribution

The main experimentally testable contribution is the comparison of retrieval strategies on structured compliance reports:

```text
Dense retrieval
      vs
BM25 retrieval
      vs
Hybrid RRF
      vs
Hybrid RRF + targeted structured reranking
```

The final report should quantify how these strategies affect Recall@5 and MRR and separately report final answer and abstention accuracy.
