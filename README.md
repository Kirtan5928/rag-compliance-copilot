# RAG Compliance Copilot

A retrieval-augmented generation system for querying SEBI BRSR/ESG reports with grounded, source-backed answers. The system is designed for compliance-style questions where exact figures, reporting years, table rows, and traceable evidence matter.

## Current status

**Core implementation complete. Final evaluation and presentation/documentation are the remaining project tasks.**

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
9. Returns the source chunks with document and page metadata.
10. Supports document-level filtering and deletion.

The retrieval design follows the standard hybrid-search pattern of combining semantic and lexical signals before a later ranking stage. RRF is used because dense and BM25 scores are on different scales and should not be directly added. 

## Architecture

```text
                         INGESTION
PDF
 │
 ▼
Page-aware PDF extraction
 │
 ▼
Page-aware chunking
 │
 ├──────────────► text + page metadata
 │
 ▼
MiniLM embeddings
 │
 ▼
Qdrant
 │
 └── document_id / page / chunk metadata


                          QUERY
User question
 │
 ├──────────────► Dense retrieval
 │
 └──────────────► BM25 lexical retrieval
                         │
                         ▼
                 Reciprocal Rank Fusion
                         │
                         ▼
              Structured query reranking
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

This allows answers to be traced back to the report rather than presenting unsupported LLM output.

### Hybrid retrieval

Dense retrieval helps with semantic/paraphrased questions, while BM25 helps with exact terminology, table labels, identifiers, and numerical phrases. RRF combines the ranked lists without comparing their raw score scales.

### Structured reranking

BRSR reports contain structured tables where lexical evidence can be more informative than general semantic similarity. A small deterministic boost is applied only to percentage questions that explicitly target categories such as the Board of Directors or Key Management Personnel and where the candidate contains the relevant category, gender term, and percentage marker.

### Grounded generation and abstention

The generation prompt instructs the LLM to:

- use only supplied report context
- preserve numerical values exactly
- avoid inventing missing information
- respect the requested reporting year/table row
- return a fixed abstention message when evidence is insufficient

## Evaluation

The project contains a 20-question golden dataset based on the RIL BRSR FY 2024-25 report:

- **18 answerable questions**
- **2 unanswerable questions**
- energy, water, emissions, waste, workforce, and abstention cases
- numerical extraction and year-disambiguation cases
- evidence-keyword validation

The evaluation reports:

- **Recall@5** — whether required evidence appears in the five chunks passed to the generator
- **MRR** — how highly the first relevant chunk is ranked
- **Answer accuracy** — whether the generated answer matches the expected value
- **Abstention accuracy** — whether unsupported questions are rejected correctly

These metrics separate retrieval quality from final answer quality, which is important when diagnosing RAG failures. citeturn0search1turn0search2

Run the full evaluation with:

```powershell
python evaluation\run_evaluation.py
```

The full evaluation invokes the configured LLM, so it should be run deliberately rather than repeatedly during development.

A retrieval-only benchmark is also provided in `evaluation/benchmark_retrieval.py` and does not call the LLM.

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
- 25 MB upload limit
- corrupt/unreadable PDFs
- PDFs without extractable text
- PDFs that produce no searchable chunks
- empty/invalid questions
- retrieval failures
- answer-generation failures

The frontend mirrors important upload restrictions and surfaces API errors to the user.

## Project roadmap

- [x] **M1 — Core RAG:** PDF ingestion, chunking, embeddings, vector retrieval, grounded generation
- [x] **M2 — Evidence grounding:** page-aware sources, document IDs, document filtering, abstention
- [x] **M3 — Retrieval:** BM25 + dense hybrid search, RRF, structured reranking
- [x] **M4 — Application:** FastAPI backend, React frontend, deployment
- [x] **M5 — Hardening:** validation, error handling, upload limits, deletion UX, CI checks
- [ ] **M6 — Final evaluation:** complete post-fix golden-set evaluation and retrieval benchmark
- [ ] **M7 — Finalization:** results tables, architecture diagrams, screenshots, README/report, presentation and viva preparation

## Limitations

- The current system is optimized for text-based PDFs; scanned/image-only PDFs are rejected.
- The golden dataset is currently based on one BRSR report, so broader multi-company generalization still requires additional evaluation.
- The LLM is not trusted as an independent source of facts; its output is constrained by retrieved report context.
- Retrieval scores are ranking signals rather than probabilities of answer correctness.

## Research direction

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

The final report should use the fixed golden set to quantify how each retrieval strategy affects Recall@5 and MRR, while the full RAG evaluation separately measures answer and abstention accuracy.
