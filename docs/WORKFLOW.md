# End-to-End Workflow

## 1. User uploads a report

The user selects a PDF from the React frontend. The application validates the file type and enforces a 100 MB maximum upload size before sending it to the FastAPI backend.

## 2. Backend validates and parses the PDF

The `/ingest` endpoint validates the upload and uses pypdf to extract text page by page.

The pipeline rejects:

- non-PDF uploads;
- corrupted/unreadable PDFs;
- PDFs with no extractable text;
- PDFs that produce no searchable chunks.

## 3. Page-aware chunking

Text is split into searchable chunks while retaining page boundaries and document identity.

Every stored chunk carries:

- document ID;
- document name;
- starting page;
- ending page;
- chunk ID;
- source text.

This metadata is critical because the application must return evidence that can be traced to the report.

## 4. Embedding and storage

Each chunk is converted into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`.

Vectors and metadata are stored in Qdrant. A document identifier allows queries to target one report or search across the indexed collection.

BM25 lexical indexing is maintained for exact-match retrieval.

## 5. User asks a question

The frontend sends the question, optional document ID, and retrieval configuration to `POST /query`.

Example conceptual request:

```json
{
  "question": "What percentage of the Board of Directors are female?",
  "document_id": "<optional-document-id>",
  "hybrid_threshold": 0.0
}
```

## 6. Parallel retrieval

The same query is evaluated through:

### Dense retrieval

Finds semantically similar chunks and handles paraphrased wording.

### BM25 retrieval

Rewards exact lexical overlap and is especially useful for:

- table headings;
- category names;
- percentages;
- reporting terminology;
- identifiers.

## 7. Reciprocal Rank Fusion

The two ranked lists are combined using RRF.

The important point is that RRF works with rank positions rather than assuming that dense and BM25 scores are directly comparable.

## 8. Targeted structured reranking

A narrow deterministic signal handles a known structured-report failure mode.

For specific percentage questions about female representation, candidates containing the requested category, gender terminology, and percentage evidence receive a small boost.

This is intentionally targeted rather than a general heuristic layer.

## 9. Top evidence is passed to the generator

The final Top-5 chunks become the only report context supplied to the LLM.

The generator is therefore downstream of retrieval rather than being allowed to answer from general model knowledge.

## 10. Grounded answer generation

The LLM is instructed to:

- answer from supplied context only;
- preserve numerical values;
- respect reporting year and table row;
- avoid unsupported assumptions;
- abstain when the evidence is insufficient.

## 11. Abstention

When the supplied evidence does not support the question, the system returns:

```text
This information is not available in the provided report context.
```

This makes unsupported questions an explicit evaluated behavior rather than silently guessing.

## 12. Evidence returned to the UI

The frontend displays:

- the grounded answer;
- source document;
- source page;
- relevance score;
- retrieved source text.

The user can therefore inspect the evidence behind the generated answer.

## 13. Document lifecycle

Users can:

- list indexed documents;
- select a document as the active query scope;
- query all documents;
- delete a document and its indexed chunks.

## 14. Failure handling

The backend returns explicit HTTP errors for invalid uploads, parsing failures, storage failures, retrieval failures, and generation failures. The frontend surfaces these failures rather than presenting them as successful answers.

## 15. Deployment flow

```text
Browser
  │
  ▼
React/Vite on Vercel
  │ HTTPS
  ▼
FastAPI on Render
  ├──► Hugging Face embeddings
  ├──► Qdrant Cloud
  └──► Groq LLM
```

## 16. Example evidence flow

For a structured BRSR question asking for female representation on the Board of Directors, the retrieval layer was tested against the RIL BRSR FY 2024-25 table containing:

- Board of Directors: 14 total, 2 female, 14.29%;
- Key Management Personnel: 2 total, 1 female, 50.00%.

The targeted reranking signal moves the relevant table evidence into the Top-5 retrieval context, after which the grounded generator returns the supported percentage.

