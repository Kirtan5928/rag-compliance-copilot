# RAG Compliance Copilot

A retrieval-augmented generation system for querying SEBI BRSR/ESG compliance reports using natural language, with grounded (source-backed) answers instead of LLM hallucination.

## Status: Phase 1 (M1) — Core RAG Loop

This phase proves the fundamental retrieval + generation pipeline end-to-end, using the minimum viable stack before adding hybrid search, clause classification, or a web interface.

### What it does
- Ingests a PDF report and extracts raw text
- Chunks the text into overlapping segments (500 words, 50-word overlap) to preserve context across chunk boundaries
- Embeds each chunk locally using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Stores embeddings in a persistent local ChromaDB vector store
- On query: embeds the question, retrieves the top-3 most semantically similar chunks
- Feeds retrieved chunks + question to a locally-run LLM (Llama 3, via Ollama) with an explicit grounding instruction — answer only from provided context, no fabrication

### Why this matters for compliance use cases
BRSR/ESG reports are dense with precise figures (energy consumption, emissions, board composition data). A general-purpose LLM asked about these numbers from memory will hallucinate. This pipeline forces every answer to be traceable back to the actual source document, which is the baseline requirement for any tool used in a regulatory/compliance context.

### Tech stack
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — local, free, CPU-only
- **Vector store:** ChromaDB (persistent, local)
- **LLM:** Llama 3 via Ollama — local inference, zero API cost
- **PDF parsing:** pypdf

### Setup
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
pip install sentence-transformers chromadb pypdf ollama
ollama pull llama3
\`\`\`

Drop a BRSR/ESG PDF into `data/sample.pdf`, then:

\`\`\`bash
python ingest.py   # builds the vector store
python query.py    # ask questions interactively
\`\`\`

### Known limitation (by design, addressed in later phases)
LLMs are language models, not calculators — arithmetic performed *within* a generated answer (e.g. summing multiple retrieved figures) can be unreliable even when the retrieved source numbers are correct. Phase 2 adds explicit source citation so users can verify figures directly against the retrieved chunk rather than trusting model-generated arithmetic.

## Roadmap
- [x] **M1** — Core RAG loop: ingestion, chunking, embedding, retrieval, grounded generation
- [ ] **M2** — Source citation tracing, multi-document testing
- [ ] **M3** — Hybrid retrieval (BM25 + dense), DeBERTa clause classifier
- [ ] **M4** — FastAPI backend + React frontend