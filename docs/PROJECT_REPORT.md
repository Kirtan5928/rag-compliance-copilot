# RAG Compliance Copilot — Project Report Draft

## 1. Problem Statement

BRSR and ESG reports contain large volumes of structured and semi-structured information. Answering compliance questions manually can require locating the correct report, year, section, table, row, and numerical value.

A conventional LLM chatbot can produce fluent answers but may not reliably preserve exact figures or provide traceable evidence. The project therefore focuses on document-grounded question answering where retrieval quality and source traceability are first-class requirements.

## 2. Motivation

The system is designed around three practical requirements:

1. **Grounding:** answers should be based on the uploaded report rather than unsupported model knowledge.
2. **Traceability:** users should be able to inspect the source page and chunk behind an answer.
3. **Safe uncertainty:** when the report does not contain enough evidence, the system should abstain rather than fabricate an answer.

## 3. Objectives

- Build an end-to-end RAG application for BRSR/ESG reports.
- Preserve page-level evidence during PDF ingestion.
- Compare dense and lexical retrieval.
- Combine retrieval signals with Reciprocal Rank Fusion.
- Address structured table retrieval with a targeted reranking layer.
- Generate grounded answers with explicit abstention.
- Expose evidence through a usable web interface.
- Evaluate retrieval and generation separately.

## 4. Proposed System

The system consists of:

```text
PDF → extraction → page-aware chunks → embeddings/Qdrant
                                   ↓
Question → dense + BM25 → RRF → structured reranking
                                   ↓
                              Top-5 evidence
                                   ↓
                             grounded LLM
                                   ↓
                         answer / abstention
                                   ↓
                         answer + page sources
```

## 5. Document Ingestion

The backend validates the upload, enforces a 100 MB limit, extracts text page by page using pypdf, and rejects unreadable or text-empty documents.

Page-aware chunking preserves the relationship between each chunk and its source page.

## 6. Chunking Strategy

Chunks are created from page text while preserving:

- document identity;
- page number;
- chunk index;
- source text.

This avoids losing the page context needed for evidence citations.

## 7. Embedding and Vector Storage

Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in Qdrant together with source metadata.

The document ID enables document-level filtering and deletion.

## 8. Hybrid Retrieval

The retrieval stage uses two complementary approaches.

### Dense retrieval

Useful for semantic similarity and paraphrasing.

### BM25 retrieval

Useful for exact terminology, table headings, identifiers, and numerical expressions.

The ranked lists are combined using RRF.

## 9. Structured Reranking

Structured BRSR tables create a specific retrieval challenge: a semantically related paragraph can rank above the exact table row containing the required number.

A targeted deterministic reranking signal is therefore applied to a narrow class of female-representation percentage queries. It looks for the requested category, gender terminology, and percentage evidence.

The goal is not to replace the retriever but to improve placement of exact structured evidence.

## 10. Grounded Generation

Only the final retrieved evidence is supplied to the LLM.

The generation instructions require exact numerical preservation, correct year/table selection, and no unsupported inference.

## 11. Abstention

For unsupported questions, the application returns:

```text
This information is not available in the provided report context.
```

Abstention is explicitly included in the evaluation set.

## 12. Evidence and Explainability

Each answer is accompanied by retrieved source metadata including document name and page information. This makes the system inspectable and supports manual verification.

## 13. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| API | FastAPI |
| PDF extraction | pypdf |
| Embeddings | all-MiniLM-L6-v2 / Hugging Face |
| Vector database | Qdrant |
| Lexical retrieval | rank-bm25 |
| LLM | Groq |
| Backend deployment | Render |
| Frontend deployment | Vercel |

## 14. Evaluation Methodology

The fixed golden set contains 20 questions from the RIL BRSR FY 2024-25 report:

- 18 answerable;
- 2 unanswerable.

Metrics:

### Recall@5

Whether all expected evidence keywords occur within the five chunks passed to the generator.

### MRR

How highly the first chunk containing all required evidence is ranked.

### Answer accuracy

Whether the generated answer matches the expected answer.

### Abstention accuracy

Whether unsupported questions are rejected correctly.

## 15. Retrieval Results

| Retrieval strategy | Recall@5 | MRR |
|---|---:|---:|
| Dense | 77.78% | 72.22% |
| BM25 | 94.44% | 83.52% |
| Hybrid + structured reranking | **100.00%** | 82.13% |

The hybrid configuration achieved complete Top-5 evidence coverage on this benchmark. BM25 produced a slightly higher MRR, so the results demonstrate that coverage and ranking position are distinct evaluation dimensions.

## 16. Previous End-to-End Baseline

A previous clean end-to-end run, before the final structured-query fix, recorded:

| Metric | Result |
|---|---:|
| Recall@5 | 94.44% |
| MRR | 79.35% |
| Answer accuracy | 94.44% |
| Abstention accuracy | 100.00% |

The final post-fix 20-question LLM evaluation is still pending because the available generation quota was exhausted during the previous run. These values must therefore be presented as a baseline, not as final results.

## 17. Example Structured Evidence Case

For the RIL BRSR FY 2024-25 question concerning female representation on the Board of Directors, the source table contains:

```text
Board of Directors       14 total   2 female   14.29%
Key Management Personnel  2 total   1 female   50.00%
```

The targeted retrieval signal places the relevant table evidence into the Top-5 context, and the grounded generation path returns the supported percentage.

## 18. Limitations

- The current pipeline is optimized for text-based PDFs.
- Scanned/image-only PDFs are not supported.
- The golden set currently covers one BRSR report.
- Broader multi-company and multi-year evaluation is still required.
- Retrieval scores are ranking signals, not probabilities.
- Final LLM evaluation depends on available provider quota.

## 19. Novelty / Contribution

The project contribution is primarily engineering and experimental rather than claiming a new foundation model.

The testable contribution is the retrieval comparison:

```text
Dense
  vs
BM25
  vs
Hybrid RRF
  vs
Hybrid RRF + targeted structured reranking
```

The project also treats abstention and page-level evidence as explicit system behaviors rather than afterthoughts.

## 20. Future Scope

- Complete the post-fix 20-question end-to-end evaluation.
- Expand the golden set across multiple companies and reporting years.
- Add OCR for scanned reports.
- Add stronger reranking models for broader table retrieval.
- Add richer evaluation of citation correctness and faithfulness.
- Explore multi-document comparative questions.
- Improve query analytics and observability.

## 21. Conclusion

RAG Compliance Copilot provides a complete document-grounded pipeline for BRSR/ESG question answering. Its architecture combines page-aware ingestion, dense retrieval, BM25, RRF, targeted structured reranking, grounded generation, abstention, and source display.

The current retrieval benchmark demonstrates complete Top-5 evidence coverage on the fixed dataset for the hybrid configuration. The final project evaluation should preserve the separation between retrieval metrics and generation metrics and clearly identify the previous end-to-end numbers as a baseline until the post-fix run is completed.
