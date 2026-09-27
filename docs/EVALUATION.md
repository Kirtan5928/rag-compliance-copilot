# Evaluation Methodology

## Dataset

The project uses a fixed 20-question golden set derived from the Reliance Industries BRSR FY 2024-25 report:

- 18 answerable questions
- 2 intentionally unanswerable questions
- numerical and table-oriented questions
- reporting-year disambiguation
- evidence-keyword checks
- explicit abstention cases

The dataset stores expected answers and evidence keywords so retrieval and generation can be evaluated separately.

## Retrieval metrics

### Recall@5

For each answerable question, retrieval succeeds when the Top-5 chunks contain all expected evidence keywords.

```text
Recall@5 = answerable queries with required evidence in Top-5
           -----------------------------------------------
                    total answerable queries
```

### Mean Reciprocal Rank (MRR)

For each answerable query, the reciprocal of the rank of the first chunk containing all expected evidence keywords is calculated.

```text
MRR = average(1 / first relevant rank)
```

Recall@5 measures whether the generator receives the evidence at all. MRR additionally measures how early the evidence appears in the ranked results.

## Retrieval comparison

The retrieval benchmark compares:

1. Dense retrieval
2. BM25 retrieval
3. Hybrid RRF + targeted structured reranking

This benchmark does not call the LLM, so it can be run repeatedly without consuming generation quota.

| Retrieval strategy | Recall@5 | MRR |
|---|---:|---:|
| Dense | 77.78% | 72.22% |
| BM25 | 94.44% | 83.52% |
| Hybrid + structured reranking | **100.00%** | 82.13% |

The hybrid configuration achieved complete Top-5 evidence coverage on this benchmark, while BM25 had a slightly higher MRR. These metrics measure different properties and should not be collapsed into a single score.

## Final end-to-end RAG evaluation

The final run was completed successfully on 27 September 2026 using the 20-question golden set.

| Metric | Final result |
|---|---:|
| Questions | 20 |
| Answerable questions | 18 |
| Unanswerable questions | 2 |
| Recall@5 | **100.00%** |
| MRR | **0.8213** |
| Answer accuracy | **100.00%** |
| Abstention accuracy | **100.00%** |

All 18 answerable questions had the required evidence within the Top-5 retrieval results. All 20 generated responses matched the evaluator's expected outcomes, including correct abstention on both intentionally unanswerable questions.

Detailed per-question results are saved to:

`evaluation/results/latest_results.json`

Run the full evaluation with:

```powershell
python evaluation\\run_evaluation.py
```

The evaluator writes the latest detailed results to `evaluation/results/latest_results.json`.

## Interpretation

Retrieval and generation failures should be diagnosed separately.

| Retrieval | Generation | Interpretation |
|---|---|---|
| Correct | Correct | End-to-end success |
| Correct | Incorrect | Generation/grounding problem |
| Incorrect | Any | Retrieval problem |
| Abstention correct | — | Unsupported query handled safely |

The evaluation design follows the broader RAG evaluation principle of measuring retrieval and generation as distinct components. See the RAG evaluation survey: https://arxiv.org/abs/2405.07437.

## Scope of the reported metrics

The 100% results are **benchmark-specific**. They describe performance on this fixed RIL BRSR FY 2024-25 question set and should not be interpreted as a guarantee of 100% accuracy for arbitrary PDFs or unseen question distributions.

The application is designed to ingest other text-based PDFs, but new document collections require their own evaluation set to establish retrieval and answer quality.

## Limitations

- The golden set currently covers one BRSR report.
- Scanned/image-only PDFs are not supported by the current text-extraction path.
- Broader multi-company and multi-year evaluation is still required.
- Retrieval scores are ranking signals, not probabilities.
- The evaluator's answer-accuracy metric is based on the project's expected-answer matching logic; it is not a human study or a claim of universal factual accuracy.
