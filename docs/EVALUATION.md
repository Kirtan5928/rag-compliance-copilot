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

Previously recorded results:

| Retrieval strategy | Recall@5 | MRR |
|---|---:|---:|
| Dense | 77.78% | 72.22% |
| BM25 | 94.44% | 83.52% |
| Hybrid + structured reranking | 100.00% | 82.13% |

The hybrid configuration achieved complete Top-5 evidence coverage on this benchmark, while BM25 had a slightly higher MRR. These metrics measure different properties and should not be collapsed into a single score.

## Full RAG evaluation

The full evaluator measures:

- Recall@5
- MRR
- Answer accuracy
- Abstention accuracy

A previous clean end-to-end evaluation before the final structured-query fix recorded:

| Metric | Result |
|---|---:|
| Recall@5 | 94.44% |
| MRR | 79.35% |
| Answer accuracy | 94.44% |
| Abstention accuracy | 100.00% |

These values are a **previous baseline**, not the final post-fix result. The final 20-question run is pending because the available LLM quota was exhausted during the previous attempt.

Run the evaluation with:

```powershell
python evaluation\\run_evaluation.py
```

Detailed per-question results are written to evaluation/results/latest_results.json.

## Interpretation

Retrieval and generation failures should be diagnosed separately.

| Retrieval | Generation | Interpretation |
|---|---|---|
| Correct | Correct | End-to-end success |
| Correct | Incorrect | Generation/grounding problem |
| Incorrect | Any | Retrieval problem |
| Abstention correct | — | Unsupported query handled safely |

This separation follows the general RAG evaluation principle that retrieval relevance and generation quality should be measured as distinct components. See the RAG evaluation survey: https://arxiv.org/abs/2405.07437.
