# Evaluation Methodology

## Dataset

The project uses a fixed golden set derived from the Reliance Industries BRSR FY 2024-25 report.

- 20 total questions
- 18 answerable
- 2 intentionally unanswerable
- Categories: energy, water, emissions, air emissions, waste, workforce, abstention

The dataset stores expected answers and evidence keywords so retrieval and generation can be evaluated separately.

## Retrieval metrics

### Recall@5

For each answerable question, retrieval succeeds when the Top-5 chunks contain all expected evidence keywords.

```
Recall@5 = relevant answerable queries retrieved in Top-5
          --------------------------------------------
                  total answerable queries
```

### Mean Reciprocal Rank (MRR)

For each answerable query, the reciprocal of the rank of the first chunk containing all expected evidence keywords is calculated.

```
MRR = average(1 / first relevant rank)
```

Recall@5 measures whether the generator receives the evidence at all. MRR additionally measures how early the evidence appears in the ranked results.

## Retrieval comparison

`evaluation/benchmark_retrieval.py` compares:

1. Dense retrieval
2. BM25 retrieval
3. Hybrid RRF + structured reranking

This benchmark does not call the LLM, so it can be run repeatedly without consuming Groq generation quota.

Run:

```powershell
python evaluation\benchmark_retrieval.py
```

Results are written to:

```text
evaluation/results/retrieval_benchmark.json
```

## Full RAG evaluation

`evaluation/run_evaluation.py` evaluates the complete retrieval + generation pipeline.

It measures:

- Recall@5
- MRR
- Answer accuracy
- Abstention accuracy

Run it deliberately because every answerable question invokes the configured LLM.

```powershell
python evaluation\run_evaluation.py
```

## Interpretation

Retrieval and generation failures should be diagnosed separately.

| Retrieval | Generation | Interpretation |
|---|---|---|
| Correct | Correct | End-to-end success |
| Correct | Incorrect | Generation/grounding problem |
| Incorrect | Any | Retrieval problem |
| Abstention correct | — | Unsupported query handled safely |

The final report should compare dense, BM25, and hybrid retrieval using the same fixed questions and corpus. If hybrid retrieval does not improve the measured workload, that result should be reported rather than assuming hybrid search is automatically better.
