# Retrieval Pipeline

## Objective

The retrieval layer is designed for compliance reports where the answer may depend on an exact table row, number, category, or reporting year.

## Pipeline

```text
Question
   │
   ├───────────────┐
   ▼               ▼
Dense search     BM25 search
   │               │
   └───────┬───────┘
           ▼
      RRF fusion
           │
           ▼
 Structured reranking
           │
           ▼
       Top-5 chunks
           │
           ▼
     LLM generation
```

## Dense retrieval

The query is embedded into the same vector space as indexed report chunks. Dense retrieval is useful when the question and source use different but semantically related wording.

Example:

```text
Question: How much of the board is female?
Source: No. and percentage of females — Board of Directors
```

The semantic relationship can be captured even when exact words differ.

## BM25 retrieval

BM25 provides lexical matching. It is particularly valuable for structured documents because exact tokens can carry strong meaning:

- Board of Directors
- Key Management Personnel
- percentage
- female
- emissions
- FY 2024-25

It also helps with numbers and table terminology that may be less reliably represented by broad semantic similarity.

## Reciprocal Rank Fusion

Dense and BM25 scores should not simply be added because they are produced on different scales.

RRF instead uses ranked positions. Conceptually:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

where `k` is the rank constant and the contribution comes from each retrieval list.

The implementation uses equal weighting of the dense and BM25 ranked lists.

## Targeted structured reranking

The BRSR corpus contains many tables. A generic semantic ranker can place a conceptually related chunk above the exact table row required to answer a numerical question.

The implementation therefore adds a small deterministic boost for a narrow query class:

1. the query asks for a percentage;
2. the query targets female representation;
3. the query targets Board of Directors or Key Management Personnel;
4. the candidate contains the relevant category and percentage evidence.

This does not replace retrieval. It only resolves a known structured-evidence ranking failure mode.

## Why Top-5?

The evaluator measures whether the required evidence reaches the Top-5. The same Top-5 context is passed to the generator, keeping the context window focused while making retrieval quality directly measurable.

## Retrieval benchmark

| Strategy | Recall@5 | MRR |
|---|---:|---:|
| Dense | 77.78% | 72.22% |
| BM25 | 94.44% | 83.52% |
| Hybrid + structured reranking | **100.00%** | 82.13% |

Interpretation:

- Dense retrieval alone misses some exact structured evidence.
- BM25 improves coverage substantially on the current BRSR benchmark.
- Hybrid retrieval provides complete Top-5 evidence coverage on the benchmark.
- BM25's MRR is slightly higher than the hybrid configuration, showing why Recall@5 and MRR should be reported separately.

## Reproducibility

Run the retrieval-only benchmark with:

```powershell
python evaluation\benchmark_retrieval.py
```

This benchmark does not call the LLM, so it can be repeated without consuming generation quota.

## Failure diagnosis

| Retrieval result | Generation result | Likely issue |
|---|---|---|
| Evidence absent | Any answer | Retrieval failure |
| Evidence present | Wrong answer | Generation/grounding failure |
| Evidence present | Correct answer | End-to-end success |
| Unsupported query | Abstention | Safe handling |

