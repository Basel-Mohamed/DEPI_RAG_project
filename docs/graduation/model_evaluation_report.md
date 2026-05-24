# Model Evaluation Report

## Pipeline Under Evaluation

The trained RAG pipeline combines the `intfloat/multilingual-e5-large` dense embedding model, FastEmbed BM25 sparse retrieval, Qdrant vector storage, reciprocal-rank-fusion hybrid search, optional Cohere or Azure Cohere reranking, and Cohere or Azure OpenAI generation.

## Experiment Results

| Run name | Faithfulness | Answer relevancy | Context recall | Context precision |
| --- | ---: | ---: | ---: | ---: |
| dense_no_reranker | nan | 1.0000 | 1.0000 | 1.0000 |
| sparse_no_reranker | nan | 1.0000 | 1.0000 | 1.0000 |
| hybrid_no_reranker | 0.8000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_with_reranker | nan | 0.9788 | 1.0000 | 1.0000 |

## Interpretation

The available evaluation set shows perfect context recall and context precision across the compared retrieval modes, which means the target context is being retrieved for the current test question. Hybrid retrieval produced a measurable faithfulness score of 0.8000 and maintained perfect answer relevancy, recall, and precision. The reranked hybrid run kept recall and precision at 1.0000, with answer relevancy at 0.9788.

## Current Limitations

- The evaluation dataset currently contains a very small number of examples, so results should be treated as a smoke test rather than a statistically stable benchmark.
- Several faithfulness values are `nan`, which indicates the metric could not be computed for those runs and should be investigated before final production reporting.
- BLEU and ROUGE are listed in the graduation requirement; the implemented project currently emphasizes RAGAS-style faithfulness, answer relevancy, context recall, and context precision, which are better aligned with grounded support QA.

## Optimization Plan

- Expand `data/eval_dataset.jsonl` with 30-50 representative support queries.
- Track dense, sparse, hybrid, hybrid plus reranker, and threshold variants in MLflow.
- Add exact-answer checks for policy numbers, dates, rates, and approval roles.
- Use negative-feedback records from `/feedback` to seed future regression tests.
