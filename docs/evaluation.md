# Model Evaluation

Run:

```powershell
python scripts/evaluate_rag.py --data data/evaluation/questions.json --output-dir reports/evaluation
```

Dataset format:

```json
[
  {
    "question": "How do refunds work?",
    "expected_answer": "Refunds are available for eligible orders.",
    "expected_source": "refund-policy.pdf"
  }
]
```

Outputs:

- `reports/evaluation/evaluation_results.json`
- `reports/evaluation/evaluation_report.md`
- Optional MLflow local run under `mlruns/` when `mlflow` is installed.

Metrics:

- Average latency.
- Source hit rate when `expected_source` is provided.
- Token overlap against `expected_answer`.
- Retrieval settings and reranker provider.

BLEU/ROUGE can be added later as optional dependencies for stricter text-generation scoring.
