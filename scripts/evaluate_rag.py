from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.dependencies import get_inference_pipeline


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("items", [])
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError("Evaluation dataset must be .json or .csv")


def evaluate_items(items: list[dict[str, Any]], pipeline: Any) -> dict[str, Any]:
    results = []
    for item in items:
        question = str(item.get("question") or "").strip()
        expected_answer = str(item.get("expected_answer") or "").strip()
        expected_source = str(item.get("expected_source") or "").strip()
        if not question:
            continue

        start = time.perf_counter()
        response = pipeline.run(question)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        answer = str(response.get("answer") or "")
        sources = response.get("sources") or []
        source_ids = [
            str(source.get("metadata", {}).get("source") or source.get("title") or source.get("id"))
            for source in sources
        ]
        results.append(
            {
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "expected_source": expected_source,
                "latency_ms": latency_ms,
                "retrieved_documents": response.get("retrieval", {}).get("documents", 0),
                "source_hit": bool(expected_source and expected_source in source_ids),
                "answer_overlap": token_overlap(answer, expected_answer) if expected_answer else None,
            }
        )
    return summarize(results)


def token_overlap(answer: str, expected: str) -> float:
    answer_tokens = set(answer.lower().split())
    expected_tokens = set(expected.lower().split())
    if not expected_tokens:
        return 0.0
    return round(len(answer_tokens & expected_tokens) / len(expected_tokens), 4)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    source_items = [item for item in results if item["expected_source"]]
    overlap_items = [item for item in results if item["answer_overlap"] is not None]
    return {
        "summary": {
            "count": count,
            "avg_latency_ms": round(sum(item["latency_ms"] for item in results) / count, 2) if count else 0,
            "source_hit_rate": round(sum(item["source_hit"] for item in source_items) / len(source_items), 4) if source_items else None,
            "avg_answer_overlap": round(sum(item["answer_overlap"] for item in overlap_items) / len(overlap_items), 4) if overlap_items else None,
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "candidate_top_k": settings.RETRIEVAL_CANDIDATE_TOP_K,
            "context_top_k": settings.RERANKED_CONTEXT_TOP_K,
            "reranker_provider": settings.reranker_provider,
        },
        "results": results,
    }


def write_reports(report: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = report["summary"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions: {summary['count']}",
        f"- Average latency ms: {summary['avg_latency_ms']}",
        f"- Source hit rate: {summary['source_hit_rate']}",
        f"- Average answer token overlap: {summary['avg_answer_overlap']}",
        f"- Retrieval mode: {summary['retrieval_mode']}",
        f"- Candidate top-k: {summary['candidate_top_k']}",
        f"- Context top-k: {summary['context_top_k']}",
        f"- Reranker provider: {summary['reranker_provider']}",
        "",
        "## Lowest Overlap Samples",
    ]
    ranked = sorted(
        report["results"],
        key=lambda item: item["answer_overlap"] if item["answer_overlap"] is not None else 1,
    )
    for item in ranked[:5]:
        lines.extend(
            [
                "",
                f"### {item['question']}",
                f"- Expected source: {item['expected_source'] or 'n/a'}",
                f"- Source hit: {item['source_hit']}",
                f"- Answer overlap: {item['answer_overlap']}",
                f"- Latency ms: {item['latency_ms']}",
            ]
        )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def log_mlflow(report: dict[str, Any]) -> None:
    try:
        import mlflow
    except ImportError:
        return

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="rag-evaluation"):
        for key, value in report["summary"].items():
            if isinstance(value, int | float) and value is not None:
                mlflow.log_metric(key, value)
            elif value is not None:
                mlflow.log_param(key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--data", default=settings.EVAL_DATA_PATH)
    parser.add_argument("--output-dir", default=settings.EVAL_OUTPUT_DIR)
    args = parser.parse_args()

    report = evaluate_items(load_dataset(args.data), get_inference_pipeline())
    write_reports(report, args.output_dir)
    log_mlflow(report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
