from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import settings
from app.services.embedding.embedding_service import EmbeddingService
from app.services.llm.llm_factory import create_llm_service
from app.services.rag.rag_inference import RagInferencePipeline
from app.services.vectorstore.qdrant_store import QdrantService

logger = logging.getLogger(__name__)

EXPERIMENT_CONFIGS = [
    {
        "run_name": "dense_no_reranker",
        "retrieval_mode": "dense",
        "reranker_provider": None,
        "score_threshold": 0.6,
    },
    {
        "run_name": "sparse_no_reranker",
        "retrieval_mode": "sparse",
        "reranker_provider": None,
        "score_threshold": 0.0,
    },
    {
        "run_name": "hybrid_no_reranker",
        "retrieval_mode": "hybrid",
        "reranker_provider": None,
        "score_threshold": 0.6,
    },
    {
        "run_name": "hybrid_with_reranker",
        "retrieval_mode": "hybrid",
        "reranker_provider": "cohere",
        "score_threshold": 0.6,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG configurations and log RAGAS scores to MLflow.")
    parser.add_argument("--eval-dataset", required=True, type=Path, help="Path to golden Q&A JSONL.")
    parser.add_argument("--experiment-name", default="RAG Pipeline Comparison", help="MLflow experiment name.")
    parser.add_argument("--tracking-uri", default="./mlruns", help="MLflow tracking URI.")
    parser.add_argument("--top-k", default=5, type=int, help="Number of documents to retrieve per query.")
    parser.add_argument("--sample", default=0, type=int, help="Max evaluation records per run, 0 means all.")
    return parser.parse_args()


def load_eval_records(path: Path, sample: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if sample and len(records) >= sample:
                break
    return records


def should_skip_config(config: dict[str, Any]) -> bool:
    reranker_provider = config.get("reranker_provider")
    if reranker_provider == "cohere" and not (
        os.getenv("COHERE_API_KEY") or os.getenv("cohere_api_key") or settings.cohere_api_key
    ):
        logger.warning("Skipping %s because COHERE_API_KEY is not set.", config["run_name"])
        return True
    return False


def build_pipeline(config: dict[str, Any]) -> RagInferencePipeline:
    settings.RETRIEVAL_MODE = str(config["retrieval_mode"])
    settings.reranker_provider = config.get("reranker_provider")
    settings.SCORE_THRESHOLD = float(config["score_threshold"])
    embedding_service = EmbeddingService(settings)
    vector_store = QdrantService(embedding_service=embedding_service, settings=settings)
    return RagInferencePipeline(vector_store=vector_store, settings=settings)


def question_from_record(record: dict[str, Any]) -> str:
    return str(record.get("question") or record.get("user_input") or record.get("query") or "").strip()


def ground_truth_from_record(record: dict[str, Any]) -> str:
    value = record.get("ground_truth", record.get("reference", record.get("answer", "")))
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def contexts_from_response(response: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for source in response.get("sources", []) or []:
        content = source.get("content")
        if content is not None:
            contexts.append(str(content))
    return contexts


def run_pipeline_eval(
    pipeline: RagInferencePipeline,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        question = question_from_record(record)
        if not question:
            logger.warning("Skipping eval record %s without a question.", index)
            continue
        response = pipeline.run(question, top_k=top_k, mode=retrieval_mode)
        results.append(
            {
                "question": question,
                "answer": str(response.get("answer", "")),
                "contexts": contexts_from_response(response),
                "ground_truth": ground_truth_from_record(record),
            }
        )
    return results


def evaluate_ragas(results: list[dict[str, Any]]) -> dict[str, float]:
    try:
        _install_ragas_vertexai_compat()
        import ragas
        from datasets import Dataset as HuggingFaceDataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing .* from 'ragas.metrics' is deprecated.*",
                category=DeprecationWarning,
            )
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
    except ImportError as exc:
        print(
            "RAGAS evaluation dependencies are not installed. "
            f"Install ragas and datasets, then rerun this script. Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    ragas_dataset = getattr(ragas, "Dataset", None)
    if ragas_dataset is not None and hasattr(ragas_dataset, "from_list"):
        dataset = ragas_dataset.from_list(results)
    else:
        dataset = HuggingFaceDataset.from_list(results)
    ragas_llm = LangchainLLMWrapper(create_llm_service(settings)._get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(
        RagasEmbeddingAdapter(EmbeddingService(settings))
    )
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        column_map={
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        },
    )
    return {
        "faithfulness": _metric_value(scores, "faithfulness"),
        "answer_relevancy": _metric_value(scores, "answer_relevancy"),
        "context_recall": _metric_value(scores, "context_recall"),
        "context_precision": _metric_value(scores, "context_precision"),
    }


def _metric_value(scores: Any, name: str) -> float:
    try:
        if isinstance(scores, dict):
            return float(scores[name])
        if hasattr(scores, "__getitem__"):
            return float(scores[name])
    except Exception:
        pass
    try:
        dataframe = scores.to_pandas()
        columns = [column for column in dataframe.columns if column.startswith(name)]
        if columns:
            return float(dataframe[columns[0]].mean())
    except Exception:
        logger.exception("failed to extract ragas metric %s", name)
    return 0.0


class RagasEmbeddingAdapter:
    """Minimal LangChain-style embeddings adapter for RAGAS evaluation."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        chunks = [{"text": text, "metadata": {}} for text in texts]
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        return [list(chunk["embedding"]) for chunk in embedded_chunks]

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_service.embed_query(text)


def _install_ragas_vertexai_compat() -> None:
    """Provide an unused VertexAI module expected by some RAGAS/LangChain combos."""

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = module


def log_results_artifact(results: list[dict[str, Any]]) -> Path:
    artifact_path = PROJECT_ROOT / "data" / "results_summary.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return artifact_path


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run name | faithfulness | answer_relevancy | context_recall | context_precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {faithfulness:.4f} | {answer_relevancy:.4f} | "
            "{context_recall:.4f} | {context_precision:.4f} |".format(**row)
        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        import mlflow
    except ImportError as exc:
        print("MLflow is not installed. Install mlflow, then rerun this script.", file=sys.stderr)
        return 1

    eval_records = load_eval_records(args.eval_dataset, args.sample)
    if not eval_records:
        raise ValueError(f"No evaluation records found in {args.eval_dataset}.")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    comparison_rows: list[dict[str, Any]] = []
    for config in EXPERIMENT_CONFIGS:
        if should_skip_config(config):
            continue

        start = time.perf_counter()
        pipeline = build_pipeline(config)
        try:
            results = run_pipeline_eval(
                pipeline,
                eval_records,
                top_k=args.top_k,
                retrieval_mode=str(config["retrieval_mode"]),
            )
            ragas_scores = evaluate_ragas(results)
            duration_seconds = time.perf_counter() - start

            with mlflow.start_run(run_name=str(config["run_name"])):
                mlflow.log_params(
                    {
                        "retrieval_mode": config["retrieval_mode"],
                        "reranker_provider": config.get("reranker_provider") or "none",
                        "score_threshold": config["score_threshold"],
                        "top_k": args.top_k,
                        "embedding_model": settings.EMBEDDING_MODEL,
                        "chunk_size": settings.CHUNK_SIZE,
                        "eval_sample_size": len(results),
                    }
                )
                mlflow.log_metrics(
                    {
                        **{name: round(float(value), 4) for name, value in ragas_scores.items()},
                        "eval_duration_seconds": round(duration_seconds, 4),
                    }
                )
                artifact_path = log_results_artifact(results)
                mlflow.log_artifact(str(artifact_path))

            comparison_rows.append(
                {
                    "run_name": config["run_name"],
                    **{name: round(float(value), 4) for name, value in ragas_scores.items()},
                }
            )
        finally:
            vector_store = getattr(pipeline, "vector_store", None)
            close = getattr(vector_store, "close", None)
            if callable(close):
                close()

    table = markdown_table(comparison_rows)
    print(table)
    output_path = PROJECT_ROOT / "data" / "mlflow_comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
