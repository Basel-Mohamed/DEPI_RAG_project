# requires: pip install schedule

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import settings
from app.services.embedding.embedding_service import EmbeddingService
from app.services.metadata_store import get_metadata_store
from app.services.preprocessing.preprocessing_service import DocumentProcessor
from app.services.rag.rag_builder import BuildService
from app.services.vectorstore.qdrant_store import QdrantService

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-embed and re-index built documents.")
    parser.add_argument("--mode", choices=["once", "schedule"], default="once", help="Run once or on a schedule.")
    parser.add_argument("--interval-hours", type=int, default=24, help="Hours between scheduled runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print files without rebuilding them.")
    return parser.parse_args()


def create_build_service() -> BuildService:
    embedding_service = EmbeddingService(settings)
    vector_store = QdrantService(embedding_service=embedding_service, settings=settings)
    return BuildService(
        document_processor=DocumentProcessor(settings),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def eligible_registry_items(registry: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    eligible: list[tuple[str, dict[str, Any]]] = []
    skipped = 0
    for file_id, metadata in registry.items():
        if metadata.get("status") == "built":
            eligible.append((file_id, metadata))
        else:
            skipped += 1
    return eligible, skipped


def run_reindex(*, mode: str, interval_hours: int, dry_run: bool) -> dict[str, Any]:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required for the reindex pipeline.") from exc

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Reindex Pipeline")

    start = time.perf_counter()
    registry_path = PROJECT_ROOT / "uploads" / "files.json"
    registry = get_metadata_store().read_file_registry(registry_path)
    eligible_files, skipped_count = eligible_registry_items(registry)
    build_service = None if dry_run else create_build_service()
    file_results: list[dict[str, Any]] = []
    reindexed_count = 0
    failed_count = 0

    with mlflow.start_run(run_name="reindex"):
        mlflow.log_params(
            {
                "mode": mode,
                "interval_hours": interval_hours,
                "dry_run": dry_run,
                "embedding_model": settings.EMBEDDING_MODEL,
            }
        )

        for file_id, metadata in eligible_files:
            filename = str(metadata.get("filename") or metadata.get("name") or file_id)
            file_path = Path(str(metadata.get("path", "")))
            logger.info("reindex started file_id=%s filename=%s", file_id, filename)

            if dry_run:
                print(f"[DRY RUN] Would reindex: {filename} (file_id={file_id})")
                file_results.append(
                    {
                        "file_id": file_id,
                        "filename": filename,
                        "path": str(file_path),
                        "status": "dry_run",
                    }
                )
                continue

            try:
                if build_service is None:
                    raise RuntimeError("Build service was not initialized.")
                result = build_service.build_document(file_path, source=file_id)
                reindexed_count += 1
                file_results.append(
                    {
                        "file_id": file_id,
                        "filename": filename,
                        "path": str(file_path),
                        "status": "reindexed",
                        "result": result,
                    }
                )
            except Exception as exc:
                failed_count += 1
                logger.exception("reindex failed file_id=%s filename=%s", file_id, filename)
                file_results.append(
                    {
                        "file_id": file_id,
                        "filename": filename,
                        "path": str(file_path),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        duration_seconds = time.perf_counter() - start
        summary = {
            "total_files": len(registry),
            "reindexed": reindexed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "duration_seconds": round(duration_seconds, 4),
            "files": file_results,
        }
        mlflow.log_metrics(
            {
                "total_files": len(registry),
                "reindexed_count": reindexed_count,
                "failed_count": failed_count,
                "duration_seconds": round(duration_seconds, 4),
            }
        )
        artifact_path = PROJECT_ROOT / "data" / "reindex_summary.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_path))

    print_summary(summary)
    if build_service is not None:
        close = getattr(build_service.vector_store, "close", None)
        if callable(close):
            close()
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("| total_files | reindexed | failed | skipped | duration_seconds |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    print(
        "| {total_files} | {reindexed} | {failed} | {skipped} | {duration_seconds:.4f} |".format(
            **summary
        )
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.mode == "once":
        run_reindex(mode=args.mode, interval_hours=args.interval_hours, dry_run=args.dry_run)
        return 0

    try:
        import schedule
    except ImportError as exc:
        raise RuntimeError("The schedule package is required for --mode schedule.") from exc

    schedule.every(args.interval_hours).hours.do(
        run_reindex,
        mode=args.mode,
        interval_hours=args.interval_hours,
        dry_run=args.dry_run,
    )
    print(f"Scheduler started. Next run in {args.interval_hours}h.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("scheduler stopped by user")
        print("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
