from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.core.config import Settings
from app.services.rag.rag_inference import RagInferenceError, RagInferencePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question using the RAG inference pipeline."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--source",
        help=(
            "Optional exact source filter. Use the source value printed by "
            "build_document.py, usually the absolute document path."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Final number of chunks sent to the LLM.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="Number of chunks retrieved before optional reranking.",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse", "hybrid"),
        default=None,
        help="Retrieval mode. Defaults to RETRIEVAL_MODE from .env.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full response as JSON.",
    )
    return parser.parse_args()


def compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for source in sources:
        metadata = source.get("metadata") or {}
        media = source.get("media") or []
        compact.append(
            {
                "rank": source.get("rank"),
                "id": source.get("id"),
                "title": source.get("title"),
                "source": metadata.get("source"),
                "page_number": metadata.get("page_number"),
                "retrieval_score": metadata.get("retrieval_score"),
                "rerank_score": metadata.get("rerank_score"),
                "media_count": len(media),
                "content_preview": str(source.get("content") or "")[:240],
            }
        )
    return compact


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": response.get("answer"),
        "retrieval": response.get("retrieval") or {},
        "sources": compact_sources(response.get("sources") or []),
        "model_returned_images": response.get("images") or [],
    }


def print_human_response(response: dict[str, Any]) -> None:
    print("\nANSWER")
    print(response.get("answer", "").strip() or "(empty answer)")

    retrieval = response.get("retrieval") or {}
    print("\nRETRIEVAL")
    print_json(retrieval)

    sources = compact_sources(response.get("sources") or [])
    print("\nSOURCES")
    print_json({"sources": sources})

    images = response.get("images") or []
    if images:
        print("\nMODEL RETURNED IMAGES")
        print_json({"image_count": len(images), "images": images})


def main() -> int:
    args = parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()

    if not question:
        print_json({"ok": False, "error": "Question is required."})
        return 1

    filter_field = "source" if args.source else None
    filter_value = args.source

    pipeline: RagInferencePipeline | None = None
    try:
        pipeline = RagInferencePipeline(settings=Settings())
        response = pipeline.run(
            question,
            top_k=args.top_k,
            retrieval_top_k=args.retrieval_top_k,
            mode=args.mode,
            filter_field=filter_field,
            filter_value=filter_value,
        )
    except RagInferenceError as exc:
        print_json({"ok": False, "error": str(exc)})
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()

    if args.json:
        print_json({"ok": True, "response": compact_response(response)})
    else:
        print_human_response(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
