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
from app.services.rag.rag_builder import RagBuildPipeline, RagBuilderError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RAG index for one local document."
    )
    parser.add_argument(
        "document_path",
        help="Local path to the document you want to index.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing chunks for this source before upserting.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not attach extracted page images to chunk metadata.",
    )
    return parser.parse_args()


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    document_path = Path(args.document_path).expanduser().resolve()

    if not document_path.exists():
        print_json(
            {
                "ok": False,
                "error": f"Document not found: {document_path}",
            }
        )
        return 1

    pipeline: RagBuildPipeline | None = None
    try:
        pipeline = RagBuildPipeline(settings=Settings())
        summary = pipeline.build(
            document_path,
            include_page_images=not args.no_images,
            replace_existing=args.replace_existing,
        )
    except RagBuilderError as exc:
        print_json({"ok": False, "error": str(exc)})
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()

    print_json(
        {
            "ok": True,
            "document_path": str(document_path),
            "summary": summary,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
