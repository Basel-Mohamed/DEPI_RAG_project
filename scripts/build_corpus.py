import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.preprocessing.deduplicator import deduplicate_chunks
from app.services.preprocessing.preprocessing_service import DocumentProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL chunk corpus from raw documents.")
    parser.add_argument("--input", required=True, help="Path to folder containing raw documents.")
    parser.add_argument("--output", required=True, help="Path for the output JSONL file.")
    parser.add_argument(
        "--source-prefix",
        default=None,
        help="Optional prefix stripped from source paths in metadata.",
    )
    return parser.parse_args()


def discover_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS
    )


def process_file(processor: DocumentProcessor, file_path: Path) -> tuple[list[dict[str, Any]], int]:
    pages = processor._load_pages(file_path)
    chunks = processor._split_and_clean(pages, source=str(file_path))
    return deduplicate_chunks(chunks)


def strip_source_prefix(chunks: list[dict[str, Any]], source_prefix: str | None) -> None:
    if source_prefix is None:
        return

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = str(metadata.get("source", ""))
        if source.startswith(source_prefix):
            metadata["source"] = source[len(source_prefix):].lstrip("/\\")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_dir}")

    files = discover_files(input_dir)
    processor = DocumentProcessor()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    total_chunks = 0
    duplicates_removed = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for file_index, file_path in enumerate(files, start=1):
            print(f"Processing file {file_index}/{len(files)}: {file_path.name}")
            try:
                chunks, removed_count = process_file(processor, file_path)
            except Exception:
                logger.exception("Failed to process file: %s", file_path)
                continue

            strip_source_prefix(chunks, args.source_prefix)
            for chunk in chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            files_processed += 1
            total_chunks += len(chunks)
            duplicates_removed += removed_count

    print(f"Files processed: {files_processed}")
    print(f"Total chunks: {total_chunks}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
