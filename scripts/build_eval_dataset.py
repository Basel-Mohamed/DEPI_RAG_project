from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import settings
from app.services.llm.llm_factory import create_llm_service
from app.services.llm.providers.base_llm import BaseLlmService, LlmServiceError

logger = logging.getLogger(__name__)

QUESTION_GENERATION_SYSTEM_PROMPT = (
    "You are a question generation assistant. Given a text passage, generate one "
    "realistic customer support question that can be answered using only this "
    "passage, and the correct answer grounded strictly in the passage. Respond "
    "only with valid JSON in this exact format, no extra text:\n"
    '{"question": "...", "answer": "...", "context": "..."}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a golden JSONL Q&A evaluation dataset from corpus chunks."
    )
    parser.add_argument(
        "--corpus",
        default="data/corpus.jsonl",
        help="Path to the JSONL corpus produced by scripts/build_corpus.py.",
    )
    parser.add_argument(
        "--output",
        default="data/eval_dataset.jsonl",
        help="Path for the output evaluation JSONL file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of corpus chunks to sample for Q&A generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file does not exist: {corpus_path}")

    chunks: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {corpus_path}"
                ) from exc
            if isinstance(chunk, dict):
                chunks.append(chunk)

    return chunks


def sample_chunks(
    chunks: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples < 0:
        raise ValueError("--samples must be greater than or equal to 0.")

    random.seed(seed)
    sample_size = min(samples, len(chunks))
    return random.sample(chunks, sample_size)


def chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("content") or "").strip()


def generate_eval_record(llm_service: BaseLlmService, passage: str) -> dict[str, str]:
    raw_response = invoke_llm_directly(llm_service, passage)
    record = parse_json_response(raw_response)

    missing_keys = {"question", "answer", "context"} - set(record)
    if missing_keys:
        raise ValueError(
            "LLM response is missing required keys: " + ", ".join(sorted(missing_keys))
        )

    return {
        "question": str(record["question"]).strip(),
        "answer": str(record["answer"]).strip(),
        "context": str(record["context"]).strip(),
    }


def invoke_llm_directly(llm_service: BaseLlmService, passage: str) -> str:
    get_llm = getattr(llm_service, "_get_llm", None)
    if get_llm is None:
        raise LlmServiceError(
            "The configured LLM service does not expose a direct chat model."
        )

    try:
        response = get_llm().invoke(
            [
                ("system", QUESTION_GENERATION_SYSTEM_PROMPT),
                ("human", f"Passage:\n{passage.strip()}"),
            ]
        )
    except Exception as exc:
        raise LlmServiceError("Question generation LLM invocation failed.") from exc

    return extract_response_text(response)


def extract_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def parse_json_response(raw_response: str) -> dict[str, Any]:
    cleaned = strip_json_fence(raw_response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def build_dataset(
    *,
    corpus_path: Path,
    output_path: Path,
    samples: int,
    seed: int,
) -> dict[str, int]:
    chunks = load_corpus(corpus_path)
    sampled_chunks = sample_chunks(chunks, samples=samples, seed=seed)
    llm_service = create_llm_service(settings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped_short = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, chunk in enumerate(sampled_chunks, start=1):
            passage = chunk_text(chunk)
            if len(passage) < 150:
                skipped_short += 1
                continue

            try:
                record = generate_eval_record(llm_service, passage)
            except Exception:
                failed += 1
                logger.exception("Failed to generate eval record for sample %d.", index)
                continue

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated += 1

    return {
        "corpus_chunks": len(chunks),
        "sampled": len(sampled_chunks),
        "generated": generated,
        "skipped_short": skipped_short,
        "failed": failed,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    summary = build_dataset(
        corpus_path=Path(args.corpus),
        output_path=Path(args.output),
        samples=args.samples,
        seed=args.seed,
    )

    print(f"Corpus chunks: {summary['corpus_chunks']}")
    print(f"Sampled chunks: {summary['sampled']}")
    print(f"Generated records: {summary['generated']}")
    print(f"Skipped short chunks: {summary['skipped_short']}")
    print(f"Failed generations: {summary['failed']}")
    print(f"Output path: {Path(args.output)}")


if __name__ == "__main__":
    main()
