import hashlib
import re


def deduplicate_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """
    Remove exact-duplicate chunks by SHA-256 hash of normalized text.
    Returns (unique_chunks, removed_count).
    Normalize text before hashing: lowercase + collapse whitespace.
    """

    seen_hashes: set[str] = set()
    unique_chunks: list[dict] = []

    for chunk in chunks:
        normalized_text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip().lower()
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)
        unique_chunks.append(chunk)

    return unique_chunks, len(chunks) - len(unique_chunks)
