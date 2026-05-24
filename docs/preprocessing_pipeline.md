# Preprocessing Pipeline

## Overview

The preprocessing pipeline ingests documents from multiple formats, converts each source into page-like text records, splits text into retrieval-friendly chunks, redacts common PII, removes exact duplicate chunks, embeds the cleaned text, and upserts the resulting vectors and metadata into Qdrant for retrieval.

## Supported input formats

| Extension | Loader class | Text columns used | Notes |
| --- | --- | --- | --- |
| `.pdf` | `PdfLoader` | Docling main text by page | Uses the existing Docling extraction behavior and keeps page numbers from provenance. |
| `.csv` | `CsvLoader` | `subject`, `body`, `description`, `text`, `content`, `message` when present | Each row becomes one page. Non-text columns are preserved as metadata. Comma is tried first, then semicolon. |
| `.json` | `JsonLoader` | `answer`, with optional `question` prepended | Supports a single object or an array of objects. Remaining keys are preserved as metadata. |
| `.txt` | `TextLoader` | Entire file | The whole file is read as one page. |
| `.md` | `TextLoader` | Entire file | Markdown is read as plain text and chunked normally. |

## Chunking strategy

The pipeline uses `RecursiveCharacterTextSplitter` because it preserves larger semantic units first, then falls back to smaller separators when text is too long. The default `CHUNK_SIZE=1000` is a practical balance for retrieval, roughly 200-250 tokens in many documents, and fits comfortably inside the embedding model context. The default `CHUNK_OVERLAP=200` keeps nearby context across chunk boundaries, which helps answers that depend on text split between adjacent chunks. Tune these values through environment variables, for example `CHUNK_SIZE=1200` or `CHUNK_OVERLAP=150`, depending on document density and retrieval quality.

## Why Docling over PyMuPDF

Docling provides structure-aware extraction, handles tables and multi-column layouts, improves Arabic and RTL text ordering, and integrates cleanly with LangChain-oriented document processing. Those traits make it a better default for a RAG corpus than lower-level PDF text extraction with PyMuPDF.

## PII redaction

Tickets and support documents can contain customer PII that should not be indexed. The pipeline redacts email addresses, phone numbers, credit and debit card numbers, IPv4 addresses, and US ZIP codes before chunks are embedded. Regex-based redaction is used instead of Presidio because it keeps dependencies simple and is sufficient for the common PII types in this milestone. Disable it with `ENABLE_PII_REDACTION=False` when building a trusted internal-only corpus.

## Deduplication

Deduplication uses a SHA-256 hash of normalized text, where normalization lowercases the chunk and collapses whitespace. Exact-hash deduplication is intentionally simple and predictable, and it works well for ticket logs that repeat boilerplate footers, signatures, or disclaimers. Typical removal rates should be recorded from the EDA notebook here: [SEE EDA SECTION 6].

## Running the pipeline

API-based ingestion:

```bash
curl -X POST "http://localhost:8000/files" \
  -H "accept: application/json" \
  -F "file=@data/raw/example.pdf"

curl -X POST "http://localhost:8000/files/build" \
  -H "accept: application/json"
```

Batch CLI ingestion:

```bash
python scripts/build_corpus.py --input data/raw --output data/corpus.jsonl
```

## Configuration reference

| Env var | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk before overlap. |
| `CHUNK_OVERLAP` | `200` | Characters shared between adjacent chunks. |
| `ENABLE_PII_REDACTION` | `True` | Enables regex redaction before indexing. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Dense embedding model used for vector generation. |
| `RETRIEVAL_MODE` | `hybrid` | Retrieval strategy: `dense`, `sparse`, or `hybrid`. |
