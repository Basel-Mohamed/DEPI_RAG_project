# Support Data EDA Report

## Dataset Snapshot

The current repository corpus is generated at `data/corpus.jsonl`. It contains 7 retrieval chunks from 1 source file(s), covering 3 page records and roughly 844 words. Current source coverage: pdfs\Overtime Policy v1.3.pdf.

## Observed Content Themes

The seed corpus is an HR/support policy document, so the dominant query families are policy eligibility, overtime approval, compensation rates, public holiday handling, on-call coverage, recording and reporting, and escalation approvals.

## Retrieval Implications

- Policy queries often ask for a process, which means chunk overlap is important because the answer may span a heading and following bullets.
- Numeric policy answers such as monthly overtime limits, shift percentages, and overtime multipliers require exact source grounding.
- The corpus includes operational terms such as PM, RM, Project Manager, Direct Manager, HR, and public holiday; hybrid retrieval is appropriate because user wording may mix exact acronyms with semantic paraphrases.

## Data Quality Findings

- The processed corpus preserves source path, page number, and chunk index metadata for citation and support traceability.
- Some extracted symbols show encoding artifacts from PDF extraction. They do not block retrieval but should be normalized before a larger production ingestion.
- The current graduation dataset is intentionally small. A production rollout should add historical tickets, FAQ pairs, product manuals, and support portal articles.

## Recommended Next EDA Additions

- Topic distribution by source type: ticket, FAQ, manual, portal article.
- Query-answer effectiveness labels from support agents.
- Duplicate and boilerplate removal rate after SHA-256 normalized text deduplication.
- PII detection counts before and after redaction.
