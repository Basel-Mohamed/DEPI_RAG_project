# Preprocessing Pipeline

Current supported input is user-provided PDF files uploaded through `POST /files`.

Pipeline:

1. Validate file extension and PDF signature.
2. Store the PDF under `uploads/`.
3. Convert PDF pages with Docling.
4. Extract page text.
5. Clean whitespace and split text with `RecursiveCharacterTextSplitter`.
6. Attach metadata: source file id, page number, chunk index, and build id.
7. Embed chunks with FastEmbed.
8. Upsert dense and sparse vectors into Qdrant.
9. Delete stale chunks for the same source after successful rebuild.

Support data guidance:

- Product manuals and policy docs should be uploaded as PDFs.
- FAQs and ticket exports should be converted to PDF or curated into documents before upload.
- Preserve source names and sections in the document text so answers can cite useful context.
