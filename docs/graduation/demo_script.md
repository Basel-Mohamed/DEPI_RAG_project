# Demo Script

## 1. Start The Stack

Run Qdrant and the API with Docker Compose, or run Qdrant in Docker and start `uvicorn main:app --reload` from the Python environment.

## 2. Upload And Build Knowledge

Upload the sample overtime policy PDF through `POST /files`, then call `POST /files/build`. Show that the file status changes to built and that chunks are indexed in Qdrant.

## 3. Ask A Real Support Question

Ask: `What is the process for employees to request time in lieu for working on a public holiday?`

Expected answer: the user must submit a formal request by email within the same month, and approval is confirmed by the Project Manager and Direct Manager.

## 4. Show Sources

Point out that the response includes source path, page number, chunk index, score, and retrieval mode. This proves that the answer is grounded instead of free-form.

## 5. Submit Feedback

Submit positive or negative feedback through `POST /feedback`, then open the satisfaction endpoint.

## 6. Show Monitoring And MLOps

Open the Grafana dashboard JSON and MLflow comparison table. Mention the scheduled reindex script for embedding refresh.
