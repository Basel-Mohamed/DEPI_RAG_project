# Business KPI Impact Analysis

## Target KPIs

| KPI | Baseline challenge | Expected project impact | Measurement source |
| --- | --- | --- | --- |
| First response time | Agents manually search policies and tickets | Lower response time through instant grounded answers | `/metrics` request latency and support portal timestamps |
| Deflection rate | Repeated policy questions reach agents | Self-service answers handle common questions before ticket creation | Portal analytics |
| Agent handle time | Agents re-read long documents for exact policy details | Source-backed snippets reduce lookup and drafting time | Ticketing system |
| Answer satisfaction | Users may receive inconsistent policy guidance | Feedback loop tracks positive and negative answer ratings | `/feedback/satisfaction` |
| Knowledge freshness | Manual re-indexing causes outdated answers | Scheduled reindex keeps embeddings aligned with documents | MLflow reindex runs |

## Impact Narrative

The project converts static support knowledge into a measurable support automation service. Its strongest near-term value is reducing repetitive lookup work for policy-style questions while preserving traceability through returned sources.

## Example KPI Targets For Pilot

- Reduce average first response time for covered knowledge-base questions by 40%.
- Reach at least 60% positive feedback during the pilot, then raise the threshold as the corpus expands.
- Keep average RAG response latency below 3 seconds for non-streaming requests.
- Re-index approved support documents every 24 hours.
- Review 100% of negative chatbot feedback during the first month.
