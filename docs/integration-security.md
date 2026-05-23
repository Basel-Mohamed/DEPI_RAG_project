# Integration And Security

API security:

- All non-health endpoints support API-key protection.
- Set `AUTH_ENABLED=True` and provide `API_KEY`.
- Clients send `X-API-Key: <API_KEY>`.

Support portal integration:

- Upload curated support PDFs with `POST /files`.
- Trigger indexing with `POST /files/build`.
- Ask questions through `POST /ask`.
- Use `sources` in responses to show supporting text context.

Recommended production hardening:

- Store secrets in Azure Key Vault references.
- Restrict App Service inbound network access where possible.
- Add Azure AD JWT auth for user-specific authorization if the portal has identity requirements.
- Move uploads and artifacts to Azure Blob Storage for multi-instance deployments.
