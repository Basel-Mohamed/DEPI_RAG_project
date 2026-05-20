from app.services.types import RetrievedContext

__all__ = [
    "AzureLlmService",
    "BaseLlmService",
    "CohereLlmService",
    "LlmServiceError",
    "RetrievedContext",
    "create_llm_service",
]


def __getattr__(name: str):
    if name == "create_llm_service":
        from app.services.llm.llm_factory import create_llm_service

        return create_llm_service

    if name == "AzureLlmService":
        from app.services.llm.providers.azure_llm import AzureLlmService

        return AzureLlmService

    if name == "CohereLlmService":
        from app.services.llm.providers.cohere_llm import CohereLlmService

        return CohereLlmService

    if name in {"BaseLlmService", "LlmServiceError"}:
        from app.services.llm.providers.base_llm import BaseLlmService, LlmServiceError

        return {"BaseLlmService": BaseLlmService, "LlmServiceError": LlmServiceError}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
