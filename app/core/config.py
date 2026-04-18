from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="DEPI RAG Backend", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ORIGINS",
    )
    vector_store_provider: str = Field(
        default="inmemory",
        alias="VECTOR_STORE_PROVIDER",
    )
    llm_provider: str = Field(default="", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="stub", alias="EMBEDDING_PROVIDER")
    reranker_provider: str = Field(default="", alias="RERANKER_PROVIDER")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=400, alias="LLM_MAX_TOKENS")
    reranker_top_n: int | None = Field(default=None, alias="RERANKER_TOP_N")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_chat_deployment: str = Field(
        default="",
        alias="AZURE_OPENAI_CHAT_DEPLOYMENT",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
        alias="AZURE_OPENAI_API_VERSION",
    )
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    cohere_chat_model: str = Field(default="command-r-plus", alias="COHERE_CHAT_MODEL")
    cohere_rerank_model: str = Field(default="rerank-v3.5", alias="COHERE_RERANK_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return value
        return []

    @field_validator(
        "vector_store_provider",
        "llm_provider",
        "embedding_provider",
        "reranker_provider",
        mode="before",
    )
    @classmethod
    def normalize_provider_names(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return str(value).strip().lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
