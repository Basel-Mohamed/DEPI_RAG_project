from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    APP_NAME: str
    APP_VERSION: str

    # Chunker settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Preprocessing image settings
    IMAGE_SCALE: float = 2.0
    IMAGE_FORMAT: str = "PNG"

    # Embedding settings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

settings = Settings()
