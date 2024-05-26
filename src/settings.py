from pathlib import Path

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    AZURE_AI_ENDPOINT: AnyHttpUrl
    AZURE_AI_SUBSCRIPTION_KEY: str
    AZURE_AI_VIDEO_INDEXER_KEY: str
    AZURE_AI_VIDEO_ACCOUNT_ID: str
    AZURE_AI_LANGUAGE_SUBSCRIPTION_KEY: str
    AZURE_AI_LANGUAGE_ENDPOINT: AnyHttpUrl
    AZURE_AI_LANGUAGE_PROJECT_NAME: str
    AZURE_AI_LANGUAGE_DEPLOYMENT_NAME: str

    BASE_DIR: Path = Path(__file__).resolve().parent.parent


settings = Settings()
