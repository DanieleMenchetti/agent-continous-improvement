"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Google / Gemini
    google_api_key: str = ""
    chat_model: str = "gemini-2.5-flash"

    # PostgreSQL (relational store for Q&A pairs + feedback metadata)
    database_url: str = "postgresql+psycopg2://app:app@postgres:5432/app"

    # Wiki (LLM-maintained markdown knowledge base built from ingested documents
    # and expert feedback)
    wiki_dir: str = "/data/wiki"


settings = Settings()
